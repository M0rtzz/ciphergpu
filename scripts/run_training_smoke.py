#!/usr/bin/env python3
"""Run an offline CUDA smoke test against prepared model and dataset packages."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def extract(package: Path, destination: Path) -> Path:
    destination.mkdir(parents=True)
    with tarfile.open(package, "r:*") as bundle:
        bundle.extractall(destination, filter="data")
    entries = list(destination.iterdir())
    return entries[0] if len(entries) == 1 and entries[0].is_dir() else destination


def validate_manifest(directory: Path) -> dict:
    manifest = json.loads((directory / "result-manifest.json").read_text())
    for item in manifest["files"]:
        path = directory / item["path"]
        assert path.stat().st_size == item["size"]
        assert digest(path) == item["sha256"]
    assert manifest["runtime"]["globalSteps"] > 0
    assert manifest["runtime"]["gpuName"]
    assert manifest["runtime"]["peakGpuMemoryBytes"] > 0
    return manifest


def run(kind: str, model_package: Path, data_package: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="ciphergpu-smoke-") as temporary:
        root = Path(temporary)
        model = extract(model_package, root / "model-input")
        data = extract(data_package, root / "data-input")
        output = root / "output"
        checkpoint = root / "checkpoints"
        output.mkdir()
        checkpoint.mkdir()
        if kind == "small":
            adapter = "hf-sequence-classification-v1"
            config = {
                "epochs": 1,
                "maxSteps": 2,
                "learningRate": 0.00002,
                "trainBatchSize": 8,
                "evalBatchSize": 32,
                "mixedPrecision": "bf16",
                "textColumn": "sentence",
                "labelColumn": "label",
                "numLabels": 2,
                "maxLength": 128,
                "seed": 42,
            }
        else:
            adapter = "hf-causal-lm-sft-lora-v1"
            config = {
                "epochs": 1,
                "maxSteps": 1,
                "learningRate": 0.0002,
                "trainBatchSize": 1,
                "evalBatchSize": 1,
                "mixedPrecision": "bf16",
                "maxSequenceLength": 512,
                "gradientAccumulationSteps": 1,
                "gradientCheckpointing": True,
                "assistantOnlyLoss": True,
                "packing": False,
                "lora": {
                    "r": 8,
                    "alpha": 16,
                    "dropout": 0.05,
                    "targetModules": "all-linear",
                    "bias": "none",
                },
                "seed": 42,
            }
        spec = {
            "jobId": f"smoke-{kind}",
            "adapterId": adapter,
            "trainingConfig": config,
            "modelDir": str(model),
            "trainDataDir": str(data),
            "validationDataDir": None,
            "checkpointDir": str(checkpoint),
            "outputDir": str(output),
            "progressPath": str(root / "progress.json"),
        }
        spec_path = root / "job-spec.json"
        spec_path.write_text(json.dumps(spec))
        environment = os.environ.copy()
        environment.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "WANDB_DISABLED": "true",
                "TOKENIZERS_PARALLELISM": "false",
            }
        )
        subprocess.run(
            [sys.executable, "-m", "ciphergpu.training_worker", str(spec_path)],
            check=True,
            env=environment,
        )
        model_manifest = validate_manifest(output / "model")
        data_manifest = validate_manifest(output / "data")
        if kind == "small":
            from transformers import AutoModelForSequenceClassification

            AutoModelForSequenceClassification.from_pretrained(
                output / "model", local_files_only=True, trust_remote_code=False
            )
            input_weights = next(model.glob("*.safetensors"))
            output_weights = next((output / "model").glob("*.safetensors"))
            assert digest(input_weights) != digest(output_weights)
        else:
            import torch
            from peft import PeftConfig, PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer

            configuration = PeftConfig.from_pretrained(output / "model")
            assert configuration.peft_type.value == "LORA"
            assert next((output / "model").glob("*.safetensors")).stat().st_size > 0
            base = AutoModelForCausalLM.from_pretrained(
                model,
                local_files_only=True,
                trust_remote_code=False,
                dtype=torch.bfloat16,
            ).to("cuda")
            adapted = PeftModel.from_pretrained(base, output / "model").eval()
            tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
            inputs = tokenizer("GPU training verification", return_tensors="pt").to("cuda")
            with torch.no_grad():
                generated = adapted.generate(**inputs, max_new_tokens=1)
            assert generated.shape[-1] == inputs["input_ids"].shape[-1] + 1
            metrics = json.loads((output / "data" / "metrics.json").read_text())
            assert math.isfinite(float(metrics["eval_loss"]))
        report = {
            "adapterId": adapter,
            "modelFiles": len(model_manifest["files"]),
            "dataFiles": len(data_manifest["files"]),
            "runtime": model_manifest["runtime"],
            "plaintextRootRemovedAfterContext": True,
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        shutil.rmtree(output)
        assert not output.exists()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("small", "llm"))
    parser.add_argument("model_package", type=Path)
    parser.add_argument("data_package", type=Path)
    arguments = parser.parse_args()
    run(arguments.kind, arguments.model_package, arguments.data_package)


if __name__ == "__main__":
    main()
