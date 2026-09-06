"""Fixed-entry training worker. It never imports code from customer packages."""
from __future__ import annotations

import json
import hashlib
import os
import sys
from pathlib import Path
from typing import Any


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _dataset_files(directory: Path, split: str) -> tuple[str, list[str]]:
    manifest = json.loads((directory / "dataset-manifest.json").read_text(encoding="utf-8"))
    names = manifest["splits"].get(split, [])
    if not names:
        return str(manifest["format"]), []
    return str(manifest["format"]), [str(directory / name) for name in names]


def _load_split(directory: Path, split: str):
    from datasets import load_dataset

    kind, files = _dataset_files(directory, split)
    if not files:
        return None
    loader = "json" if kind in {"json", "jsonl"} else kind
    return load_dataset(loader, data_files=files, split="train")


def _require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is disabled")


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _result_files(directory: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.relative_to(directory)),
            "sha256": _digest(path),
            "size": path.stat().st_size,
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "result-manifest.json"
    ]


def _runtime_receipt(global_steps: int) -> dict[str, Any]:
    import torch
    import transformers
    import datasets
    import peft
    import trl

    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    return {
        "cudaDevice": device,
        "gpuName": str(properties.name),
        "gpuUuid": str(getattr(properties, "uuid", "unavailable")),
        "peakGpuMemoryBytes": torch.cuda.max_memory_allocated(device),
        "cudaVersion": str(torch.version.cuda),
        "pytorchVersion": str(torch.__version__),
        "transformersVersion": str(transformers.__version__),
        "datasetsVersion": str(datasets.__version__),
        "peftVersion": str(peft.__version__),
        "trlVersion": str(trl.__version__),
        "globalSteps": global_steps,
        "runtimeImageDigest": os.environ.get(
            "CIPHERGPU_TRAINING_IMAGE_DIGEST", "sha256:builtin-training-v1"
        ),
    }


def _write_result_manifests(
    spec: dict[str, Any], model_output: Path, data_output: Path, receipt: dict[str, Any]
) -> None:
    common = {
        "formatVersion": "1",
        "jobId": spec["jobId"],
        "adapterId": spec["adapterId"],
        "trainingConfig": spec["trainingConfig"],
        "runtime": receipt,
    }
    _write_json(
        model_output / "result-manifest.json",
        {**common, "resultType": "MODEL", "files": _result_files(model_output)},
    )
    _write_json(
        data_output / "result-manifest.json",
        {**common, "resultType": "DATA", "files": _result_files(data_output)},
    )


class _ProgressCallback:
    def __init__(self, path: Path, total_epochs: float):
        self.path = path
        self.total_epochs = max(1.0, total_epochs)

    def on_log(self, args, state, control, logs=None, **kwargs):
        epoch = float(state.epoch or 0)
        _write_json(
            self.path,
            {
                "progress": min(95, max(1, round(epoch / self.total_epochs * 90))),
                "currentEpoch": epoch,
                "metrics": logs or {},
            },
        )


def _sequence_classification(spec: dict[str, Any]) -> None:
    import numpy as np
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    _require_cuda()
    config = spec["trainingConfig"]
    model_dir = Path(spec["modelDir"])
    train_dir = Path(spec["trainDataDir"])
    validation_dir = Path(spec["validationDataDir"] or spec["trainDataDir"])
    train = _load_split(train_dir, "train")
    validation = _load_split(validation_dir, "validation")
    if validation is None:
        validation = _load_split(train_dir, "validation")
    if train is None or validation is None:
        raise RuntimeError("train and validation splits are required")
    text_column = str(config.get("textColumn", "sentence"))
    label_column = str(config.get("labelColumn", "label"))
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
        num_labels=int(config.get("numLabels", 2)),
    )

    def tokenize(batch):
        values = tokenizer(
            batch[text_column],
            truncation=True,
            max_length=int(config.get("maxLength", 256)),
        )
        values["labels"] = batch[label_column]
        return values

    train = train.map(tokenize, batched=True)
    validation = validation.map(tokenize, batched=True)
    epochs = float(config.get("epochs", 2))
    output_dir = Path(spec["outputDir"])
    checkpoint_dir = Path(spec["checkpointDir"])
    progress_path = Path(spec["progressPath"])
    use_bf16 = str(config.get("mixedPrecision", "bf16")) == "bf16" and torch.cuda.is_bf16_supported()

    class Callback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            _ProgressCallback(progress_path, epochs).on_log(args, state, control, logs)

    args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=epochs,
        learning_rate=float(config.get("learningRate", 2e-5)),
        per_device_train_batch_size=int(config.get("trainBatchSize", 16)),
        per_device_eval_batch_size=int(config.get("evalBatchSize", 32)),
        weight_decay=float(config.get("weightDecay", 0.01)),
        warmup_ratio=float(config.get("warmupRatio", 0.1)),
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=10,
        max_steps=int(config.get("maxSteps", -1)),
        bf16=use_bf16,
        fp16=not use_bf16,
        report_to=[],
        seed=int(config.get("seed", 42)),
        dataloader_num_workers=0,
        disable_tqdm=True,
    )

    def metrics(value):
        predictions = np.argmax(value.predictions, axis=-1)
        return {"accuracy": float((predictions == value.label_ids).mean())}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train,
        eval_dataset=validation,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=metrics,
        callbacks=[Callback()],
    )
    train_result = trainer.train()
    evaluation = trainer.evaluate()
    model_output = output_dir / "model"
    data_output = output_dir / "data"
    model_output.mkdir(parents=True)
    data_output.mkdir(parents=True)
    trainer.save_model(str(model_output))
    tokenizer.save_pretrained(model_output)
    prediction = trainer.predict(validation)
    labels = np.argmax(prediction.predictions, axis=-1).tolist()
    with (data_output / "predictions.jsonl").open("w", encoding="utf-8") as output:
        for index, label in enumerate(labels):
            output.write(json.dumps({"row": index, "prediction": label}) + "\n")
    all_metrics = {**train_result.metrics, **evaluation}
    receipt = _runtime_receipt(int(trainer.state.global_step))
    all_metrics["runtime"] = receipt
    _write_json(data_output / "metrics.json", all_metrics)
    _write_json(model_output / "training-metrics.json", all_metrics)
    _write_result_manifests(spec, model_output, data_output, receipt)
    _write_json(progress_path, {"progress": 96, "currentEpoch": epochs, "metrics": all_metrics})


def _causal_lm_sft(spec: dict[str, Any]) -> None:
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    _require_cuda()
    config = spec["trainingConfig"]
    model_dir = Path(spec["modelDir"])
    train_dir = Path(spec["trainDataDir"])
    validation_dir = Path(spec["validationDataDir"] or spec["trainDataDir"])
    train = _load_split(train_dir, "train")
    validation = _load_split(validation_dir, "validation")
    if validation is None:
        validation = _load_split(train_dir, "validation")
    if train is None or validation is None:
        raise RuntimeError("train and validation splits are required")
    messages_column = str(config.get("messagesColumn", "messages"))
    if messages_column not in train.column_names or messages_column not in validation.column_names:
        raise RuntimeError("the conversational dataset messages column is missing")

    def conversational_only(dataset):
        if messages_column != "messages":
            dataset = dataset.rename_column(messages_column, "messages")
        return dataset.select_columns(["messages"])

    train = conversational_only(train)
    validation = conversational_only(validation)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    if not tokenizer.chat_template:
        raise RuntimeError("the tokenizer package requires a chat template")
    dropped_train = 0
    dropped_validation = 0
    if bool(config.get("assistantOnlyLoss", True)):
        sample = train[0].get("messages")
        if not isinstance(sample, list):
            raise RuntimeError("the conversational dataset requires a messages list")
        rendered = tokenizer.apply_chat_template(
            sample,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
        mask = rendered.get("assistant_masks") or rendered.get("assistant_tokens_mask")
        if not mask or not any(mask):
            raise RuntimeError("the chat template cannot produce an assistant token mask")
        maximum = int(config.get("maxSequenceLength", 1024))

        def has_assistant_tokens(example):
            value = tokenizer.apply_chat_template(
                example["messages"],
                tokenize=True,
                return_dict=True,
                return_assistant_tokens_mask=True,
            )
            item_mask = value.get("assistant_masks") or value.get("assistant_tokens_mask") or []
            return any(item_mask[:maximum])

        train_before = len(train)
        validation_before = len(validation)
        train = train.filter(has_assistant_tokens, desc="Validating train assistant masks")
        validation = validation.filter(
            has_assistant_tokens, desc="Validating validation assistant masks"
        )
        dropped_train = train_before - len(train)
        dropped_validation = validation_before - len(validation)
        if len(train) == 0 or len(validation) == 0:
            raise RuntimeError("no trainable assistant tokens remain after truncation")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
    )
    lora = config.get("lora", {})
    peft_config = LoraConfig(
        r=int(lora.get("r", 16)),
        lora_alpha=int(lora.get("alpha", 32)),
        lora_dropout=float(lora.get("dropout", 0.05)),
        target_modules=lora.get("targetModules", "all-linear"),
        bias=str(lora.get("bias", "none")),
        task_type="CAUSAL_LM",
    )
    epochs = float(config.get("epochs", 1))
    output_dir = Path(spec["outputDir"])
    checkpoint_dir = Path(spec["checkpointDir"])
    progress_path = Path(spec["progressPath"])

    class Callback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            _ProgressCallback(progress_path, epochs).on_log(args, state, control, logs)

    args = SFTConfig(
        output_dir=str(checkpoint_dir),
        num_train_epochs=epochs,
        learning_rate=float(config.get("learningRate", 2e-4)),
        per_device_train_batch_size=int(config.get("trainBatchSize", 1)),
        per_device_eval_batch_size=int(config.get("evalBatchSize", 1)),
        gradient_accumulation_steps=int(config.get("gradientAccumulationSteps", 8)),
        gradient_checkpointing=bool(config.get("gradientCheckpointing", True)),
        bf16=True,
        max_length=int(config.get("maxSequenceLength", 1024)),
        assistant_only_loss=bool(config.get("assistantOnlyLoss", True)),
        packing=bool(config.get("packing", False)),
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=5,
        max_steps=int(config.get("maxSteps", -1)),
        report_to=[],
        seed=int(config.get("seed", 42)),
        dataset_num_proc=1,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train,
        eval_dataset=validation,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[Callback()],
    )
    result = trainer.train()
    evaluation = trainer.evaluate()
    model_output = output_dir / "model"
    data_output = output_dir / "data"
    model_output.mkdir(parents=True)
    data_output.mkdir(parents=True)
    trainer.model.save_pretrained(model_output, safe_serialization=True)
    tokenizer.save_pretrained(model_output)
    metrics = {**result.metrics, **evaluation}
    metrics["droppedTrainRowsWithoutAssistantTokens"] = dropped_train
    metrics["droppedValidationRowsWithoutAssistantTokens"] = dropped_validation
    receipt = _runtime_receipt(int(trainer.state.global_step))
    metrics["runtime"] = receipt
    _write_json(model_output / "training-metrics.json", metrics)
    _write_json(data_output / "metrics.json", metrics)
    _write_result_manifests(spec, model_output, data_output, receipt)
    _write_json(progress_path, {"progress": 96, "currentEpoch": epochs, "metrics": metrics})


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m ciphergpu.training_worker JOB_SPEC")
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    adapter = spec.get("adapterId")
    if adapter == "hf-sequence-classification-v1":
        _sequence_classification(spec)
    elif adapter == "hf-causal-lm-sft-lora-v1":
        _causal_lm_sft(spec)
    else:
        raise RuntimeError("training adapter is not allowed")


if __name__ == "__main__":
    main()
