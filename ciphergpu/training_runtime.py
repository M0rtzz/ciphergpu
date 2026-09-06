"""Isolated plaintext lifecycle and worker management for confidential training."""
from __future__ import annotations

import json
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import CipherGpuError


ALLOWED_ADAPTERS = {
    "hf-sequence-classification-v1",
    "hf-causal-lm-sft-lora-v1",
}
ALLOWED_DATA_SUFFIXES = {".parquet", ".json", ".jsonl", ".csv", ".txt"}


@dataclass
class TrainingProcess:
    job_id: str
    process: subprocess.Popen[bytes]
    log_path: Path
    progress_path: Path


class TrainingRuntimeManager:
    def __init__(self, root: str | None = None):
        self.root = Path(
            root
            or os.environ.get(
                "CIPHERGPU_TRAINING_RUNTIME_DIR", "/var/lib/ciphergpu/training"
            )
        )
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._processes: dict[str, TrainingProcess] = {}
        self._last_logs: dict[str, str] = {}

    def prepare_job(self, job_id: str) -> Path:
        directory = self._job_dir(job_id)
        self.cancel(job_id)
        if directory.exists():
            self._wipe(directory)
        for child in ("packages", "input", "checkpoints", "output", "encrypted", "logs"):
            (directory / child).mkdir(parents=True, mode=0o700)
        return directory

    def append_input(self, job_id: str, slot: str, plaintext: bytes) -> None:
        path = self._package_path(job_id, slot)
        maximum = int(os.environ.get("CIPHERGPU_TRAINING_MAX_INPUT_BYTES", str(20 * 1024**3)))
        if (path.stat().st_size if path.exists() else 0) + len(plaintext) > maximum:
            raise CipherGpuError("TRAINING_PACKAGE_TOO_LARGE", "training package exceeds its size limit")
        with path.open("ab") as output:
            output.write(plaintext)

    def finalize_input(
        self, job_id: str, slot: str, package_format: str, expected_adapter: str
    ) -> Path:
        archive = self._package_path(job_id, slot)
        if not archive.is_file() or archive.stat().st_size == 0:
            raise CipherGpuError("TRAINING_INPUT_MISSING", f"{slot} package is empty")
        destination = self._job_dir(job_id) / "input" / slot
        destination.mkdir(mode=0o700)
        self._extract(archive, destination, package_format)
        directory = self._single_root(destination)
        if slot == "model":
            self._validate_model(directory, expected_adapter)
        else:
            self._validate_dataset(directory, expected_adapter, slot)
        archive.unlink(missing_ok=True)
        return directory

    def start(
        self,
        job_id: str,
        adapter_id: str,
        training_config: dict[str, Any],
    ) -> TrainingProcess:
        if adapter_id not in ALLOWED_ADAPTERS:
            raise CipherGpuError("TRAINING_ADAPTER_DENIED", "training adapter is not allowed")
        directory = self._job_dir(job_id)
        if not (directory / "input" / "model").exists() or not (
            directory / "input" / "train-data"
        ).exists():
            raise CipherGpuError("TRAINING_INPUT_MISSING", "model and train-data are required")
        spec = {
            "jobId": job_id,
            "adapterId": adapter_id,
            "trainingConfig": training_config,
            "modelDir": str(self._single_root(directory / "input" / "model")),
            "trainDataDir": str(
                self._single_root(directory / "input" / "train-data")
            ),
            "validationDataDir": str(
                self._single_root(directory / "input" / "validation-data")
            )
            if (directory / "input" / "validation-data").exists()
            else None,
            "checkpointDir": str(directory / "checkpoints"),
            "outputDir": str(directory / "output"),
            "progressPath": str(directory / "progress.json"),
        }
        spec_path = directory / "job-spec.json"
        spec_path.write_text(json.dumps(spec, separators=(",", ":")), encoding="utf-8")
        log_path = directory / "logs" / "training.log"
        runtime_env = os.environ.copy()
        runtime_env.update(
            {
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "WANDB_DISABLED": "true",
            }
        )
        with log_path.open("ab", buffering=0) as log_output:
            process = subprocess.Popen(
                [sys.executable, "-m", "ciphergpu.training_worker", str(spec_path)],
                stdout=log_output,
                stderr=subprocess.STDOUT,
                cwd=str(directory),
                start_new_session=True,
                env=runtime_env,
            )
        value = TrainingProcess(job_id, process, log_path, directory / "progress.json")
        self._processes[job_id] = value
        return value

    def wait(self, job_id: str, timeout: float | None = None) -> int:
        runtime = self._processes.get(job_id)
        if runtime is None:
            raise CipherGpuError("TRAINING_NOT_RUNNING", "training worker is not running")
        try:
            return runtime.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as failure:
            raise CipherGpuError(
                "TRAINING_TIMEOUT", "training worker exceeded its time limit"
            ) from failure

    def progress(self, job_id: str) -> dict[str, Any]:
        runtime = self._processes.get(job_id)
        path = runtime.progress_path if runtime else self._job_dir(job_id) / "progress.json"
        if not path.is_file():
            return {"progress": 0, "currentEpoch": 0, "metrics": {}}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def logs(self, job_id: str, max_bytes: int = 64 * 1024) -> str:
        path = self._job_dir(job_id) / "logs" / "training.log"
        if not path.is_file():
            return self._last_logs.get(job_id, "")
        with path.open("rb") as source:
            source.seek(max(0, path.stat().st_size - max_bytes))
            return source.read(max_bytes).decode("utf-8", errors="replace")

    def cancel(self, job_id: str) -> None:
        runtime = self._processes.pop(job_id, None)
        if runtime and runtime.process.poll() is None:
            runtime.process.terminate()
            try:
                runtime.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                runtime.process.kill()
                runtime.process.wait(timeout=5)
        if runtime and runtime.log_path.is_file():
            self._last_logs[job_id] = self.logs(job_id)

    def cleanup_plaintext(self, job_id: str) -> None:
        directory = self._job_dir(job_id)
        for name in ("packages", "input", "checkpoints", "output", "job-spec.json"):
            path = directory / name
            if path.is_dir():
                self._wipe(path)
            elif path.exists():
                path.unlink()

    def cleanup_all(self, job_id: str) -> None:
        self.cancel(job_id)
        directory = self._job_dir(job_id)
        if directory.exists():
            self._last_logs[job_id] = self.logs(job_id)
            self._wipe(directory)

    def output_directory(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "output"

    def encrypted_directory(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "encrypted"

    def _job_dir(self, job_id: str) -> Path:
        if not job_id or any(
            ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for ch in job_id
        ):
            raise CipherGpuError("CONTRACT_INVALID", "invalid training job identifier")
        return self.root / job_id

    def _package_path(self, job_id: str, slot: str) -> Path:
        if slot not in {"model", "train-data", "validation-data"}:
            raise CipherGpuError("CONTRACT_INVALID", "invalid training input slot")
        return self._job_dir(job_id) / "packages" / f"{slot}.package"

    @staticmethod
    def _single_root(directory: Path) -> Path:
        entries = list(directory.iterdir()) if directory.exists() else []
        return entries[0] if len(entries) == 1 and entries[0].is_dir() else directory

    @staticmethod
    def _safe_member(name: str) -> bool:
        path = Path(name)
        return bool(name) and not path.is_absolute() and ".." not in path.parts

    def _extract(self, archive: Path, destination: Path, package_format: str) -> None:
        maximum = int(os.environ.get("CIPHERGPU_TRAINING_MAX_INPUT_BYTES", str(20 * 1024**3)))
        maximum_files = int(os.environ.get("CIPHERGPU_TRAINING_MAX_FILES", "10000"))
        try:
            if package_format == "ZIP":
                with zipfile.ZipFile(archive) as bundle:
                    members = bundle.infolist()
                    if any(
                        not self._safe_member(item.filename)
                        or (not item.is_dir() and (item.external_attr >> 16) & 0o170000 == 0o120000)
                        for item in members
                    ):
                        raise CipherGpuError(
                            "TRAINING_PACKAGE_INVALID", "archive contains unsafe path or link"
                        )
                    if len(members) > maximum_files or sum(item.file_size for item in members) > maximum:
                        raise CipherGpuError(
                            "TRAINING_PACKAGE_TOO_LARGE", "archive expansion exceeds its limit"
                        )
                    bundle.extractall(destination)
            elif package_format in {"TAR", "TAR_GZ"}:
                with tarfile.open(archive, "r:*") as bundle:
                    members = bundle.getmembers()
                    if any(
                        not self._safe_member(item.name)
                        or item.issym()
                        or item.islnk()
                        or item.isdev()
                        for item in members
                    ):
                        raise CipherGpuError(
                            "TRAINING_PACKAGE_INVALID", "archive contains unsafe path or link"
                        )
                    if len(members) > maximum_files or sum(item.size for item in members) > maximum:
                        raise CipherGpuError(
                            "TRAINING_PACKAGE_TOO_LARGE", "archive expansion exceeds its limit"
                        )
                    bundle.extractall(destination, filter="data")
            else:
                raise CipherGpuError("TRAINING_PACKAGE_INVALID", "unsupported package format")
        except CipherGpuError:
            raise
        except (OSError, tarfile.TarError, zipfile.BadZipFile) as failure:
            raise CipherGpuError(
                "TRAINING_PACKAGE_INVALID", "training package cannot be extracted"
            ) from failure

    @staticmethod
    def _load_manifest(directory: Path, name: str) -> dict[str, Any]:
        path = directory / name
        if not path.is_file():
            raise CipherGpuError("TRAINING_PACKAGE_INVALID", f"{name} is required")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as failure:
            raise CipherGpuError("TRAINING_PACKAGE_INVALID", f"{name} is invalid") from failure
        if not isinstance(value, dict) or value.get("formatVersion") != "1":
            raise CipherGpuError("TRAINING_PACKAGE_INVALID", f"{name} format is invalid")
        for item in value.get("files", []):
            relative = str(item.get("path", ""))
            target = directory / relative
            if not TrainingRuntimeManager._safe_member(relative) or not target.is_file():
                raise CipherGpuError("TRAINING_PACKAGE_INVALID", "manifest file is missing")
            digest_value = hashlib.sha256()
            with target.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest_value.update(chunk)
            digest = digest_value.hexdigest()
            if digest != item.get("sha256") or target.stat().st_size != int(item.get("size", -1)):
                raise CipherGpuError("TRAINING_PACKAGE_INVALID", "manifest file digest mismatch")
        return value

    @staticmethod
    def _validate_model(directory: Path, expected_adapter: str) -> None:
        manifest = TrainingRuntimeManager._load_manifest(directory, "model-manifest.json")
        if manifest.get("adapterId") != expected_adapter or expected_adapter not in ALLOWED_ADAPTERS:
            raise CipherGpuError("TRAINING_ADAPTER_DENIED", "model adapter is not allowed")
        expected_task = (
            "CAUSAL_LM"
            if expected_adapter == "hf-causal-lm-sft-lora-v1"
            else "SEQUENCE_CLASSIFICATION"
        )
        if manifest.get("taskType") != expected_task:
            raise CipherGpuError("TRAINING_PACKAGE_INVALID", "model task type is incompatible")
        if not (directory / "config.json").is_file() or not any(
            directory.glob("*.safetensors")
        ):
            raise CipherGpuError(
                "TRAINING_PACKAGE_INVALID", "model requires config.json and safetensors"
            )
        forbidden = {".py", ".pyc", ".so", ".dll", ".dylib", ".sh", ".exe"}
        if any(path.suffix.lower() in forbidden for path in directory.rglob("*")):
            raise CipherGpuError("TRAINING_PACKAGE_INVALID", "executable model content is denied")

    @staticmethod
    def _validate_dataset(directory: Path, expected_adapter: str, slot: str) -> None:
        manifest = TrainingRuntimeManager._load_manifest(directory, "dataset-manifest.json")
        splits = manifest.get("splits")
        required_split = "validation" if slot == "validation-data" else "train"
        expected_task = (
            "CAUSAL_LM_SFT"
            if expected_adapter == "hf-causal-lm-sft-lora-v1"
            else "SEQUENCE_CLASSIFICATION"
        )
        if manifest.get("taskType") != expected_task:
            raise CipherGpuError("TRAINING_PACKAGE_INVALID", "dataset task type is incompatible")
        if not isinstance(splits, dict) or not splits.get(required_split):
            raise CipherGpuError(
                "TRAINING_PACKAGE_INVALID", f"dataset {required_split} split is required"
            )
        for files in splits.values():
            if not isinstance(files, list):
                raise CipherGpuError("TRAINING_PACKAGE_INVALID", "dataset split is invalid")
            for name in files:
                path = directory / str(name)
                if (
                    not TrainingRuntimeManager._safe_member(str(name))
                    or path.suffix.lower() not in ALLOWED_DATA_SUFFIXES
                    or not path.is_file()
                ):
                    raise CipherGpuError("TRAINING_PACKAGE_INVALID", "dataset shard is invalid")

    @staticmethod
    def _wipe(directory: Path) -> None:
        if directory.is_symlink():
            raise CipherGpuError("TRAINING_PACKAGE_INVALID", "runtime path cannot be a symlink")
        shutil.rmtree(directory)
