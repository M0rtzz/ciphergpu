"""Local, per-deployment vLLM runtime management.

Model archives are only ever written below ``CIPHERGPU_MODEL_RUNTIME_DIR`` after
the confidential execution service has authenticated and decrypted them.  This
module deliberately does not accept archive paths from HTTP callers.
"""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import tarfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import httpx

from .errors import CipherGpuError


@dataclass
class RuntimeProcess:
    deployment_id: str
    model_dir: Path
    archive_path: Path
    port: int
    process: subprocess.Popen[bytes]
    log_path: Path


class ModelRuntimeManager:
    """Owns model plaintext directories and one vLLM child per deployment."""

    def __init__(self, root: str | None = None, vllm_command: str | None = None):
        self.root = Path(root or os.environ.get("CIPHERGPU_MODEL_RUNTIME_DIR", "/var/lib/ciphergpu/models"))
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.vllm_command = vllm_command or os.environ.get("CIPHERGPU_VLLM_COMMAND", "vllm")
        self.gpu_memory_utilization = os.environ.get("CIPHERGPU_VLLM_GPU_MEMORY_UTILIZATION", "0.10")
        self.max_model_len = os.environ.get("CIPHERGPU_VLLM_MAX_MODEL_LEN", "1024")
        self._runtimes: dict[str, RuntimeProcess] = {}
        self._last_logs: dict[str, str] = {}

    def archive_path(self, deployment_id: str) -> Path:
        return self._deployment_dir(deployment_id) / "model-package"

    def prepare_archive(self, deployment_id: str) -> Path:
        self.stop(deployment_id)
        self._last_logs.pop(deployment_id, None)
        directory = self._deployment_dir(deployment_id)
        if directory.exists():
            self._wipe(directory)
        directory.mkdir(parents=True, mode=0o700)
        return self.archive_path(deployment_id)

    def append_ciphertext_plaintext(self, deployment_id: str, plaintext: bytes) -> None:
        """Append one authenticated plaintext chunk to the runtime-only archive."""
        with self.archive_path(deployment_id).open("ab") as output:
            output.write(plaintext)

    def start(self, deployment_id: str, model_name: str, timeout_seconds: int) -> RuntimeProcess:
        archive = self.archive_path(deployment_id)
        if not archive.is_file() or archive.stat().st_size == 0:
            raise CipherGpuError("MODEL_PACKAGE_MISSING", "decrypted model package is unavailable")
        model_dir = self._deployment_dir(deployment_id) / "model"
        self._extract(archive, model_dir)
        model_dir = self._validate_model_layout(model_dir)
        port = self._free_port()
        # ``trust_remote_code`` is deliberately omitted: its vLLM default is
        # false and packages must not execute user-supplied Python at load time.
        command = [self.vllm_command, "serve", str(model_dir), "--host", "127.0.0.1", "--port", str(port),
                   "--served-model-name", model_name, "--gpu-memory-utilization", self.gpu_memory_utilization,
                   "--max-model-len", self.max_model_len, "--enforce-eager"]
        runtime_env = os.environ.copy()
        # Scientific Python libraries otherwise size their thread pools from
        # the host's CPU count. A vLLM worker plus 64 OpenBLAS threads can
        # exhaust the deliberately small container PID budget before CUDA is
        # initialized.
        runtime_env.update({
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        })
        log_path = self._deployment_dir(deployment_id) / "vllm.log"
        try:
            with log_path.open("ab", buffering=0) as log_output:
                process = subprocess.Popen(command, stdout=log_output, stderr=subprocess.STDOUT,
                                           cwd=str(model_dir), start_new_session=True, env=runtime_env)
        except FileNotFoundError as failure:
            raise CipherGpuError("VLLM_NOT_INSTALLED", "vLLM runtime is not installed on this node", 503) from failure
        runtime = RuntimeProcess(deployment_id, model_dir, archive, port, process, log_path)
        self._runtimes[deployment_id] = runtime
        try:
            self._wait_healthy(runtime, timeout_seconds)
            return runtime
        except Exception:
            self.stop(deployment_id)
            raise

    def endpoint(self, deployment_id: str) -> str | None:
        runtime = self._runtimes.get(deployment_id)
        return f"http://127.0.0.1:{runtime.port}/v1" if runtime and runtime.process.poll() is None else None

    def logs(self, deployment_id: str, max_bytes: int = 64 * 1024) -> str:
        log_path = self._deployment_dir(deployment_id) / "vllm.log"
        if not log_path.is_file():
            return self._last_logs.get(deployment_id, "")
        with log_path.open("rb") as source:
            source.seek(max(0, log_path.stat().st_size - max_bytes))
            return source.read(max_bytes).decode("utf-8", errors="replace")

    def stop(self, deployment_id: str, remove_plaintext: bool = True) -> None:
        runtime = self._runtimes.pop(deployment_id, None)
        if runtime and runtime.process.poll() is None:
            runtime.process.terminate()
            try:
                runtime.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                runtime.process.kill()
                runtime.process.wait(timeout=5)
        log_path = self._deployment_dir(deployment_id) / "vllm.log"
        if log_path.is_file():
            with log_path.open("rb") as source:
                source.seek(max(0, log_path.stat().st_size - 64 * 1024))
                self._last_logs[deployment_id] = source.read().decode("utf-8", errors="replace")
        if remove_plaintext:
            directory = self._deployment_dir(deployment_id)
            if directory.exists():
                self._wipe(directory)

    def pid(self, deployment_id: str) -> int | None:
        runtime = self._runtimes.get(deployment_id)
        return runtime.process.pid if runtime and runtime.process.poll() is None else None

    def _deployment_dir(self, deployment_id: str) -> Path:
        if not deployment_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for ch in deployment_id):
            raise CipherGpuError("CONTRACT_INVALID", "invalid deployment identifier")
        return self.root / deployment_id

    @staticmethod
    def _wipe(directory: Path) -> None:
        # Directories are created exclusively by this manager; do not follow links.
        if directory.is_symlink():
            raise CipherGpuError("MODEL_PACKAGE_INVALID", "runtime directory cannot be a symlink")
        shutil.rmtree(directory)

    @staticmethod
    def _safe_member(name: str) -> bool:
        path = Path(name)
        return bool(name) and not path.is_absolute() and ".." not in path.parts

    def _extract(self, archive: Path, destination: Path) -> None:
        destination.mkdir(mode=0o700)
        try:
            if zipfile.is_zipfile(archive):
                with zipfile.ZipFile(archive) as bundle:
                    infos = bundle.infolist()
                    if any(not self._safe_member(info.filename) or info.is_dir() is False and (info.external_attr >> 16) & 0o170000 == 0o120000 for info in infos):
                        raise CipherGpuError("MODEL_PACKAGE_INVALID", "archive contains unsafe path or symlink")
                    bundle.extractall(destination)
            else:
                with tarfile.open(archive, "r:*") as bundle:
                    members = bundle.getmembers()
                    if any(not self._safe_member(member.name) or member.issym() or member.islnk() or member.isdev() for member in members):
                        raise CipherGpuError("MODEL_PACKAGE_INVALID", "archive contains unsafe path or link")
                    bundle.extractall(destination, filter="data")
        except CipherGpuError:
            raise
        except (tarfile.TarError, zipfile.BadZipFile, OSError) as failure:
            raise CipherGpuError("MODEL_PACKAGE_INVALID", "model package cannot be extracted") from failure

    @staticmethod
    def _validate_model_layout(directory: Path) -> Path:
        # Accept a single enclosing directory, as Hugging Face snapshots commonly use one.
        entries = [item for item in directory.iterdir()]
        if len(entries) == 1 and entries[0].is_dir():
            directory = entries[0]
        if not (directory / "config.json").is_file() or not any(directory.glob("*.safetensors")):
            raise CipherGpuError("MODEL_PACKAGE_INVALID", "model package requires config.json and safetensors weights")
        if not any((directory / name).is_file() for name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")):
            raise CipherGpuError("MODEL_PACKAGE_INVALID", "model package requires tokenizer files")
        return directory

    @staticmethod
    def _free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    @staticmethod
    def _wait_healthy(runtime: RuntimeProcess, timeout_seconds: int) -> None:
        # Small models can still need more than one minute on the first CUDA
        # initialization. Keep the configured upper bound while guaranteeing a
        # practical startup window for the bundled TinyLlama acceptance model.
        deadline = time.monotonic() + min(max(timeout_seconds, 180), 300)
        with httpx.Client(timeout=2, trust_env=False) as client:
            while time.monotonic() < deadline:
                if runtime.process.poll() is not None:
                    raise CipherGpuError("VLLM_START_FAILED", "vLLM process exited while loading the model", 503)
                try:
                    if client.get(f"http://127.0.0.1:{runtime.port}/health").status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                time.sleep(1)
        raise CipherGpuError("VLLM_START_TIMEOUT", "vLLM model loading timed out", 503)
