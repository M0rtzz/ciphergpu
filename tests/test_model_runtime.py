import io
import tarfile
from pathlib import Path

import pytest

from ciphergpu.errors import CipherGpuError
from ciphergpu.model_runtime import ModelRuntimeManager


def write_package(path: Path, entries: dict[str, bytes]) -> None:
    with tarfile.open(path, "w") as archive:
        for name, content in entries.items():
            item = tarfile.TarInfo(name)
            item.size = len(content)
            archive.addfile(item, io.BytesIO(content))


def test_validates_and_extracts_huggingface_package(tmp_path: Path) -> None:
    manager = ModelRuntimeManager(str(tmp_path / "runtime"))
    archive = manager.prepare_archive("deploy_abc")
    write_package(archive, {
        "snapshot/config.json": b"{}",
        "snapshot/model.safetensors": b"weights",
        "snapshot/tokenizer.json": b"{}",
    })
    manager._extract(archive, archive.parent / "model")
    model_dir = manager._validate_model_layout(archive.parent / "model")
    assert model_dir.name == "snapshot"


def test_rejects_archive_path_traversal(tmp_path: Path) -> None:
    manager = ModelRuntimeManager(str(tmp_path / "runtime"))
    archive = manager.prepare_archive("deploy_safe")
    write_package(archive, {"../outside": b"no"})
    with pytest.raises(CipherGpuError, match="unsafe"):
        manager._extract(archive, archive.parent / "model")
