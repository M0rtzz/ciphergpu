from __future__ import annotations

import io
import json
import os
import tarfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from fastapi.testclient import TestClient

from ciphergpu.app import create_app
from ciphergpu.crypto import (
    CONTRACT_VERSION,
    HPKE_DEK_INFO,
    HPKE_RESULT_DEK_INFO,
    EvidenceSigner,
    b64u,
    canonical,
    content_open,
    content_seal,
    hpke_open,
    hpke_seal,
    new_hpke_key_pair,
    sha256,
    unb64u,
)
from ciphergpu.service import ConfidentialExecutionService
from ciphergpu.models import TrainingJobPrepareRequest
from ciphergpu.training_runtime import TrainingRuntimeManager


def iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def package(files: dict[str, bytes], manifest_name: str, manifest: dict) -> bytes:
    manifest["files"] = [
        {"path": name, "sha256": sha256(value), "size": len(value)}
        for name, value in files.items()
    ]
    values = {**files, manifest_name: json.dumps(manifest).encode()}
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as bundle:
        for name, value in values.items():
            info = tarfile.TarInfo(name)
            info.size = len(value)
            bundle.addfile(info, io.BytesIO(value))
    return output.getvalue()


class FakeTrainingRuntime(TrainingRuntimeManager):
    def start(self, job_id: str, adapter_id: str, training_config: dict):
        directory = self._job_dir(job_id)
        model = directory / "output" / "model"
        data = directory / "output" / "data"
        model.mkdir(parents=True)
        data.mkdir(parents=True)
        (model / "config.json").write_text("{}")
        (model / "model.safetensors").write_bytes(b"trained-weights")
        (data / "metrics.json").write_text('{"accuracy":0.9}')
        (directory / "progress.json").write_text(
            '{"progress":96,"currentEpoch":2,"metrics":{"accuracy":0.9}}'
        )
        return object()

    def wait(self, job_id: str, timeout: float | None = None) -> int:
        return 0


def training_fixture(tmp_path: Path):
    service = ConfidentialExecutionService(EvidenceSigner(Ed25519PrivateKey.generate()))
    service._training_runtime = FakeTrainingRuntime(str(tmp_path / "training"))
    client = TestClient(create_app(service))
    current = datetime.now(UTC)
    recipient = new_hpke_key_pair()
    recipient_public = b64u(recipient.public_raw)
    assets = ["model-1@v1", "data-1@v1"]
    config = {
        "epochs": 2,
        "learningRate": 0.00002,
        "trainBatchSize": 16,
        "evalBatchSize": 32,
        "mixedPrecision": "bf16",
        "seed": 42,
        "maxRuntimeSeconds": 3600,
        "textColumn": "sentence",
        "labelColumn": "label",
        "numLabels": 2,
        "maxLength": 256,
        "weightDecay": 0.01,
        "warmupRatio": 0.1,
    }
    config_hash = sha256(canonical(config))
    task = {
        "contractVersion": CONTRACT_VERSION,
        "taskId": "train-job-1",
        "domainId": "domain-a",
        "securityProfile": "a100-sim",
        "evidenceType": "SIMULATED_LAB_V1",
        "simulated": True,
        "hardwareModel": "NVIDIA A100",
        "runtimeSecurityRequirement": "controlled-sim-ok",
        "purpose": "train",
        "workloadId": (
            "training/train-job-1/hf-sequence-classification-v1/" + config_hash
        ),
        "assetVersionIds": assets,
        "outputRecipients": ["uek-1"],
        "attestationPolicyId": "policy/a100-sim/v1",
        "workloadDigest": service.workload_digest,
        "policyDigest": service.policy_digest,
        "egressPolicy": "deny-all",
        "issuedAt": iso(current),
        "expiresAt": iso(current + timedelta(minutes=5)),
    }
    digest = sha256(canonical(task))
    session = client.post(
        "/v1/attestation/sessions",
        json={
            "domainId": "domain-a",
            "clientNonce": b64u(os.urandom(32)),
            "taskSpecDigest": digest,
            "expectedSecurityProfile": "a100-sim",
            "runtimeSecurityRequirement": "controlled-sim-ok",
            "workloadDigest": service.workload_digest,
            "policyDigest": service.policy_digest,
        },
    ).json()
    signing = Ed25519PrivateKey.generate()
    signing_public = b64u(
        signing.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    )
    exp = iso(current + timedelta(minutes=4))
    claims = {
        "contractVersion": CONTRACT_VERSION,
        "grantId": "training-grant-1",
        "jti": "training-jti-1",
        "taskSpecDigest": digest,
        "teeSessionId": session["sessionId"],
        "teeEphemeralPublicKeyHash": sha256(unb64u(session["teeEphemeralPublicKey"])),
        "securityProfile": "a100-sim",
        "evidenceType": "SIMULATED_LAB_V1",
        "simulated": True,
        "hardwareModel": "NVIDIA A100",
        "runtimeSecurityRequirement": "controlled-sim-ok",
        "assetVersionIds": assets,
        "outputRecipients": ["uek-1"],
        "nbf": iso(current - timedelta(seconds=1)),
        "exp": exp,
        "maxUses": 1,
    }
    model_package = package(
        {"config.json": b"{}", "model.safetensors": b"initial-weights"},
        "model-manifest.json",
        {
            "formatVersion": "1",
            "adapterId": "hf-sequence-classification-v1",
            "taskType": "SEQUENCE_CLASSIFICATION",
        },
    )
    dataset_package = package(
        {"train.jsonl": b'{"sentence":"good","label":1}\n'},
        "dataset-manifest.json",
        {
            "formatVersion": "1",
            "format": "jsonl",
            "taskType": "SEQUENCE_CLASSIFICATION",
            "splits": {"train": ["train.jsonl"]},
        },
    )
    encrypted_inputs = []
    raw_ciphertexts = {}
    for slot, asset, plaintext in (
        ("model", assets[0], model_package),
        ("train-data", assets[1], dataset_package),
    ):
        dek = os.urandom(32)
        envelope_id = f"env-{slot}"
        aad = {"assetVersionId": asset, "chunkIndex": 0}
        nonce = os.urandom(12)
        ciphertext = content_seal(
            dek, envelope_id, "AES-256-GCM", nonce, plaintext, aad
        )
        raw_ciphertexts[slot] = ciphertext
        sealed_aad = f"{digest}|{asset}|training-grant-1|{exp}".encode()
        sealed = hpke_seal(
            session["teeEphemeralPublicKey"], dek, sealed_aad, HPKE_DEK_INFO
        )
        manifest = {
            "format": "ds-envelope/v2",
            "envelopeId": envelope_id,
            "algorithm": "AES-256-GCM",
            "chunks": [
                {
                    "index": 0,
                    "plaintextLength": len(plaintext),
                    "nonce": b64u(nonce),
                    "sha256": sha256(ciphertext),
                    "aad": aad,
                }
            ],
        }
        encrypted_inputs.append(
            {
                "slot": slot,
                "assetVersionId": asset,
                "packageFormat": "TAR",
                "manifestHash": sha256(canonical(manifest)),
                "manifest": manifest,
                "ownerSigningPublicKey": signing_public,
                "ownerSignature": b64u(signing.sign(canonical(manifest))),
                "sealedDek": {
                    "assetVersionId": asset,
                    "enc": sealed["enc"],
                    "ciphertext": sealed["ciphertext"],
                    "aad": b64u(sealed_aad),
                },
                "chunks": [
                    {
                        "index": 0,
                        "format": "ds-envelope/v2",
                        "envelopeId": envelope_id,
                        "implementationVersion": "1",
                        "algorithm": "AES-256-GCM",
                        "nonce": b64u(nonce),
                        "aad": aad,
                        "ciphertextSha256": sha256(ciphertext),
                    }
                ],
            }
        )
    request = {
        "jobId": "train-job-1",
        "taskSpec": task,
        "taskSpecDigest": digest,
        "sessionId": session["sessionId"],
        "grants": [
            {
                "claims": claims,
                "signingPublicKey": signing_public,
                "signature": b64u(signing.sign(canonical(claims))),
            }
        ],
        "inputs": encrypted_inputs,
        "outputRecipient": {
            "kid": "uek-1",
            "encryptionPublicKey": recipient_public,
        },
        "adapterId": "hf-sequence-classification-v1",
        "trainingConfig": config,
        "trainingConfigHash": config_hash,
    }
    return client, service, request, raw_ciphertexts, recipient


def test_training_inputs_outputs_replay_and_cleanup(tmp_path: Path) -> None:
    client, service, request, ciphertexts, recipient = training_fixture(tmp_path)
    prepared = client.post("/v1/training-jobs", json=request)
    assert prepared.status_code == 200, prepared.text
    assert prepared.json()["status"] == "STAGING_INPUTS"
    for slot in ("model", "train-data"):
        uploaded = client.put(
            f"/v1/training-jobs/train-job-1/inputs/{slot}/chunks/0",
            content=ciphertexts[slot],
        )
        assert uploaded.status_code == 200, uploaded.text
        finalized = client.post(
            f"/v1/training-jobs/train-job-1/inputs/{slot}/finalize"
        )
        assert finalized.status_code == 200, finalized.text
        assert not any(service._training_jobs["train-job-1"].inputs[slot].dek)
    started = client.post("/v1/training-jobs/train-job-1/start")
    assert started.status_code == 200, started.text
    for _ in range(100):
        status = client.get("/v1/training-jobs/train-job-1").json()
        if status["status"] == "OUTPUT_READY":
            break
        time.sleep(0.01)
    assert status["status"] == "OUTPUT_READY", status
    outputs = client.get("/v1/training-jobs/train-job-1/outputs").json()["outputs"]
    model_manifest = outputs["result-model"]["manifest"]
    key_envelope = model_manifest["keyEnvelope"]
    result_dek = hpke_open(
        recipient.private,
        key_envelope["enc"],
        key_envelope["ciphertext"],
        unb64u(key_envelope["aad"]),
        HPKE_RESULT_DEK_INFO,
    )
    plaintext = bytearray()
    for chunk in model_manifest["chunks"]:
        response = client.get(
            f"/v1/training-jobs/train-job-1/outputs/result-model/chunks/{chunk['index']}"
        )
        assert response.status_code == 200
        plaintext.extend(
            content_open(
                result_dek,
                model_manifest["envelopeId"],
                model_manifest["algorithm"],
                chunk["nonce"],
                b64u(response.content),
                chunk["aad"],
            )
        )
    with tarfile.open(fileobj=io.BytesIO(plaintext), mode="r:gz") as bundle:
        assert bundle.extractfile("model/model.safetensors").read() == b"trained-weights"
    acknowledged = client.post("/v1/training-jobs/train-job-1/outputs/ack")
    assert acknowledged.json()["status"] == "COMPLETED"
    assert not (tmp_path / "training" / "train-job-1").exists()
    replay = client.post("/v1/training-jobs", json=request)
    assert replay.status_code == 400
    assert replay.json()["error"]["code"] == "GRANT_REPLAYED"


def test_training_tampered_chunk_fails_and_cleans_plaintext(tmp_path: Path) -> None:
    client, _, request, _, _ = training_fixture(tmp_path)
    assert client.post("/v1/training-jobs", json=request).status_code == 200
    response = client.put(
        "/v1/training-jobs/train-job-1/inputs/model/chunks/0",
        content=b"tampered",
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "DATA_INTEGRITY_FAILED"
    status = client.get("/v1/training-jobs/train-job-1").json()
    assert status["status"] == "FAILED"
    assert not (tmp_path / "training" / "train-job-1" / "input").exists()


def test_service_shutdown_clears_training_keys_and_directory(tmp_path: Path) -> None:
    _client, service, request, _ciphertexts, _recipient = training_fixture(tmp_path)
    service.prepare_training_job(TrainingJobPrepareRequest.model_validate(request))
    job = service._training_jobs["train-job-1"]
    assert any(any(state.dek) for state in job.inputs.values())
    assert (tmp_path / "training" / "train-job-1").exists()

    service.close()

    assert all(not any(state.dek) for state in job.inputs.values())
    assert job.status == "FAILED"
    assert job.error_code == "RUNTIME_SHUTDOWN"
    assert not (tmp_path / "training" / "train-job-1").exists()
    assert not service._sessions
