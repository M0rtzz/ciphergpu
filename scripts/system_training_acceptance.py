#!/usr/bin/env python3
"""Exercise the deployed SecretPad -> CipherGPU confidential training workflow."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import ssl
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from ciphergpu.crypto import (
    CONTRACT_VERSION,
    HPKE_DEK_INFO,
    HPKE_RESULT_DEK_INFO,
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


def utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class Platform:
    def __init__(self, base_url: str, username: str, password: str):
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self.ssl_context = (
            ssl._create_unverified_context() if self.base_url.startswith("https://") else None
        )

    def login(self, role: str) -> str:
        response = self.json(
            "POST",
            "/api/login",
            {
                "name": self.username,
                "passwordHash": hashlib.sha256(self.password.encode()).hexdigest(),
                "endRole": role,
            },
        )
        return str(response["token"])

    def request(
        self,
        method: str,
        path: str,
        token: str | None = None,
        payload: Any | None = None,
        content_type: str = "application/json",
        expect_success: bool = True,
        timeout: int = 300,
        extra_headers: dict[str, str] | None = None,
    ) -> tuple[int, bytes]:
        data = None
        if payload is not None:
            data = canonical(payload) if content_type == "application/json" else payload
        headers = {"Content-Type": content_type}
        if token:
            headers["User-Token"] = token
        headers.update(extra_headers or {})
        request = urllib.request.Request(
            self.base_url + path, method=method, headers=headers, data=data
        )
        try:
            with urllib.request.urlopen(
                request, timeout=timeout, context=self.ssl_context
            ) as response:
                status, body = response.status, response.read()
        except urllib.error.HTTPError as failure:
            status, body = failure.code, failure.read()
        if expect_success and status != 200:
            raise RuntimeError(f"{method} {path} failed with HTTP {status}: {body[:500]!r}")
        return status, body

    def json(
        self,
        method: str,
        path: str,
        payload: Any | None = None,
        token: str | None = None,
        expect_success: bool = True,
        timeout: int = 300,
    ) -> Any:
        status, raw = self.request(
            method, path, token, payload, expect_success=expect_success, timeout=timeout
        )
        body = json.loads(raw)
        success = status == 200 and body.get("status", {}).get("code") == 0
        if expect_success and not success:
            raise RuntimeError(f"{method} {path} was rejected: {body}")
        if not expect_success:
            return success, body
        return body.get("data")


def manifest_binding(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "format": manifest["format"],
        "envelopeId": manifest["envelopeId"],
        "contentEncryptionAlgorithm": manifest["algorithm"],
        "implementationVersion": manifest["contentEncryption"]["implementationVersion"],
        "domainId": manifest["domainId"],
        "publicKeyId": manifest["publicKeyId"],
        "publicKeyVersion": manifest["publicKeyVersion"],
        "originalSize": manifest["originalSize"],
        "chunks": [
            {
                "index": item["index"],
                "plaintextLength": item["plaintextLength"],
                "sha256": item["sha256"],
            }
            for item in manifest["chunks"]
        ],
    }


def upload_asset(
    platform: Platform,
    token: str,
    package: Path,
    asset_type: str,
    name: str,
    kid: str,
    encryption_public_key: str,
    signing_key: Ed25519PrivateKey,
) -> tuple[dict[str, Any], bytearray]:
    chunk_size = 32 * 1024 * 1024
    algorithm = "AES-256-GCM"
    domain_id = "a100-domain-a"
    envelope_id = f"env_acceptance_{os.urandom(8).hex()}"
    dek = bytearray(os.urandom(32))
    expected = (package.stat().st_size + chunk_size - 1) // chunk_size
    session = platform.json(
        "POST",
        "/api/v1alpha1/confidential-assets/upload-sessions",
        {
            "assetType": asset_type,
            "sourceType": "UPLOAD",
            "name": name,
            "description": "自动化 CipherGPU 训练验收资产",
            "originalFileName": package.name,
            "originalSize": package.stat().st_size,
            "domainId": domain_id,
            "algorithm": algorithm,
            "expectedChunks": expected,
        },
        token,
    )
    chunks: list[dict[str, Any]] = []
    cipher_size = 0
    with package.open("rb") as source:
        for index in range(expected):
            plaintext = bytearray(source.read(chunk_size))
            nonce = os.urandom(12)
            aad = {
                "format": "ds-envelope/v2",
                "envelopeId": envelope_id,
                "contentEncryptionAlgorithm": algorithm,
                "implementationVersion": "1",
                "domainId": domain_id,
                "publicKeyId": kid,
                "publicKeyVersion": 1,
                "chunkIndex": index,
                "plaintextLength": len(plaintext),
            }
            ciphertext = bytearray(
                content_seal(bytes(dek), envelope_id, algorithm, nonce, bytes(plaintext), aad)
            )
            digest = sha256(ciphertext)
            path = (
                "/api/v1alpha1/confidential-assets/upload-sessions/"
                f"{session['uploadSessionId']}/chunks?index={index}"
            )
            _, response = platform.request(
                "POST",
                path,
                token,
                bytes(ciphertext),
                "application/octet-stream",
                extra_headers={"X-Cipher-SHA256": digest},
            )
            if json.loads(response).get("status", {}).get("code") != 0:
                raise RuntimeError("ciphertext chunk upload was rejected")
            chunks.append(
                {
                    "index": index,
                    "plaintextLength": len(plaintext),
                    "nonce": b64u(nonce),
                    "sha256": digest,
                    "aad": aad,
                }
            )
            cipher_size += len(ciphertext)
            plaintext[:] = b"\x00" * len(plaintext)
            ciphertext[:] = b"\x00" * len(ciphertext)
    binding = {
        "format": "ds-envelope/v2",
        "envelopeId": envelope_id,
        "contentEncryptionAlgorithm": algorithm,
        "implementationVersion": "1",
        "domainId": domain_id,
        "publicKeyId": kid,
        "publicKeyVersion": 1,
        "originalSize": package.stat().st_size,
        "chunks": [
            {
                "index": item["index"],
                "plaintextLength": item["plaintextLength"],
                "sha256": item["sha256"],
            }
            for item in chunks
        ],
    }
    key_envelope = hpke_seal(
        encryption_public_key,
        bytes(dek),
        canonical(binding),
        b"ds-envelope/v2:asset-dek",
    )
    manifest = {
        **binding,
        "cipherHash": sha256(canonical(binding)),
        "keyEnvelope": {
            "recipientKid": kid,
            **key_envelope,
            "info": b64u(b"ds-envelope/v2:asset-dek"),
            "aadHash": sha256(canonical(binding)),
        },
        "algorithm": algorithm,
        "cipherSize": cipher_size,
        "chunkSize": chunk_size,
        "contentEncryption": {
            "algorithm": algorithm,
            "keyDerivation": "HKDF-SHA256",
            "implementationVersion": "1",
            "keySize": 32,
            "nonceSize": 12,
            "tagSize": 16,
        },
        "chunks": chunks,
    }
    result = platform.json(
        "POST",
        f"/api/v1alpha1/confidential-assets/upload-sessions/{session['uploadSessionId']}/commit",
        {
            "manifest": manifest,
            "manifestHash": sha256(canonical(manifest)),
            "ownerSigningPublicKey": b64u(
                signing_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
            ),
            "ownerSignature": b64u(signing_key.sign(canonical(manifest))),
        },
        token,
    )
    return result, dek


def decrypt_result(
    platform: Platform,
    token: str,
    asset_id: str,
    private_key: Any,
    destination: Path,
) -> dict[str, Any]:
    manifest = platform.json(
        "GET", f"/api/v1alpha1/confidential-assets/{asset_id}/download-manifest", token=token
    )
    binding = manifest_binding(manifest)
    envelope = manifest["keyEnvelope"]
    dek = bytearray(
        hpke_open(
            private_key,
            envelope["enc"],
            envelope["ciphertext"],
            canonical(binding),
            HPKE_RESULT_DEK_INFO,
        )
    )
    try:
        with destination.open("wb") as output:
            for chunk in manifest["chunks"]:
                _, ciphertext = platform.request(
                    "GET",
                    f"/api/v1alpha1/confidential-assets/{asset_id}/chunks/{chunk['index']}",
                    token,
                )
                if sha256(ciphertext) != chunk["sha256"]:
                    raise RuntimeError("result ciphertext digest mismatch")
                plaintext = bytearray(
                    content_open(
                        bytes(dek),
                        manifest["envelopeId"],
                        manifest["algorithm"],
                        chunk["nonce"],
                        b64u(ciphertext),
                        chunk["aad"],
                    )
                )
                output.write(plaintext)
                plaintext[:] = b"\x00" * len(plaintext)
    finally:
        dek[:] = b"\x00" * len(dek)
    return manifest


def run(arguments: argparse.Namespace) -> None:
    platform = Platform(arguments.base_url, arguments.username, arguments.password)
    client = platform.login("CLIENT")
    center = platform.login("CENTER")
    user_keys = new_hpke_key_pair()
    signing = Ed25519PrivateKey.generate()
    kid = f"acceptance-{os.urandom(7).hex()}"
    public_key = b64u(user_keys.public_raw)
    signing_public = b64u(signing.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw))
    proof = {
        "kid": kid,
        "encryptionPublicKey": public_key,
        "signingPublicKey": signing_public,
    }
    platform.json(
        "POST",
        "/api/v1alpha1/crypto/identities",
        {**proof, "proofOfPossession": b64u(signing.sign(canonical(proof)))},
        client,
    )
    model, model_dek = upload_asset(
        platform,
        client,
        arguments.model,
        "MODEL",
        "E2E DistilBERT 模型包",
        kid,
        public_key,
        signing,
    )
    data, data_dek = upload_asset(
        platform,
        client,
        arguments.data,
        "DATA",
        "E2E SST-2 数据包",
        kid,
        public_key,
        signing,
    )
    denied, _ = platform.json(
        "POST",
        "/api/v1alpha1/confidential-training-tasks",
        {"taskName": "role-denied"},
        client,
        expect_success=False,
    )
    if denied:
        raise RuntimeError("CLIENT unexpectedly created a node training task")
    config = {
        "epochs": 1,
        "maxSteps": 2,
        "learningRate": 0.00002,
        "trainBatchSize": 8,
        "evalBatchSize": 32,
        "mixedPrecision": "bf16",
        "seed": 42,
        "maxRuntimeSeconds": 600,
        "textColumn": "sentence",
        "labelColumn": "label",
        "numLabels": 2,
        "maxLength": 128,
        "weightDecay": 0.01,
        "warmupRatio": 0.1,
    }
    task = platform.json(
        "POST",
        "/api/v1alpha1/confidential-training-tasks",
        {
            "taskName": "E2E CipherGPU DistilBERT 训练",
            "purpose": "验证双资产授权、GPU 训练、结果加密和客户解密",
            "computeNode": "confidential-hust",
            "dataAssetVersionId": data["assetVersionId"],
            "modelAssetVersionId": model["assetVersionId"],
            "adapterId": "hf-sequence-classification-v1",
            "trainingConfig": config,
        },
        center,
    )
    for request_id in (task["dataRequestId"], task["modelRequestId"]):
        platform.json(
            "POST",
            f"/api/v1alpha1/confidential-use-requests/{request_id}/decision",
            {"action": "APPROVE", "comment": "自动化端到端验收"},
            client,
        )
    task = platform.json(
        "POST",
        f"/api/v1alpha1/confidential-training-tasks/{task['taskId']}/prepare",
        {"clientNonce": b64u(os.urandom(32))},
        center,
    )
    task = platform.json(
        "GET",
        f"/api/v1alpha1/confidential-training-tasks/{task['taskId']}",
        token=client,
    )
    now = datetime.now(UTC)
    expiry = utc(
        min(
            datetime.fromisoformat(task["taskSpec"]["expiresAt"].replace("Z", "+00:00")),
            datetime.fromisoformat(task["attestation"]["expiresAt"].replace("Z", "+00:00")),
            now + timedelta(minutes=3),
        )
        - timedelta(seconds=1)
    )
    grant_id = f"grant-{os.urandom(8).hex()}"
    versions = [item["assetVersionId"] for item in task["inputManifests"]]
    claims = {
        "contractVersion": CONTRACT_VERSION,
        "grantId": grant_id,
        "jti": f"jti-{os.urandom(10).hex()}",
        "taskSpecDigest": task["taskSpecDigest"],
        "teeSessionId": task["attestation"]["sessionId"],
        "teeEphemeralPublicKeyHash": sha256(
            unb64u(task["attestation"]["teeEphemeralPublicKey"])
        ),
        "securityProfile": "a100-sim",
        "evidenceType": "SIMULATED_LAB_V1",
        "simulated": True,
        "hardwareModel": "NVIDIA A100",
        "runtimeSecurityRequirement": "controlled-sim-ok",
        "assetVersionIds": versions,
        "outputRecipients": [kid],
        "nbf": utc(now - timedelta(seconds=1)),
        "exp": expiry,
        "maxUses": 1,
    }
    dek_by_version = {
        model["assetVersionId"]: model_dek,
        data["assetVersionId"]: data_dek,
    }
    sealed_deks = []
    for version in versions:
        aad = f"{task['taskSpecDigest']}|{version}|{grant_id}|{expiry}".encode()
        sealed_deks.append(
            {
                "assetVersionId": version,
                **hpke_seal(
                    task["attestation"]["teeEphemeralPublicKey"],
                    bytes(dek_by_version[version]),
                    aad,
                    HPKE_DEK_INFO,
                ),
                "aad": b64u(aad),
            }
        )
    platform.json(
        "POST",
        f"/api/v1alpha1/confidential-training-tasks/{task['taskId']}/key-releases",
        {
            "grant": {
                "claims": claims,
                "signingPublicKey": signing_public,
                "signature": b64u(signing.sign(canonical(claims))),
            },
            "sealedDeks": sealed_deks,
        },
        client,
    )
    for value in dek_by_version.values():
        value[:] = b"\x00" * len(value)
    task = platform.json(
        "POST",
        f"/api/v1alpha1/confidential-training-tasks/{task['taskId']}/start",
        {},
        center,
        timeout=600,
    )
    deadline = time.monotonic() + 600
    while task["status"] not in {"OUTPUT_READY", "FAILED", "CANCELLED"}:
        if time.monotonic() >= deadline:
            raise RuntimeError("training did not finish before timeout")
        time.sleep(2)
        task = platform.json(
            "GET",
            f"/api/v1alpha1/confidential-training-tasks/{task['taskId']}",
            token=center,
        )
    if task["status"] != "OUTPUT_READY":
        raise RuntimeError(f"training failed: {task}")
    task = platform.json(
        "POST",
        f"/api/v1alpha1/confidential-training-tasks/{task['taskId']}/outputs/collect",
        {},
        center,
        timeout=600,
    )
    if task["status"] != "COMPLETED" or task.get("cleanupStatus") != "COMPLETED":
        raise RuntimeError(f"output collection did not complete cleanup: {task}")
    success, _ = platform.json(
        "GET",
        f"/api/v1alpha1/confidential-assets/{task['resultModelAssetId']}/download-manifest",
        token=center,
        expect_success=False,
    )
    if success:
        raise RuntimeError("CENTER unexpectedly obtained a result key envelope")
    with tempfile.TemporaryDirectory(prefix="ciphergpu-system-acceptance-") as temporary:
        archive = Path(temporary) / "result-model.tar.gz"
        manifest = decrypt_result(
            platform, client, task["resultModelAssetId"], user_keys.private, archive
        )
        with tarfile.open(archive, "r:gz") as bundle:
            names = set(bundle.getnames())
            if "model/result-manifest.json" not in names or not any(
                name.endswith("model.safetensors") for name in names
            ):
                raise RuntimeError("decrypted result model package is incomplete")
    print(
        json.dumps(
            {
                "taskId": task["taskId"],
                "status": task["status"],
                "cleanupStatus": task["cleanupStatus"],
                "resultModelAssetId": task["resultModelAssetId"],
                "resultDataAssetId": task["resultDataAssetId"],
                "resultManifestHash": sha256(canonical(manifest)),
                "centerResultEnvelopeDenied": True,
                "clientResultDecryption": "PASSED",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="https://127.0.0.1:39189")
    parser.add_argument("--username", default=os.environ.get("SECRETPAD_USER_NAME"), required=False)
    parser.add_argument("--password", default=os.environ.get("SECRETPAD_PASSWORD"), required=False)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    arguments = parser.parse_args()
    if not arguments.username or not arguments.password:
        parser.error("username and password are required through arguments or environment")
    run(arguments)


if __name__ == "__main__":
    main()
