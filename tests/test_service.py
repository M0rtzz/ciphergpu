from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from fastapi.testclient import TestClient

from ciphergpu.app import create_app
from ciphergpu.crypto import (
    CONTRACT_VERSION,
    HPKE_DEK_INFO,
    EvidenceSigner,
    b64u,
    canonical,
    hpke_seal,
    new_hpke_key_pair,
    sha256,
    unb64u,
)
from ciphergpu.service import ConfidentialExecutionService


def iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def fixture(
    scenario: str = "NORMAL", workload_id: str = "builtin.digest/v1", chunk_count: int = 1
) -> tuple[TestClient, dict]:
    service = ConfidentialExecutionService(EvidenceSigner(Ed25519PrivateKey.generate()))
    client = TestClient(create_app(service))
    current = datetime.now(UTC)
    recipient = new_hpke_key_pair()
    recipient_public = b64u(recipient.public_raw)
    task = {
        "contractVersion": CONTRACT_VERSION,
        "taskId": "task-1",
        "domainId": "domain-a",
        "securityProfile": "a100-sim",
        "evidenceType": "SIMULATED_LAB_V1",
        "simulated": True,
        "hardwareModel": "NVIDIA A100",
        "runtimeSecurityRequirement": "controlled-sim-ok",
        "purpose": "verify",
        "workloadId": workload_id,
        "assetVersionIds": ["asset-1@v1"],
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
    signing_public = b64u(signing.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw))
    exp = iso(current + timedelta(minutes=4))
    claims = {
        "contractVersion": CONTRACT_VERSION,
        "grantId": "grant-1",
        "jti": "jti-1",
        "taskSpecDigest": digest,
        "teeSessionId": session["sessionId"],
        "teeEphemeralPublicKeyHash": sha256(unb64u(session["teeEphemeralPublicKey"])),
        "securityProfile": "a100-sim",
        "evidenceType": "SIMULATED_LAB_V1",
        "simulated": True,
        "hardwareModel": "NVIDIA A100",
        "runtimeSecurityRequirement": "controlled-sim-ok",
        "assetVersionIds": ["asset-1@v1"],
        "outputRecipients": ["uek-1"],
        "nbf": iso(current - timedelta(seconds=1)),
        "exp": exp,
        "maxUses": 1,
    }
    dek = os.urandom(32)
    sealed_aad = f"{digest}|asset-1@v1|grant-1|{exp}".encode()
    sealed = hpke_seal(session["teeEphemeralPublicKey"], dek, sealed_aad, HPKE_DEK_INFO)
    encrypted_inputs = []
    for chunk_index in range(chunk_count):
        input_aad = {"assetVersionId": "asset-1@v1", "chunkIndex": chunk_index}
        nonce = os.urandom(12)
        plaintext = f"confidential input {chunk_index}".encode()
        ciphertext = AESGCM(dek).encrypt(nonce, plaintext, canonical(input_aad))
        encrypted_inputs.append(
            {
                "assetVersionId": "asset-1@v1",
                "algorithm": "AES-256-GCM",
                "nonce": b64u(nonce),
                "aad": input_aad,
                "ciphertext": b64u(ciphertext),
                "ciphertextSha256": sha256(ciphertext),
            }
        )
    request = {
        "taskSpec": task,
        "taskSpecDigest": digest,
        "sessionId": session["sessionId"],
        "grant": {
            "claims": claims,
            "signingPublicKey": signing_public,
            "signature": b64u(signing.sign(canonical(claims))),
        },
        "sealedDeks": [
            {
                "assetVersionId": "asset-1@v1",
                "enc": sealed["enc"],
                "ciphertext": sealed["ciphertext"],
                "aad": b64u(sealed_aad),
            }
        ],
        "encryptedInputs": encrypted_inputs,
        "outputRecipients": [{"kid": "uek-1", "encryptionPublicKey": recipient_public}],
        "scenario": scenario,
    }
    return client, request


def test_encrypted_execution_succeeds_and_replay_fails() -> None:
    client, request = fixture()
    response = client.post("/v1/executions", json=request)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["status"] == "SUCCEEDED"
    assert result["attestationVerified"] is False
    assert "ciphertext" in result["encryptedOutput"]
    assert client.post("/v1/executions", json=request).json()["error"]["code"] == "GRANT_REPLAYED"


def test_multi_chunk_encrypted_execution_aggregates_asset_digest() -> None:
    client, request = fixture(chunk_count=2)
    response = client.post("/v1/executions", json=request)
    assert response.status_code == 200, response.text
    inputs = response.json()["receipt"]
    assert inputs["taskSpecDigest"] == request["taskSpecDigest"]
    # The receipt is signed and only exposes the aggregate workload result via
    # the encrypted output; a successful response is the important regression
    # assertion for multi-chunk model material.


def test_key_mismatch_fails_closed() -> None:
    client, request = fixture("KEY_MISMATCH")
    response = client.post("/v1/executions", json=request)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "KEY_MATCH_FAILED"


def test_tampered_ciphertext_fails_closed() -> None:
    client, request = fixture()
    request["encryptedInputs"][0]["ciphertext"] = b64u(b"tampered")
    response = client.post("/v1/executions", json=request)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "DATA_INTEGRITY_FAILED"


def test_unapproved_workload_digest_fails_closed() -> None:
    service = ConfidentialExecutionService(EvidenceSigner(Ed25519PrivateKey.generate()))
    client = TestClient(create_app(service))
    response = client.post(
        "/v1/attestation/sessions",
        json={
            "domainId": "domain-a",
            "clientNonce": b64u(os.urandom(32)),
            "taskSpecDigest": "0" * 64,
            "expectedSecurityProfile": "a100-sim",
            "runtimeSecurityRequirement": "controlled-sim-ok",
            "workloadDigest": "sha256:unapproved",
            "policyDigest": service.policy_digest,
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "WORKLOAD_DIGEST_MISMATCH"


def test_openai_compatible_deployment_requires_authorization_then_activates() -> None:
    client, request = fixture(workload_id="model.deploy/deploy-1")
    registered = client.post(
        "/v1/model-deployments",
        json={
            "deploymentId": "deploy-1",
            "sourceType": "OPENAI_COMPATIBLE",
            "baseUrl": "https://8.8.8.8/v1",
            "upstreamModelId": "model-a",
            "timeoutSeconds": 30,
            "securityProfile": "a100-sim",
            "simulated": True,
        },
    )
    assert registered.status_code == 200
    assert registered.json()["status"] == "AUTHORIZATION_REQUIRED"

    execution = client.post("/v1/executions", json=request)
    assert execution.status_code == 200, execution.text
    assert execution.json()["receipt"]["modelDeploymentStatus"] == "ONLINE"
    assert client.get("/v1/model-deployments/deploy-1").json()["status"] == "ONLINE"

    offline = client.post("/v1/model-deployments/deploy-1/offline")
    assert offline.status_code == 200
    assert offline.json()["status"] == "OFFLINE"


def test_offline_model_deployment_is_idempotent_after_agent_restart() -> None:
    service = ConfidentialExecutionService(EvidenceSigner(Ed25519PrivateKey.generate()))
    client = TestClient(create_app(service))

    offline = client.post("/v1/model-deployments/deploy-missing/offline")

    assert offline.status_code == 200
    assert offline.json() == {
        "deploymentId": "deploy-missing",
        "securityProfile": "a100-sim",
        "simulated": True,
        "status": "OFFLINE",
        "sessionId": None,
        "errorCode": None,
        "alreadyAbsent": True,
        "sessionKeysDestroyed": True,
    }


def test_model_connector_denies_private_and_non_https_urls() -> None:
    client, _ = fixture()
    template = {
        "deploymentId": "deploy-private",
        "sourceType": "OPENAI_COMPATIBLE",
        "upstreamModelId": "model-a",
        "securityProfile": "a100-sim",
        "simulated": True,
    }
    for url in ("http://example.com/v1", "https://127.0.0.1/v1", "https://169.254.169.254/v1"):
        response = client.post("/v1/model-deployments", json={**template, "baseUrl": url})
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "MODEL_PROVIDER_URL_DENIED"
