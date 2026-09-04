from __future__ import annotations

from fastapi.testclient import TestClient

from ciphergpu.crypto import CONTRACT_VERSION
from ciphergpu.sim_attestation import app

client = TestClient(app)


def simulation_claims() -> dict[str, object]:
    return {
        "contractVersion": CONTRACT_VERSION,
        "runtimeMode": "SIMULATION",
        "attestationVerified": False,
        "securityProfile": "a100-sim",
        "evidenceType": "SIMULATED_LAB_V1",
        "simulated": True,
        "hardwareModel": "NVIDIA A100",
    }


def test_evidence_signing_accepts_only_attestation_evidence() -> None:
    evidence = {
        **simulation_claims(),
        "clientNonce": "nonce",
        "taskSpecDigest": "a" * 64,
        "teeEphemeralPublicKeyHash": "b" * 64,
        "workloadDigest": "sha256:workload",
        "policyDigest": "sha256:policy",
        "tlsPublicKeyHash": "c" * 64,
        "sessionId": "tees_test",
        "expiresAt": "2026-09-03T20:00:00Z",
        "runtimeSecurityRequirement": "controlled-sim-ok",
    }
    response = client.post("/v1/evidence/sign", json={"payload": evidence})
    assert response.status_code == 200
    assert response.json()["evidenceSignature"]

    receipt_response = client.post(
        "/v1/evidence/sign",
        json={"payload": {**simulation_claims(), "executionId": "exec_test"}},
    )
    assert receipt_response.status_code == 400


def test_receipt_signing_accepts_only_execution_receipts() -> None:
    receipt = {
        **simulation_claims(),
        "executionId": "exec_test",
        "taskId": "task_test",
        "taskSpecDigest": "a" * 64,
        "sessionId": "tees_test",
        "outputCiphertextSha256": "b" * 64,
        "completedAt": "2026-09-03T20:00:00Z",
        "modelDeploymentStatus": "ONLINE",
    }
    response = client.post("/v1/receipts/sign", json={"payload": receipt})
    assert response.status_code == 200
    assert response.json()["evidenceSignature"]

    arbitrary_response = client.post(
        "/v1/receipts/sign",
        json={"payload": {**receipt, "arbitrary": "must not be signed"}},
    )
    assert arbitrary_response.status_code == 400
