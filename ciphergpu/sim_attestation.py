from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from .crypto import CONTRACT_VERSION, EvidenceSigner, sha256


class SignRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    payload: dict[str, Any]


signer = EvidenceSigner.load(os.environ.get("SIM_ATTESTATION_SIGNING_KEY"))
app = FastAPI(title="A100 Simulation Attestation Service", docs_url=None, redoc_url=None)


def validate(evidence: dict[str, Any]) -> None:
    required = {
        "contractVersion": CONTRACT_VERSION,
        "runtimeMode": "SIMULATION",
        "attestationVerified": False,
        "securityProfile": "a100-sim",
        "evidenceType": "SIMULATED_LAB_V1",
        "simulated": True,
        "hardwareModel": "NVIDIA A100",
    }
    if any(evidence.get(key) != value for key, value in required.items()):
        raise HTTPException(status_code=400, detail="evidence is not an explicit A100 simulation claim")
    for field in (
        "clientNonce",
        "taskSpecDigest",
        "teeEphemeralPublicKeyHash",
        "workloadDigest",
        "policyDigest",
        "tlsPublicKeyHash",
        "sessionId",
        "expiresAt",
    ):
        if not evidence.get(field):
            raise HTTPException(status_code=400, detail=f"missing evidence field: {field}")
    if evidence.get("runtimeSecurityRequirement") == "gpu-cc":
        raise HTTPException(status_code=400, detail="gpu-cc requirement cannot be signed by the simulation root")


def validate_receipt(receipt: dict[str, Any]) -> None:
    required = {
        "contractVersion": CONTRACT_VERSION,
        "runtimeMode": "SIMULATION",
        "attestationVerified": False,
        "securityProfile": "a100-sim",
        "evidenceType": "SIMULATED_LAB_V1",
        "simulated": True,
        "hardwareModel": "NVIDIA A100",
    }
    if any(receipt.get(key) != value for key, value in required.items()):
        raise HTTPException(status_code=400, detail="receipt is not an explicit A100 simulation claim")
    for field in (
        "executionId",
        "taskId",
        "taskSpecDigest",
        "sessionId",
        "outputCiphertextSha256",
        "completedAt",
    ):
        if not receipt.get(field):
            raise HTTPException(status_code=400, detail=f"missing receipt field: {field}")
    allowed = set(required) | {
        "executionId",
        "taskId",
        "taskSpecDigest",
        "sessionId",
        "outputCiphertextSha256",
        "completedAt",
        "modelDeploymentStatus",
    }
    if set(receipt) - allowed:
        raise HTTPException(status_code=400, detail="unsupported receipt field")
    if receipt.get("modelDeploymentStatus") not in {None, "ONLINE", "RUNTIME_REQUIRED"}:
        raise HTTPException(status_code=400, detail="invalid model deployment status")


@app.get("/v1/health")
def health() -> dict[str, Any]:
    return {
        "status": "UP",
        "contractVersion": CONTRACT_VERSION,
        "securityProfile": "a100-sim",
        "evidenceType": "SIMULATED_LAB_V1",
        "simulated": True,
        "hardwareModel": "NVIDIA A100",
        "evidenceSigningPublicKey": signer.public_key,
    }


@app.post("/v1/evidence/sign")
def sign(request: SignRequest) -> dict[str, str]:
    validate(request.payload)
    return {
        "evidenceDigest": sha256(__import__("rfc8785").dumps(request.payload)),
        "evidenceSignature": signer.sign(request.payload),
        "evidenceSigningPublicKey": signer.public_key,
    }


@app.post("/v1/receipts/sign")
def sign_receipt(request: SignRequest) -> dict[str, str]:
    validate_receipt(request.payload)
    return {
        "evidenceDigest": sha256(__import__("rfc8785").dumps(request.payload)),
        "evidenceSignature": signer.sign(request.payload),
        "evidenceSigningPublicKey": signer.public_key,
    }
