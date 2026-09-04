from __future__ import annotations

import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import __version__
from .crypto import CONTRACT_VERSION, EvidenceSigner, content_capabilities
from .models import (
    AttestationRequest,
    AttestationResponse,
    ConfidentialInferenceRequest,
    ExecutionRequest,
    ExecutionResponse,
    ModelDeploymentRequest,
)
from .service import CipherGpuError, ConfidentialExecutionService, HttpAttestationIssuer


def create_app(service: ConfidentialExecutionService | None = None) -> FastAPI:
    if service is not None:
        execution_service = service
    elif os.environ.get("SIM_ATTESTATION_URL"):
        client_cert = os.environ.get("SIM_ATTESTATION_CLIENT_CERT")
        client_key = os.environ.get("SIM_ATTESTATION_CLIENT_KEY")
        execution_service = ConfidentialExecutionService(
            HttpAttestationIssuer(
                os.environ["SIM_ATTESTATION_URL"],
                os.environ.get("SIM_ATTESTATION_CA"),
                (client_cert, client_key) if client_cert and client_key else None,
            )
        )
    else:
        execution_service = ConfidentialExecutionService(
            EvidenceSigner.load(os.environ.get("CIPHERGPU_EVIDENCE_SIGNING_KEY"))
        )
    app = FastAPI(title="CipherGPU", version=__version__, docs_url=None, redoc_url=None)
    app.state.execution_service = execution_service

    @app.exception_handler(CipherGpuError)
    async def handle_ciphergpu_error(_: Request, error: CipherGpuError) -> JSONResponse:
        return JSONResponse(
            status_code=error.status,
            content={"error": {"code": error.code, "message": error.message, "retryable": False}},
        )

    @app.get("/v1/health")
    def health() -> dict[str, object]:
        return {
            "status": "UP",
            "contractVersion": CONTRACT_VERSION,
            "runtimeMode": "SIMULATION",
            "attestationVerified": False,
            "securityProfile": "a100-sim",
            "evidenceType": "SIMULATED_LAB_V1",
            "simulated": True,
            "hardwareModel": "NVIDIA A100",
            "evidenceSigningPublicKey": execution_service.signer.public_key,
        }

    @app.get("/v1/crypto/capabilities")
    def crypto_capabilities() -> dict[str, object]:
        return {
            "format": "ds-envelope/v2",
            "defaultAlgorithm": "AES-256-GCM",
            "contentEncryptionAlgorithms": content_capabilities(),
        }

    @app.post("/v1/attestation/sessions", response_model=AttestationResponse, response_model_by_alias=True)
    def create_attestation(request: AttestationRequest) -> AttestationResponse:
        return execution_service.create_session(request)

    @app.post("/v1/executions", response_model=ExecutionResponse, response_model_by_alias=True)
    def create_execution(request: ExecutionRequest) -> ExecutionResponse:
        return execution_service.execute(request)

    @app.post("/v1/model-deployments")
    def register_model_deployment(request: ModelDeploymentRequest) -> dict[str, object]:
        return execution_service.register_model_deployment(request)

    @app.get("/v1/model-deployments/{deployment_id}")
    def get_model_deployment(deployment_id: str) -> dict[str, object]:
        return execution_service.model_deployment(deployment_id)

    @app.post("/v1/model-deployments/{deployment_id}/offline")
    def offline_model_deployment(deployment_id: str) -> dict[str, object]:
        return execution_service.offline_model_deployment(deployment_id)

    @app.post("/v1/confidential-inference/chat/completions")
    def confidential_inference(request: ConfidentialInferenceRequest) -> dict[str, object]:
        return execution_service.infer(request)

    @app.get("/v1/executions/{execution_id}/receipt")
    def get_receipt(execution_id: str) -> dict[str, object]:
        return execution_service.receipt(execution_id)

    @app.post("/v1/executions/{execution_id}/cancel")
    def cancel_execution(execution_id: str) -> dict[str, object]:
        return execution_service.cancel(execution_id)

    return app


app = create_app()
