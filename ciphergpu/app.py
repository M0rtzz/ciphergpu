from __future__ import annotations

import os
from contextlib import asynccontextmanager

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
    StreamDeploymentPrepareRequest,
    TrainingJobPrepareRequest,
)
from .errors import CipherGpuError
from .service import ConfidentialExecutionService, HttpAttestationIssuer


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
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        execution_service.close()

    app = FastAPI(
        title="CipherGPU",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
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

    @app.get("/v1/model-deployments/{deployment_id}/logs")
    def get_model_deployment_logs(deployment_id: str) -> dict[str, object]:
        return execution_service.model_deployment_logs(deployment_id)

    @app.post("/v1/model-deployments/{deployment_id}/offline")
    def offline_model_deployment(deployment_id: str) -> dict[str, object]:
        return execution_service.offline_model_deployment(deployment_id)

    @app.post("/v1/model-deployments/{deployment_id}/stream/prepare")
    def prepare_stream(deployment_id: str, request: StreamDeploymentPrepareRequest) -> dict[str, object]:
        return execution_service.prepare_stream_deployment(deployment_id, request)

    @app.put("/v1/model-deployments/{deployment_id}/stream/chunks/{index}")
    async def append_stream_chunk(deployment_id: str, index: int, request: Request) -> dict[str, object]:
        return execution_service.append_stream_chunk(deployment_id, index, await request.body())

    @app.post("/v1/model-deployments/{deployment_id}/stream/finalize")
    def finalize_stream(deployment_id: str) -> dict[str, object]:
        return execution_service.finalize_stream_deployment(deployment_id)

    @app.post("/v1/training-jobs")
    def prepare_training_job(request: TrainingJobPrepareRequest) -> dict[str, object]:
        return execution_service.prepare_training_job(request)

    @app.put("/v1/training-jobs/{job_id}/inputs/{slot}/chunks/{index}")
    async def append_training_input(
        job_id: str, slot: str, index: int, request: Request
    ) -> dict[str, object]:
        return execution_service.append_training_input(job_id, slot, index, await request.body())

    @app.post("/v1/training-jobs/{job_id}/inputs/{slot}/finalize")
    def finalize_training_input(job_id: str, slot: str) -> dict[str, object]:
        return execution_service.finalize_training_input(job_id, slot)

    @app.post("/v1/training-jobs/{job_id}/start")
    def start_training_job(job_id: str) -> dict[str, object]:
        return execution_service.start_training_job(job_id)

    @app.get("/v1/training-jobs/{job_id}")
    def get_training_job(job_id: str) -> dict[str, object]:
        return execution_service.training_job(job_id)

    @app.get("/v1/training-jobs/{job_id}/logs")
    def get_training_logs(job_id: str) -> dict[str, object]:
        return execution_service.training_logs(job_id)

    @app.post("/v1/training-jobs/{job_id}/cancel")
    def cancel_training_job(job_id: str) -> dict[str, object]:
        return execution_service.cancel_training_job(job_id)

    @app.get("/v1/training-jobs/{job_id}/outputs")
    def get_training_outputs(job_id: str) -> dict[str, object]:
        return execution_service.training_outputs(job_id)

    @app.get("/v1/training-jobs/{job_id}/outputs/{slot}/chunks/{index}")
    def get_training_output_chunk(job_id: str, slot: str, index: int):
        from fastapi.responses import Response

        return Response(
            execution_service.training_output_chunk(job_id, slot, index),
            media_type="application/octet-stream",
        )

    @app.post("/v1/training-jobs/{job_id}/outputs/ack")
    def acknowledge_training_outputs(job_id: str) -> dict[str, object]:
        return execution_service.acknowledge_training_outputs(job_id)

    @app.post("/v1/confidential-inference/chat/completions")
    def confidential_inference(request: ConfidentialInferenceRequest) -> dict[str, object]:
        return execution_service.infer(request)

    @app.post("/v1/model-deployments/{deployment_id}/chat/completions")
    def runtime_chat(deployment_id: str, request: dict[str, object]) -> dict[str, object]:
        return execution_service.runtime_chat(deployment_id, request)

    @app.get("/v1/executions/{execution_id}/receipt")
    def get_receipt(execution_id: str) -> dict[str, object]:
        return execution_service.receipt(execution_id)

    @app.post("/v1/executions/{execution_id}/cancel")
    def cancel_execution(execution_id: str) -> dict[str, object]:
        return execution_service.cancel(execution_id)

    return app


app = create_app()
