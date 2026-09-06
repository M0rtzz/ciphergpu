from __future__ import annotations

import ipaddress
import json
import os
import socket
import ssl
import tarfile
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

import httpx
from cryptography.exceptions import InvalidSignature
from pyhpke import PyHPKEError

from .crypto import (
    CONTRACT_VERSION,
    HPKE_DEK_INFO,
    HPKE_ODK_INFO,
    HPKE_REQUEST_KEY_INFO,
    HPKE_RESULT_DEK_INFO,
    aes_open,
    aes_seal,
    b64u,
    canonical,
    content_open,
    content_seal,
    hpke_open,
    hpke_seal,
    json_bytes,
    new_hpke_key_pair,
    sha256,
    unb64u,
    verify_ed25519,
)
from .errors import CipherGpuError
from .model_runtime import ModelRuntimeManager
from .training_runtime import TrainingRuntimeManager
from .models import (
    AttestationRequest,
    AttestationResponse,
    ConfidentialInferenceRequest,
    ExecutionRequest,
    ExecutionResponse,
    ModelDeploymentRequest,
    StreamDeploymentPrepareRequest,
    TrainingInputSpec,
    TrainingJobPrepareRequest,
)


class AttestationIssuer(Protocol):
    @property
    def public_key(self) -> str: ...

    def sign_evidence(self, payload: Any) -> str: ...

    def sign_receipt(self, payload: Any) -> str: ...


class HttpAttestationIssuer:
    """Client for the separately deployed, simulation-only attestation signer."""

    def __init__(self, base_url: str, ca: str | None = None, cert: tuple[str, str] | None = None):
        tls: ssl.SSLContext | bool = True
        if ca:
            tls = ssl.create_default_context(cafile=ca)
        elif cert:
            tls = ssl.create_default_context()
        if cert:
            assert isinstance(tls, ssl.SSLContext)
            tls.load_cert_chain(certfile=cert[0], keyfile=cert[1])
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"), verify=tls, timeout=5, trust_env=False
        )
        health = self._client.get("/v1/health")
        health.raise_for_status()
        body = health.json()
        if body.get("securityProfile") != "a100-sim" or body.get("simulated") is not True:
            raise RuntimeError("attestation signer is not an a100-sim trust service")
        self._public_key = str(body["evidenceSigningPublicKey"])

    @property
    def public_key(self) -> str:
        return self._public_key

    def _sign(self, endpoint: str, payload: Any) -> str:
        response = self._client.post(endpoint, json={"payload": payload})
        response.raise_for_status()
        body = response.json()
        if body.get("evidenceSigningPublicKey") != self._public_key:
            raise RuntimeError("simulation trust root changed during a session")
        return str(body["evidenceSignature"])

    def sign_evidence(self, payload: Any) -> str:
        return self._sign("/v1/evidence/sign", payload)

    def sign_receipt(self, payload: Any) -> str:
        return self._sign("/v1/receipts/sign", payload)


def now() -> datetime:
    return datetime.now(UTC)


def iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def parse_time(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value).astimezone(UTC)
    except ValueError as failure:
        raise CipherGpuError("CONTRACT_INVALID", "invalid timestamp") from failure


@dataclass
class Session:
    session_id: str
    domain_id: str
    task_spec_digest: str
    public_key: bytes
    private_key: Any
    expires_at: datetime
    consumed: bool = False


@dataclass
class ModelDeployment:
    deployment_id: str
    source_type: str
    base_url: str | None
    upstream_model_id: str
    timeout_seconds: int
    status: str = "AUTHORIZATION_REQUIRED"
    session_id: str | None = None
    secret: bytearray | None = None
    error_code: str | None = None


@dataclass
class StreamDeployment:
    asset_version_id: str
    dek: bytearray
    chunks: list[Any]
    next_index: int = 0


@dataclass
class TrainingInputState:
    spec: TrainingInputSpec
    dek: bytearray
    next_index: int = 0
    finalized: bool = False


@dataclass
class TrainingOutputState:
    manifest: dict[str, Any]
    chunks: list[Path]


@dataclass
class TrainingJob:
    job_id: str
    task_spec_digest: str
    session_id: str
    domain_id: str
    adapter_id: str
    training_config: dict[str, Any]
    output_recipient: Any
    inputs: dict[str, TrainingInputState]
    status: str = "STAGING_INPUTS"
    progress: int = 0
    current_epoch: float = 0
    metrics: dict[str, Any] | None = None
    outputs: dict[str, TrainingOutputState] | None = None
    error_code: str | None = None


class ConfidentialExecutionService:
    def __init__(
        self,
        signer: AttestationIssuer,
        workload_digest: str | None = None,
        policy_digest: str | None = None,
    ):
        self.signer = signer
        self.workload_digest = workload_digest or os.environ.get(
            "CIPHERGPU_WORKLOAD_DIGEST", sha256(b"ciphergpu:builtin-digest/v1")
        )
        self.policy_digest = policy_digest or os.environ.get(
            "CIPHERGPU_POLICY_DIGEST", "sha256:a100-sim-policy-v1"
        )
        self.tls_public_key_hash = os.environ.get("CIPHERGPU_TLS_PUBLIC_KEY_HASH", "simulation-unbound")
        self._sessions: dict[str, Session] = {}
        self._consumed_jti: set[str] = set()
        self._receipts: dict[str, dict[str, Any]] = {}
        self._model_deployments: dict[str, ModelDeployment] = {}
        self._stream_deployments: dict[str, StreamDeployment] = {}
        self._runtime = ModelRuntimeManager()
        self._training_runtime = TrainingRuntimeManager()
        self._training_jobs: dict[str, TrainingJob] = {}
        self._lock = threading.RLock()

    def close(self) -> None:
        """Stop workers and clear all ephemeral key material before process exit."""
        with self._lock:
            training_jobs = list(self._training_jobs.values())
            deployments = list(self._model_deployments.values())
            streams = list(self._stream_deployments.values())
            sessions = list(self._sessions.values())
        for job in training_jobs:
            self._clear_training_deks(job)
            self._training_runtime.cleanup_all(job.job_id)
            if job.status not in {"COMPLETED", "FAILED", "CANCELLED"}:
                job.status = "FAILED"
                job.error_code = "RUNTIME_SHUTDOWN"
        for deployment in deployments:
            self._clear_deployment_secret(deployment)
            self._runtime.stop(deployment.deployment_id)
        for stream in streams:
            stream.dek[:] = b"\x00" * len(stream.dek)
        for session in sessions:
            session.private_key = None
        with self._lock:
            self._stream_deployments.clear()
            self._sessions.clear()

    def register_model_deployment(self, request: ModelDeploymentRequest) -> dict[str, object]:
        base_url = request.base_url
        if request.source_type == "OPENAI_COMPATIBLE":
            if not base_url:
                raise CipherGpuError("CONTRACT_INVALID", "remote model baseUrl is required")
            self._validate_remote_url(base_url)
        else:
            configured = os.environ.get("CIPHERGPU_VLLM_URL")
            base_url = configured.rstrip("/") if configured else None
        deployment = ModelDeployment(
            request.deployment_id,
            request.source_type,
            base_url,
            request.upstream_model_id,
            request.timeout_seconds,
        )
        with self._lock:
            previous = self._model_deployments.get(request.deployment_id)
            if previous and previous.status == "ONLINE":
                return self._deployment_view(previous)
            self._model_deployments[request.deployment_id] = deployment
        return self._deployment_view(deployment)

    def model_deployment(self, deployment_id: str) -> dict[str, object]:
        with self._lock:
            deployment = self._model_deployments.get(deployment_id)
        if deployment is None:
            raise CipherGpuError("MODEL_DEPLOYMENT_NOT_FOUND", "model deployment was not registered", 404)
        return self._deployment_view(deployment)

    def model_deployment_logs(self, deployment_id: str) -> dict[str, object]:
        with self._lock:
            deployment = self._model_deployments.get(deployment_id)
        if deployment is None:
            raise CipherGpuError("MODEL_DEPLOYMENT_NOT_FOUND", "model deployment was not registered", 404)
        return {"deploymentId": deployment_id, "status": deployment.status,
                "logs": self._runtime.logs(deployment_id)}

    def offline_model_deployment(self, deployment_id: str) -> dict[str, object]:
        with self._lock:
            deployment = self._model_deployments.get(deployment_id)
            if deployment is None:
                # Deployments and their unsealed secrets are intentionally ephemeral. After an
                # agent restart, an absent deployment is already in the requested safe state.
                return {
                    "deploymentId": deployment_id,
                    "securityProfile": "a100-sim",
                    "simulated": True,
                    "status": "OFFLINE",
                    "sessionId": None,
                    "errorCode": None,
                    "alreadyAbsent": True,
                    "sessionKeysDestroyed": True,
                }
            self._clear_deployment_secret(deployment)
            self._runtime.stop(deployment_id)
            stream = self._stream_deployments.pop(deployment_id, None)
            if stream:
                stream.dek[:] = b"\x00" * len(stream.dek)
            deployment.status = "OFFLINE"
            deployment.session_id = None
        return self._deployment_view(deployment)

    def prepare_stream_deployment(self, deployment_id: str, request: StreamDeploymentPrepareRequest) -> dict[str, object]:
        self._purge()
        with self._lock:
            deployment = self._model_deployments.get(deployment_id)
            session = self._sessions.get(request.session_id)
        if deployment is None or deployment.source_type != "LOCAL_WEIGHTS":
            raise CipherGpuError("MODEL_DEPLOYMENT_NOT_FOUND", "local deployment was not registered", 404)
        task = request.task_spec.model_dump(by_alias=True)
        digest = sha256(canonical(task))
        if digest != request.task_spec_digest or session is None or session.expires_at <= now():
            raise CipherGpuError("TASK_DIGEST_MISMATCH", "stream task or session is invalid")
        if (request.task_spec.workload_id != f"model.deploy/{deployment_id}"
                or request.task_spec.asset_version_ids != [request.asset_version_id]):
            raise CipherGpuError("GRANT_SCOPE_INVALID", "stream task is not bound to this deployment and asset")
        if (session.domain_id != request.task_spec.domain_id
                or request.task_spec.security_profile != "a100-sim"
                or request.task_spec.evidence_type != "SIMULATED_LAB_V1"
                or request.task_spec.simulated is not True
                or request.task_spec.hardware_model != "NVIDIA A100"
                or request.task_spec.runtime_security_requirement == "gpu-cc"
                or request.task_spec.workload_digest != self.workload_digest
                or request.task_spec.policy_digest != self.policy_digest
                or parse_time(request.task_spec.expires_at) <= now()):
            raise CipherGpuError("SECURITY_DOWNGRADE_DENIED", "stream task security policy is invalid")
        if session.task_spec_digest != digest or session.consumed:
            raise CipherGpuError("GRANT_REPLAYED", "stream authorization was already consumed")
        claims = request.grant.claims.model_dump(by_alias=True)
        try:
            verify_ed25519(request.grant.signing_public_key, request.grant.signature, claims)
        except (InvalidSignature, ValueError) as failure:
            raise CipherGpuError("GRANT_SIGNATURE_INVALID", "grant signature is invalid") from failure
        if claims["taskSpecDigest"] != digest or claims["teeSessionId"] != session.session_id:
            raise CipherGpuError("GRANT_SCOPE_INVALID", "grant is not bound to the stream session")
        expected_profile = {
            "securityProfile": request.task_spec.security_profile,
            "evidenceType": request.task_spec.evidence_type,
            "simulated": request.task_spec.simulated,
            "hardwareModel": request.task_spec.hardware_model,
            "runtimeSecurityRequirement": request.task_spec.runtime_security_requirement,
        }
        if (claims["teeEphemeralPublicKeyHash"] != sha256(session.public_key)
                or any(claims[key] != value for key, value in expected_profile.items())
                or claims["outputRecipients"] != request.task_spec.output_recipients
                or claims["maxUses"] != 1):
            raise CipherGpuError("GRANT_SCOPE_INVALID", "stream grant security scope is invalid")
        if parse_time(claims["nbf"]) > now() + timedelta(seconds=30) or parse_time(claims["exp"]) <= now():
            raise CipherGpuError("GRANT_EXPIRED", "stream grant is outside its validity window")
        if claims["jti"] in self._consumed_jti:
            raise CipherGpuError("GRANT_REPLAYED", "stream grant was already consumed")
        if claims["assetVersionIds"] != [request.asset_version_id] or request.sealed_dek.asset_version_id != request.asset_version_id:
            raise CipherGpuError("GRANT_SCOPE_INVALID", "grant is not bound to this model version")
        expected_aad = f"{digest}|{request.asset_version_id}|{claims['grantId']}|{claims['exp']}".encode()
        if unb64u(request.sealed_dek.aad) != expected_aad:
            raise CipherGpuError("KEY_MATCH_FAILED", "sealed DEK AAD is invalid")
        try:
            dek = bytearray(hpke_open(session.private_key, request.sealed_dek.enc,
                                     request.sealed_dek.ciphertext, expected_aad, HPKE_DEK_INFO))
        except (PyHPKEError, ValueError) as failure:
            raise CipherGpuError("KEY_MATCH_FAILED", "DEK cannot be opened") from failure
        if len(dek) != 32 or [item.index for item in request.chunks] != list(range(len(request.chunks))):
            dek[:] = b"\x00" * len(dek)
            raise CipherGpuError("CONTRACT_INVALID", "DEK or chunk sequence is invalid")
        self._runtime.prepare_archive(deployment_id)
        with self._lock:
            session.consumed = True
            self._consumed_jti.add(claims["jti"])
            self._stream_deployments[deployment_id] = StreamDeployment(request.asset_version_id, dek, request.chunks)
            deployment.status = "DECRYPTING"
            deployment.session_id = session.session_id
        return self._deployment_view(deployment)

    def append_stream_chunk(self, deployment_id: str, index: int, ciphertext: bytes) -> dict[str, object]:
        with self._lock:
            state = self._stream_deployments.get(deployment_id)
        if state is None or index != state.next_index:
            raise CipherGpuError("CONTRACT_INVALID", "unexpected model chunk index")
        metadata = state.chunks[index]
        if sha256(ciphertext) != metadata.ciphertext_sha256:
            raise CipherGpuError("DATA_INTEGRITY_FAILED", "model chunk digest mismatch")
        try:
            plaintext = content_open(bytes(state.dek), metadata.envelope_id, metadata.algorithm,
                                     metadata.nonce, b64u(ciphertext), metadata.aad,
                                     metadata.implementation_version)
            self._runtime.append_ciphertext_plaintext(deployment_id, plaintext)
            state.next_index += 1
            return {"deploymentId": deployment_id, "index": index, "status": "DECRYPTED"}
        except Exception as failure:
            raise CipherGpuError("DATA_INTEGRITY_FAILED", "model chunk authentication failed") from failure

    def finalize_stream_deployment(self, deployment_id: str) -> dict[str, object]:
        with self._lock:
            state = self._stream_deployments.get(deployment_id)
            deployment = self._model_deployments.get(deployment_id)
        if state is None or deployment is None or state.next_index != len(state.chunks):
            raise CipherGpuError("CONTRACT_INVALID", "model stream is incomplete")
        deployment.status = "LOADING"
        try:
            runtime = self._runtime.start(deployment_id, deployment.upstream_model_id, deployment.timeout_seconds)
            deployment.base_url = f"http://127.0.0.1:{runtime.port}/v1"
            deployment.status = "ONLINE"
            deployment.error_code = None
            return self._deployment_view(deployment)
        except CipherGpuError as failure:
            deployment.status = "FAILED"
            deployment.error_code = failure.code
            self._runtime.stop(deployment_id)
            raise
        finally:
            state.dek[:] = b"\x00" * len(state.dek)
            self._stream_deployments.pop(deployment_id, None)

    def prepare_training_job(self, request: TrainingJobPrepareRequest) -> dict[str, object]:
        self._purge()
        self._validate_training_config(request.adapter_id, request.training_config)
        if sha256(canonical(request.training_config)) != request.training_config_hash:
            raise CipherGpuError("TRAINING_CONFIG_MISMATCH", "training config hash is invalid")
        task = request.task_spec.model_dump(by_alias=True)
        digest = sha256(canonical(task))
        with self._lock:
            session = self._sessions.get(request.session_id)
            existing = self._training_jobs.get(request.job_id)
        if existing and existing.status not in {"FAILED", "CANCELLED", "COMPLETED"}:
            return self._training_job_view(existing)
        expected_workload = (
            f"training/{request.job_id}/{request.adapter_id}/{request.training_config_hash}"
        )
        if (
            digest != request.task_spec_digest
            or session is None
            or session.expires_at <= now()
            or request.task_spec.purpose != "train"
            or request.task_spec.workload_id != expected_workload
        ):
            raise CipherGpuError("TASK_DIGEST_MISMATCH", "training task or session is invalid")
        input_assets = [item.asset_version_id for item in request.inputs]
        input_slots = [item.slot for item in request.inputs]
        if (
            len(input_assets) != len(set(input_assets))
            or len(input_slots) != len(set(input_slots))
            or "model" not in input_slots
            or "train-data" not in input_slots
            or set(input_assets) != set(request.task_spec.asset_version_ids)
            or request.task_spec.output_recipients != [request.output_recipient.kid]
        ):
            raise CipherGpuError("GRANT_SCOPE_INVALID", "training inputs do not match task scope")
        if session.consumed:
            raise CipherGpuError("GRANT_REPLAYED", "training authorization was already consumed")
        if (
            session.domain_id != request.task_spec.domain_id
            or session.task_spec_digest != digest
            or request.task_spec.security_profile != "a100-sim"
            or request.task_spec.evidence_type != "SIMULATED_LAB_V1"
            or request.task_spec.simulated is not True
            or request.task_spec.hardware_model != "NVIDIA A100"
            or request.task_spec.runtime_security_requirement == "gpu-cc"
            or request.task_spec.workload_digest != self.workload_digest
            or request.task_spec.policy_digest != self.policy_digest
            or request.task_spec.egress_policy != "deny-all"
            or parse_time(request.task_spec.expires_at) <= now()
        ):
            raise CipherGpuError("SECURITY_DOWNGRADE_DENIED", "training security policy is invalid")

        grants_by_asset: dict[str, Any] = {}
        grant_claims: dict[str, dict[str, Any]] = {}
        pending_jtis: set[str] = set()
        for grant in request.grants:
            claims = grant.claims.model_dump(by_alias=True)
            try:
                verify_ed25519(grant.signing_public_key, grant.signature, claims)
            except (InvalidSignature, ValueError) as failure:
                raise CipherGpuError("GRANT_SIGNATURE_INVALID", "training grant signature is invalid") from failure
            if (
                claims["taskSpecDigest"] != digest
                or claims["teeSessionId"] != session.session_id
                or claims["teeEphemeralPublicKeyHash"] != sha256(session.public_key)
                or claims["securityProfile"] != request.task_spec.security_profile
                or claims["evidenceType"] != request.task_spec.evidence_type
                or claims["simulated"] is not True
                or claims["hardwareModel"] != request.task_spec.hardware_model
                or claims["runtimeSecurityRequirement"]
                != request.task_spec.runtime_security_requirement
                or claims["outputRecipients"] != request.task_spec.output_recipients
                or claims["maxUses"] != 1
                or parse_time(claims["nbf"]) > now() + timedelta(seconds=30)
                or parse_time(claims["exp"]) <= now()
            ):
                raise CipherGpuError("GRANT_SCOPE_INVALID", "training grant scope is invalid")
            jti = claims["jti"]
            if jti in self._consumed_jti or jti in pending_jtis:
                raise CipherGpuError("GRANT_REPLAYED", "training grant was already consumed")
            pending_jtis.add(jti)
            grant_claims[claims["grantId"]] = claims
            for asset in claims["assetVersionIds"]:
                if asset in grants_by_asset:
                    raise CipherGpuError("GRANT_SCOPE_INVALID", "asset has duplicate grants")
                grants_by_asset[asset] = grant
        if set(grants_by_asset) != set(input_assets):
            raise CipherGpuError("GRANT_SCOPE_INVALID", "every training input requires one grant")

        opened: dict[str, TrainingInputState] = {}
        try:
            for item in request.inputs:
                try:
                    verify_ed25519(
                        item.owner_signing_public_key, item.owner_signature, item.manifest
                    )
                except (InvalidSignature, ValueError) as failure:
                    raise CipherGpuError(
                        "TRAINING_MANIFEST_SIGNATURE_INVALID",
                        "training input manifest signature is invalid",
                    ) from failure
                declared = item.manifest.get("chunks")
                if (
                    sha256(canonical(item.manifest)) != item.manifest_hash
                    or item.manifest.get("format") != "ds-envelope/v2"
                    or not isinstance(declared, list)
                    or len(declared) != len(item.chunks)
                    or any(
                        declared[index].get("index") != chunk.index
                        or declared[index].get("nonce") != chunk.nonce
                        or declared[index].get("sha256") != chunk.ciphertext_sha256
                        or declared[index].get("aad") != chunk.aad
                        for index, chunk in enumerate(item.chunks)
                    )
                ):
                    raise CipherGpuError(
                        "TRAINING_MANIFEST_INVALID", "training input manifest is invalid"
                    )
                if item.sealed_dek.asset_version_id != item.asset_version_id:
                    raise CipherGpuError("GRANT_SCOPE_INVALID", "sealed DEK asset is invalid")
                grant = grants_by_asset[item.asset_version_id]
                claims = grant_claims[grant.claims.grant_id]
                expected_aad = (
                    f"{digest}|{item.asset_version_id}|{claims['grantId']}|{claims['exp']}"
                ).encode()
                if unb64u(item.sealed_dek.aad) != expected_aad:
                    raise CipherGpuError("KEY_MATCH_FAILED", "training sealed DEK AAD is invalid")
                try:
                    dek = bytearray(
                        hpke_open(
                            session.private_key,
                            item.sealed_dek.enc,
                            item.sealed_dek.ciphertext,
                            expected_aad,
                            HPKE_DEK_INFO,
                        )
                    )
                except (PyHPKEError, ValueError) as failure:
                    raise CipherGpuError("KEY_MATCH_FAILED", "training DEK cannot be opened") from failure
                if len(dek) != 32 or [chunk.index for chunk in item.chunks] != list(
                    range(len(item.chunks))
                ):
                    dek[:] = b"\x00" * len(dek)
                    raise CipherGpuError("CONTRACT_INVALID", "training chunk sequence is invalid")
                opened[item.slot] = TrainingInputState(item, dek)
            self._training_runtime.prepare_job(request.job_id)
            job = TrainingJob(
                request.job_id,
                digest,
                request.session_id,
                request.task_spec.domain_id,
                request.adapter_id,
                request.training_config,
                request.output_recipient,
                opened,
            )
            with self._lock:
                session.consumed = True
                session.private_key = None
                self._consumed_jti.update(pending_jtis)
                self._training_jobs[request.job_id] = job
            return self._training_job_view(job)
        except Exception:
            for state in opened.values():
                state.dek[:] = b"\x00" * len(state.dek)
            raise

    def append_training_input(
        self, job_id: str, slot: str, index: int, ciphertext: bytes
    ) -> dict[str, object]:
        job = self._require_training_job(job_id)
        state = job.inputs.get(slot)
        if job.status != "STAGING_INPUTS" or state is None or state.finalized:
            raise CipherGpuError("TRAINING_STATE_INVALID", "training input is not writable")
        if index != state.next_index or index >= len(state.spec.chunks):
            raise CipherGpuError("CONTRACT_INVALID", "unexpected training chunk index")
        metadata = state.spec.chunks[index]
        if sha256(ciphertext) != metadata.ciphertext_sha256:
            self._fail_training_job(job, "DATA_INTEGRITY_FAILED")
            raise CipherGpuError("DATA_INTEGRITY_FAILED", "training chunk digest mismatch")
        try:
            plaintext = bytearray(
                content_open(
                    bytes(state.dek),
                    metadata.envelope_id,
                    metadata.algorithm,
                    metadata.nonce,
                    b64u(ciphertext),
                    metadata.aad,
                    metadata.implementation_version,
                )
            )
            try:
                self._training_runtime.append_input(job_id, slot, bytes(plaintext))
            finally:
                plaintext[:] = b"\x00" * len(plaintext)
            state.next_index += 1
            total = sum(len(value.spec.chunks) for value in job.inputs.values())
            received = sum(value.next_index for value in job.inputs.values())
            job.progress = min(25, round(received / total * 25))
            return {"jobId": job_id, "slot": slot, "index": index, "status": "DECRYPTED"}
        except CipherGpuError:
            raise
        except Exception as failure:
            self._fail_training_job(job, "DATA_INTEGRITY_FAILED")
            raise CipherGpuError("DATA_INTEGRITY_FAILED", "training chunk authentication failed") from failure

    def finalize_training_input(self, job_id: str, slot: str) -> dict[str, object]:
        job = self._require_training_job(job_id)
        state = job.inputs.get(slot)
        if state is None or state.finalized or state.next_index != len(state.spec.chunks):
            raise CipherGpuError("TRAINING_STATE_INVALID", "training input is incomplete")
        try:
            self._training_runtime.finalize_input(
                job_id, slot, state.spec.package_format, job.adapter_id
            )
            state.finalized = True
            state.dek[:] = b"\x00" * len(state.dek)
            if all(value.finalized for value in job.inputs.values()):
                job.status = "VALIDATING_INPUTS"
                job.progress = 30
            return self._training_job_view(job)
        except Exception as failure:
            self._fail_training_job(job, getattr(failure, "code", "TRAINING_PACKAGE_INVALID"))
            raise

    def start_training_job(self, job_id: str) -> dict[str, object]:
        job = self._require_training_job(job_id)
        if job.status != "VALIDATING_INPUTS" or not all(
            item.finalized for item in job.inputs.values()
        ):
            raise CipherGpuError("TRAINING_STATE_INVALID", "training inputs are not ready")
        with self._lock:
            if any(
                deployment.status == "ONLINE"
                for deployment in self._model_deployments.values()
            ):
                raise CipherGpuError(
                    "GPU_RESOURCE_BUSY", "an online model deployment is using the training GPU", 409
                )
            if any(
                value.job_id != job_id
                and value.status in {"RUNNING", "ENCRYPTING_OUTPUTS"}
                for value in self._training_jobs.values()
            ):
                raise CipherGpuError("GPU_RESOURCE_BUSY", "another training job is running", 409)
        try:
            self._training_runtime.start(job_id, job.adapter_id, job.training_config)
            job.status = "RUNNING"
            job.progress = 31
            threading.Thread(
                target=self._monitor_training_job, args=(job_id,), daemon=True
            ).start()
            return self._training_job_view(job)
        except Exception as failure:
            self._fail_training_job(job, getattr(failure, "code", "TRAINING_START_FAILED"))
            raise

    def training_job(self, job_id: str) -> dict[str, object]:
        return self._training_job_view(self._require_training_job(job_id))

    def training_logs(self, job_id: str) -> dict[str, object]:
        job = self._require_training_job(job_id)
        return {
            "jobId": job_id,
            "status": job.status,
            "logs": self._training_runtime.logs(job_id),
        }

    def cancel_training_job(self, job_id: str) -> dict[str, object]:
        job = self._require_training_job(job_id)
        if job.status in {"COMPLETED", "FAILED", "CANCELLED"}:
            return self._training_job_view(job)
        job.status = "CANCELLING"
        self._clear_training_deks(job)
        self._training_runtime.cleanup_all(job_id)
        job.status = "CANCELLED"
        job.progress = 0
        return self._training_job_view(job)

    def training_outputs(self, job_id: str) -> dict[str, object]:
        job = self._require_training_job(job_id)
        if job.status not in {"OUTPUT_READY", "COMPLETED"} or not job.outputs:
            raise CipherGpuError("TRAINING_OUTPUT_NOT_READY", "training outputs are not ready", 409)
        return {
            "jobId": job_id,
            "status": job.status,
            "outputs": {
                slot: {"manifest": output.manifest, "chunkCount": len(output.chunks)}
                for slot, output in job.outputs.items()
            },
        }

    def training_output_chunk(self, job_id: str, slot: str, index: int) -> bytes:
        job = self._require_training_job(job_id)
        if job.status != "OUTPUT_READY" or not job.outputs or slot not in job.outputs:
            raise CipherGpuError("TRAINING_OUTPUT_NOT_READY", "training output is unavailable", 409)
        chunks = job.outputs[slot].chunks
        if index < 0 or index >= len(chunks):
            raise CipherGpuError("CONTRACT_INVALID", "invalid training output chunk")
        return chunks[index].read_bytes()

    def acknowledge_training_outputs(self, job_id: str) -> dict[str, object]:
        job = self._require_training_job(job_id)
        if job.status != "OUTPUT_READY" or not job.outputs:
            raise CipherGpuError("TRAINING_STATE_INVALID", "training outputs cannot be acknowledged")
        self._training_runtime.cleanup_all(job_id)
        job.status = "COMPLETED"
        job.progress = 100
        return self._training_job_view(job)

    def _monitor_training_job(self, job_id: str) -> None:
        job = self._require_training_job(job_id)
        try:
            timeout = int(job.training_config.get("maxRuntimeSeconds", 7200))
            code = self._training_runtime.wait(job_id, timeout=timeout)
            if code != 0:
                raise CipherGpuError("TRAINING_WORKER_FAILED", "training worker exited with an error")
            progress = self._training_runtime.progress(job_id)
            metrics = progress.get("metrics")
            if isinstance(metrics, dict):
                job.metrics = metrics
            job.current_epoch = float(progress.get("currentEpoch", job.current_epoch))
            job.status = "ENCRYPTING_OUTPUTS"
            job.progress = 97
            job.outputs = {
                "result-model": self._encrypt_training_output(job, "result-model", "model"),
                "result-data": self._encrypt_training_output(job, "result-data", "data"),
            }
            self._training_runtime.cleanup_plaintext(job_id)
            job.status = "OUTPUT_READY"
            job.progress = 99
        except Exception as failure:
            self._fail_training_job(job, getattr(failure, "code", "TRAINING_WORKER_FAILED"))

    def _encrypt_training_output(
        self, job: TrainingJob, slot: str, source_name: str
    ) -> TrainingOutputState:
        source = self._training_runtime.output_directory(job.job_id) / source_name
        if not source.is_dir() or not any(source.iterdir()):
            raise CipherGpuError("TRAINING_OUTPUT_INVALID", f"{source_name} output is empty")
        archive = self._training_runtime.output_directory(job.job_id) / f"{slot}.tar.gz"
        with tarfile.open(archive, "w:gz") as bundle:
            bundle.add(source, arcname=source_name, recursive=True)
        output_id = f"output_{uuid.uuid4().hex}"
        envelope_id = f"env_{uuid.uuid4().hex}"
        algorithm = "AES-256-GCM"
        chunk_size = 32 * 1024 * 1024
        dek = bytearray(os.urandom(32))
        chunks: list[dict[str, Any]] = []
        paths: list[Path] = []
        encrypted_dir = self._training_runtime.encrypted_directory(job.job_id) / slot
        encrypted_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            with archive.open("rb") as source_file:
                index = 0
                while plaintext := source_file.read(chunk_size):
                    nonce = os.urandom(12)
                    aad = {
                        "format": "ds-envelope/v2",
                        "envelopeId": envelope_id,
                        "contentEncryptionAlgorithm": algorithm,
                        "implementationVersion": "1",
                        "taskId": job.job_id,
                        "outputId": output_id,
                        "domainId": job.domain_id,
                        "assetType": "RESULT_MODEL" if slot == "result-model" else "RESULT_DATA",
                        "producerNodeId": "ciphergpu",
                        "recipientKid": job.output_recipient.kid,
                        "chunkIndex": index,
                        "plaintextLength": len(plaintext),
                    }
                    ciphertext = content_seal(
                        bytes(dek), envelope_id, algorithm, nonce, plaintext, aad
                    )
                    path = encrypted_dir / f"{index:08d}"
                    path.write_bytes(ciphertext)
                    paths.append(path)
                    chunks.append(
                        {
                            "index": index,
                            "plaintextLength": len(plaintext),
                            "nonce": b64u(nonce),
                            "sha256": sha256(ciphertext),
                            "aad": aad,
                        }
                    )
                    index += 1
            binding = {
                "format": "ds-envelope/v2",
                "envelopeId": envelope_id,
                "contentEncryptionAlgorithm": algorithm,
                "implementationVersion": "1",
                "domainId": job.domain_id,
                "publicKeyId": job.output_recipient.kid,
                "publicKeyVersion": 1,
                "originalSize": archive.stat().st_size,
                "chunks": [
                    {
                        "index": item["index"],
                        "plaintextLength": item["plaintextLength"],
                        "sha256": item["sha256"],
                    }
                    for item in chunks
                ],
            }
            envelope_aad = canonical(binding)
            envelope = hpke_seal(
                job.output_recipient.encryption_public_key,
                bytes(dek),
                envelope_aad,
                HPKE_RESULT_DEK_INFO,
            )
            manifest: dict[str, Any] = {
                **binding,
                "taskId": job.job_id,
                "outputId": output_id,
                "assetType": "RESULT_MODEL" if slot == "result-model" else "RESULT_DATA",
                "recipientKid": job.output_recipient.kid,
                "algorithm": algorithm,
                "cipherHash": sha256(envelope_aad),
                "cipherSize": sum(path.stat().st_size for path in paths),
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
                "keyEnvelope": {
                    "recipientKid": job.output_recipient.kid,
                    **envelope,
                    "info": b64u(HPKE_RESULT_DEK_INFO),
                    "aad": b64u(envelope_aad),
                    "aadHash": sha256(envelope_aad),
                },
                "producerType": "CIPHERGPU",
                "producerNodeId": "ciphergpu",
                "producerEvidenceSigningPublicKey": self.signer.public_key,
                "trainingConfigHash": sha256(canonical(job.training_config)),
                "runtimeReceipt": (job.metrics or {}).get("runtime", {}),
            }
            manifest["producerSignature"] = self.signer.sign_receipt(manifest)
            return TrainingOutputState(manifest, paths)
        finally:
            dek[:] = b"\x00" * len(dek)

    def _training_job_view(self, job: TrainingJob) -> dict[str, object]:
        if job.status == "RUNNING":
            progress = self._training_runtime.progress(job.job_id)
            job.progress = max(job.progress, int(progress.get("progress", 0)))
            job.current_epoch = float(progress.get("currentEpoch", 0))
            metrics = progress.get("metrics")
            if isinstance(metrics, dict):
                job.metrics = metrics
        return {
            "jobId": job.job_id,
            "status": job.status,
            "progress": job.progress,
            "currentEpoch": job.current_epoch,
            "metrics": job.metrics or {},
            "adapterId": job.adapter_id,
            "taskSpecDigest": job.task_spec_digest,
            "sessionId": job.session_id,
            "inputStatus": {
                slot: {
                    "receivedChunks": state.next_index,
                    "expectedChunks": len(state.spec.chunks),
                    "finalized": state.finalized,
                }
                for slot, state in job.inputs.items()
            },
            "errorCode": job.error_code,
            "plaintextCleaned": job.status in {"OUTPUT_READY", "COMPLETED", "FAILED", "CANCELLED"},
        }

    def _require_training_job(self, job_id: str) -> TrainingJob:
        with self._lock:
            job = self._training_jobs.get(job_id)
        if job is None:
            raise CipherGpuError("TRAINING_JOB_NOT_FOUND", "training job was not found", 404)
        return job

    def _clear_training_deks(self, job: TrainingJob) -> None:
        for state in job.inputs.values():
            state.dek[:] = b"\x00" * len(state.dek)

    def _fail_training_job(self, job: TrainingJob, error_code: str) -> None:
        self._clear_training_deks(job)
        self._training_runtime.cleanup_all(job.job_id)
        job.status = "FAILED"
        job.error_code = error_code

    @staticmethod
    def _validate_training_config(adapter_id: str, config: dict[str, Any]) -> None:
        common = {
            "epochs", "learningRate", "trainBatchSize", "evalBatchSize", "mixedPrecision",
            "seed", "maxRuntimeSeconds", "maxSteps",
        }
        adapter_fields = {
            "hf-sequence-classification-v1": {
                "textColumn", "labelColumn", "numLabels", "maxLength", "weightDecay", "warmupRatio",
            },
            "hf-causal-lm-sft-lora-v1": {
                "datasetFormat", "messagesColumn", "maxSequenceLength", "gradientAccumulationSteps",
                "gradientCheckpointing", "assistantOnlyLoss", "packing", "lora",
            },
        }
        if adapter_id not in adapter_fields or set(config) - (common | adapter_fields[adapter_id]):
            raise CipherGpuError("TRAINING_CONFIG_INVALID", "training config contains unsupported fields")
        epochs = float(config.get("epochs", 1))
        learning_rate = float(config.get("learningRate", 2e-5))
        batch = int(config.get("trainBatchSize", 1))
        eval_batch = int(config.get("evalBatchSize", 1))
        timeout = int(config.get("maxRuntimeSeconds", 7200))
        max_steps = int(config.get("maxSteps", -1))
        if not (
            1 <= epochs <= 20
            and 1e-7 <= learning_rate <= 1e-2
            and 1 <= batch <= 128
            and 1 <= eval_batch <= 128
        ):
            raise CipherGpuError("TRAINING_CONFIG_INVALID", "training hyperparameters are out of range")
        if not (60 <= timeout <= 86400):
            raise CipherGpuError("TRAINING_CONFIG_INVALID", "training timeout is out of range")
        if max_steps != -1 and not (1 <= max_steps <= 100000):
            raise CipherGpuError("TRAINING_CONFIG_INVALID", "training maxSteps is out of range")
        if config.get("mixedPrecision", "bf16") not in {"bf16", "fp16"}:
            raise CipherGpuError("TRAINING_CONFIG_INVALID", "training precision is not allowed")
        if adapter_id == "hf-causal-lm-sft-lora-v1":
            lora = config.get("lora", {})
            if not isinstance(lora, dict) or set(lora) - {
                "r", "alpha", "dropout", "targetModules", "bias"
            }:
                raise CipherGpuError("TRAINING_CONFIG_INVALID", "LoRA config is invalid")
            if lora.get("targetModules", "all-linear") != "all-linear":
                raise CipherGpuError("TRAINING_CONFIG_INVALID", "only all-linear LoRA is allowed")
            if (
                config.get("datasetFormat") != "conversational"
                or not str(config.get("messagesColumn", "")).strip()
                or not 64 <= int(config.get("maxSequenceLength", 0)) <= 8192
                or not 1 <= int(config.get("gradientAccumulationSteps", 0)) <= 128
                or not 1 <= int(lora.get("r", 0)) <= 256
                or not 1 <= int(lora.get("alpha", 0)) <= 1024
                or not 0 <= float(lora.get("dropout", -1)) <= 1
                or lora.get("bias", "none") != "none"
            ):
                raise CipherGpuError("TRAINING_CONFIG_INVALID", "LLM SFT config is out of range")
        elif (
            not str(config.get("textColumn", "")).strip()
            or not str(config.get("labelColumn", "")).strip()
            or not 2 <= int(config.get("numLabels", 0)) <= 1000
            or not 32 <= int(config.get("maxLength", 0)) <= 512
            or not 0 <= float(config.get("weightDecay", -1)) <= 1
            or not 0 <= float(config.get("warmupRatio", -1)) <= 1
        ):
            raise CipherGpuError(
                "TRAINING_CONFIG_INVALID", "sequence classification config is out of range"
            )

    def create_session(self, request: AttestationRequest) -> AttestationResponse:
        self._purge()
        if request.expected_security_profile != "a100-sim":
            raise CipherGpuError("SECURITY_PROFILE_UNAVAILABLE", "A100 cannot satisfy gpu-cc-prod")
        if request.runtime_security_requirement == "gpu-cc":
            raise CipherGpuError("SECURITY_DOWNGRADE_DENIED", "gpu-cc assets cannot run on A100 simulation")
        if request.workload_digest != self.workload_digest:
            raise CipherGpuError("WORKLOAD_DIGEST_MISMATCH", "workload digest is not approved by this agent")
        if request.policy_digest != self.policy_digest:
            raise CipherGpuError("EVIDENCE_POLICY_DENIED", "attestation policy digest is not approved")
        key_pair = new_hpke_key_pair()
        session_id = f"tees_{uuid.uuid4().hex}"
        expires_at = now() + timedelta(seconds=request.ttl_seconds)
        session = Session(
            session_id,
            request.domain_id,
            request.task_spec_digest,
            key_pair.public_raw,
            key_pair.private,
            expires_at,
        )
        evidence = {
            "contractVersion": CONTRACT_VERSION,
            "runtimeMode": "SIMULATION",
            "attestationVerified": False,
            "securityProfile": "a100-sim",
            "evidenceType": "SIMULATED_LAB_V1",
            "simulated": True,
            "hardwareModel": "NVIDIA A100",
            "runtimeSecurityRequirement": request.runtime_security_requirement,
            "domainId": request.domain_id,
            "clientNonce": request.client_nonce,
            "taskSpecDigest": request.task_spec_digest,
            "teeEphemeralPublicKeyHash": sha256(key_pair.public_raw),
            "tlsPublicKeyHash": self.tls_public_key_hash,
            "workloadDigest": self.workload_digest,
            "policyDigest": self.policy_digest,
            "sessionId": session_id,
            "issuedAt": iso(now()),
            "expiresAt": iso(expires_at),
        }
        with self._lock:
            self._sessions[session_id] = session
        return AttestationResponse.model_validate(
            {
                "contractVersion": CONTRACT_VERSION,
                "sessionId": session_id,
                "runtimeMode": "SIMULATION",
                "attestationVerified": False,
                "securityProfile": "a100-sim",
                "evidenceType": "SIMULATED_LAB_V1",
                "simulated": True,
                "hardwareModel": "NVIDIA A100",
                "teeEphemeralPublicKey": b64u(key_pair.public_raw),
                "evidence": evidence,
                "evidenceSignature": self.signer.sign_evidence(evidence),
                "evidenceSigningPublicKey": self.signer.public_key,
                "expiresAt": iso(expires_at),
            }
        )

    def execute(self, request: ExecutionRequest) -> ExecutionResponse:
        self._purge()
        task = request.task_spec.model_dump(by_alias=True)
        if task["contractVersion"] != CONTRACT_VERSION:
            raise CipherGpuError("CONTRACT_INVALID", "unsupported contract version")
        if request.task_spec.security_profile != "a100-sim" or request.task_spec.simulated is not True:
            raise CipherGpuError("SECURITY_PROFILE_UNAVAILABLE", "A100 agent accepts only explicit simulation tasks")
        if request.task_spec.evidence_type != "SIMULATED_LAB_V1" or request.task_spec.hardware_model != "NVIDIA A100":
            raise CipherGpuError("EVIDENCE_POLICY_DENIED", "task evidence policy is not the A100 simulation policy")
        if request.task_spec.runtime_security_requirement == "gpu-cc":
            raise CipherGpuError("SECURITY_DOWNGRADE_DENIED", "gpu-cc assets cannot run on A100 simulation")
        if request.task_spec.workload_digest != self.workload_digest:
            raise CipherGpuError("WORKLOAD_DIGEST_MISMATCH", "task workload digest is not approved")
        if request.task_spec.policy_digest != self.policy_digest:
            raise CipherGpuError("EVIDENCE_POLICY_DENIED", "task policy digest is not approved")
        if parse_time(request.task_spec.expires_at) <= now():
            raise CipherGpuError("TASK_EXPIRED", "task specification has expired")
        digest = sha256(canonical(task))
        if digest != request.task_spec_digest:
            raise CipherGpuError("TASK_DIGEST_MISMATCH", "task digest does not match task spec")
        with self._lock:
            session = self._sessions.get(request.session_id)
        if session is None or session.expires_at <= now():
            raise CipherGpuError("ATTESTATION_EXPIRED", "attestation session is unavailable")
        if request.scenario == "KEY_MISMATCH" or session.domain_id != request.task_spec.domain_id:
            raise CipherGpuError("KEY_MATCH_FAILED", "sealed key does not belong to the routed domain")
        if session.task_spec_digest != digest:
            raise CipherGpuError("TASK_DIGEST_MISMATCH", "session is bound to another task")
        if session.consumed:
            raise CipherGpuError("GRANT_REPLAYED", "attestation session was already consumed")

        claims = request.grant.claims.model_dump(by_alias=True)
        try:
            verify_ed25519(request.grant.signing_public_key, request.grant.signature, claims)
        except (InvalidSignature, ValueError) as failure:
            raise CipherGpuError("GRANT_SIGNATURE_INVALID", "grant signature is invalid") from failure
        self._validate_grant(request, session, claims)

        sealed_by_asset = {item.asset_version_id: item for item in request.sealed_deks}
        input_by_asset: dict[str, list[Any]] = {}
        for item in request.encrypted_inputs:
            input_by_asset.setdefault(item.asset_version_id, []).append(item)
        plaintexts: list[bytes] = []
        plaintexts_by_asset: dict[str, list[bytes]] = {}
        try:
            for asset_version_id in request.task_spec.asset_version_ids:
                sealed = sealed_by_asset.get(asset_version_id)
                encrypted_items = input_by_asset.get(asset_version_id)
                if sealed is None or not encrypted_items:
                    raise CipherGpuError("CONTRACT_INVALID", "task input is missing")
                # A single API credential uses one EncryptedInput. Weight
                # manifests may use one EncryptedInput per chunk; reject
                # duplicate or non-contiguous chunk indexes before decrypting.
                if len(encrypted_items) > 1:
                    indexed: list[tuple[int, Any]] = []
                    for item in encrypted_items:
                        if not isinstance(item.aad.get("chunkIndex"), int):
                            raise CipherGpuError("CONTRACT_INVALID", "encrypted chunk index is missing")
                        indexed.append((int(item.aad["chunkIndex"]), item))
                    indexed.sort(key=lambda pair: pair[0])
                    if [index for index, _ in indexed] != list(range(len(indexed))):
                        raise CipherGpuError("CONTRACT_INVALID", "encrypted chunk indexes are not contiguous")
                    encrypted_items = [item for _, item in indexed]
                expected_aad = (
                    f"{digest}|{asset_version_id}|{claims['grantId']}|{claims['exp']}".encode()
                )
                if unb64u(sealed.aad) != expected_aad:
                    raise CipherGpuError("KEY_MATCH_FAILED", "sealed DEK AAD is not bound to this task")
                dek = bytearray(
                    hpke_open(
                        session.private_key,
                        sealed.enc,
                        sealed.ciphertext,
                        expected_aad,
                        HPKE_DEK_INFO,
                    )
                )
                if len(dek) != 32:
                    raise CipherGpuError("CONTRACT_INVALID", "DEK must contain 32 bytes")
                try:
                    for encrypted in encrypted_items:
                        chunks = encrypted.chunks or [encrypted]
                        if not chunks:
                            raise CipherGpuError("CONTRACT_INVALID", "task input has no encrypted chunks")
                        for chunk in chunks:
                            ciphertext = unb64u(chunk.ciphertext)
                            if sha256(ciphertext) != chunk.ciphertext_sha256:
                                raise CipherGpuError("DATA_INTEGRITY_FAILED", "ciphertext digest mismatch")
                            if chunk.format == "ds-envelope/v1":
                                if chunk.algorithm != "AES-256-GCM":
                                    raise CipherGpuError(
                                        "CONTRACT_INVALID", "ds-envelope/v1 accepts only AES-256-GCM"
                                    )
                                plaintext = aes_open(
                                    bytes(dek), chunk.nonce, chunk.ciphertext, chunk.aad
                                )
                            else:
                                if not chunk.envelope_id:
                                    raise CipherGpuError("CONTRACT_INVALID", "v2 envelopeId is required")
                                plaintext = content_open(
                                    bytes(dek),
                                    chunk.envelope_id,
                                    chunk.algorithm,
                                    chunk.nonce,
                                    chunk.ciphertext,
                                    chunk.aad,
                                    chunk.implementation_version,
                                )
                            plaintexts.append(plaintext)
                            plaintexts_by_asset.setdefault(asset_version_id, []).append(plaintext)
                finally:
                    dek[:] = b"\x00" * len(dek)
        except CipherGpuError:
            raise
        except (PyHPKEError, ValueError) as failure:
            raise CipherGpuError("KEY_MATCH_FAILED", "DEK cannot be opened by this session") from failure
        except Exception as failure:
            raise CipherGpuError("DATA_INTEGRITY_FAILED", "encrypted input authentication failed") from failure

        with self._lock:
            if claims["jti"] in self._consumed_jti:
                raise CipherGpuError("GRANT_REPLAYED", "grant jti was already consumed")
            session.consumed = True
            self._consumed_jti.add(claims["jti"])

        execution_id = f"exec_{uuid.uuid4().hex}"
        output_id = f"out_{uuid.uuid4().hex}"
        workload_output = {
            "workloadId": request.task_spec.workload_id,
            "inputs": [
                {
                    "assetVersionId": asset,
                    "sizeBytes": sum(len(value) for value in values),
                    "sha256": sha256(b"".join(values)),
                }
                for asset in request.task_spec.asset_version_ids
                for values in [plaintexts_by_asset[asset]]
            ],
            "status": "VERIFIED",
        }
        model_deployment_status = self._activate_model_deployment(request, plaintexts)
        plaintexts.clear()
        odk = bytearray(os.urandom(32))
        output_aad = {
            "contractVersion": CONTRACT_VERSION,
            "taskId": request.task_spec.task_id,
            "outputId": output_id,
            "taskSpecDigest": digest,
        }
        encrypted_output = aes_seal(bytes(odk), json_bytes(workload_output), output_aad)
        envelopes: list[dict[str, Any]] = []
        recipient_by_kid = {item.kid: item for item in request.output_recipients}
        for kid in request.task_spec.output_recipients:
            recipient = recipient_by_kid.get(kid)
            if recipient is None:
                odk[:] = b"\x00" * len(odk)
                raise CipherGpuError("OUTPUT_RECIPIENT_DENIED", "approved output recipient is missing")
            aad = f"{digest}|{output_id}|{kid}".encode()
            envelope = hpke_seal(recipient.encryption_public_key, bytes(odk), aad, HPKE_ODK_INFO)
            envelopes.append({"recipientKid": kid, "aad": b64u(aad), **envelope})
        odk[:] = b"\x00" * len(odk)
        encrypted_output.update({"outputId": output_id})
        receipt = {
            "contractVersion": CONTRACT_VERSION,
            "executionId": execution_id,
            "taskId": request.task_spec.task_id,
            "taskSpecDigest": digest,
            "sessionId": request.session_id,
            "runtimeMode": "SIMULATION",
            "attestationVerified": False,
            "securityProfile": "a100-sim",
            "evidenceType": "SIMULATED_LAB_V1",
            "simulated": True,
            "hardwareModel": "NVIDIA A100",
            "outputCiphertextSha256": encrypted_output["ciphertextSha256"],
            "completedAt": iso(now()),
        }
        if model_deployment_status is not None:
            receipt["modelDeploymentStatus"] = model_deployment_status
        receipt["signature"] = self.signer.sign_receipt(receipt)
        with self._lock:
            self._receipts[execution_id] = receipt
        return ExecutionResponse.model_validate(
            {
                "contractVersion": CONTRACT_VERSION,
                "executionId": execution_id,
                "status": "SUCCEEDED",
                "runtimeMode": "SIMULATION",
                "attestationVerified": False,
                "encryptedOutput": encrypted_output,
                "keyEnvelopes": envelopes,
                "receipt": receipt,
            }
        )

    def receipt(self, execution_id: str) -> dict[str, Any]:
        with self._lock:
            receipt = self._receipts.get(execution_id)
        if receipt is None:
            raise CipherGpuError("EXECUTION_NOT_FOUND", "execution receipt was not found", 404)
        return receipt

    def cancel(self, execution_id: str) -> dict[str, Any]:
        return {"executionId": execution_id, "status": "CANCELLED", "cancelledAt": iso(now())}

    def infer(self, request: ConfidentialInferenceRequest) -> dict[str, object]:
        self._purge()
        with self._lock:
            deployment = self._model_deployments.get(request.deployment_id)
            session = self._sessions.get(request.session_id)
        if deployment is None or deployment.status != "ONLINE":
            raise CipherGpuError("MODEL_NOT_ONLINE", "model deployment is not online")
        if session is None or session.expires_at <= now() or deployment.session_id != session.session_id:
            raise CipherGpuError("AUTHORIZATION_REQUIRED", "model deployment authorization expired")
        encrypted = request.encrypted_request
        ciphertext = unb64u(encrypted.ciphertext)
        if sha256(ciphertext) != encrypted.cipher_hash:
            raise CipherGpuError("DATA_INTEGRITY_FAILED", "inference ciphertext digest mismatch")
        expected_aad = (
            f"inference|{deployment.deployment_id}|{session.session_id}|{encrypted.cipher_hash}".encode()
        )
        if unb64u(encrypted.sealed_request_key.aad) != expected_aad:
            raise CipherGpuError("KEY_MATCH_FAILED", "request key is not bound to this deployment")
        try:
            request_key = bytearray(
                hpke_open(
                    session.private_key,
                    encrypted.sealed_request_key.enc,
                    encrypted.sealed_request_key.ciphertext,
                    expected_aad,
                    HPKE_REQUEST_KEY_INFO,
                )
            )
            if len(request_key) != 32:
                raise CipherGpuError("CONTRACT_INVALID", "request key must contain 32 bytes")
            plaintext = aes_open(
                bytes(request_key), encrypted.nonce, encrypted.ciphertext, encrypted.aad
            )
            payload = json.loads(plaintext)
            if not isinstance(payload, dict) or not isinstance(payload.get("messages"), list):
                raise CipherGpuError("CONTRACT_INVALID", "encrypted body is not OpenAI chat/completions JSON")
            payload["model"] = deployment.upstream_model_id
            response = self._invoke_model(deployment, payload)
            output_aad = {
                "contractVersion": CONTRACT_VERSION,
                "deploymentId": deployment.deployment_id,
                "sessionId": session.session_id,
                "requestCipherHash": encrypted.cipher_hash,
            }
            output = aes_seal(bytes(request_key), json_bytes(response), output_aad)
            return {
                "contractVersion": CONTRACT_VERSION,
                "deploymentId": deployment.deployment_id,
                "sessionId": session.session_id,
                "runtimeMode": "SIMULATION",
                "attestationVerified": False,
                "securityProfile": "a100-sim",
                "simulated": True,
                "encryptedResponse": output,
            }
        except CipherGpuError:
            raise
        except (PyHPKEError, ValueError) as failure:
            raise CipherGpuError("KEY_MATCH_FAILED", "request key cannot be opened by this session") from failure
        except json.JSONDecodeError as failure:
            raise CipherGpuError("CONTRACT_INVALID", "encrypted request is not valid JSON") from failure
        except Exception as failure:
            raise CipherGpuError("DATA_INTEGRITY_FAILED", "inference request authentication failed") from failure
        finally:
            if "request_key" in locals():
                request_key[:] = b"\x00" * len(request_key)

    def runtime_chat(self, deployment_id: str, payload: dict[str, Any]) -> dict[str, object]:
        """Trusted control-plane proxy for an already-authorized local runtime.

        This endpoint is only exposed on CipherGPU's mTLS listener.  Customer
        bearer keys are validated by SecretPad before a request reaches here.
        """
        with self._lock:
            deployment = self._model_deployments.get(deployment_id)
        local_runtime_missing = (deployment is not None and deployment.source_type == "LOCAL_WEIGHTS"
                                 and not self._runtime.endpoint(deployment_id))
        if deployment is None or deployment.status != "ONLINE" or local_runtime_missing:
            raise CipherGpuError("MODEL_NOT_ONLINE", "model deployment is not online", 503)
        if not isinstance(payload.get("messages"), list):
            raise CipherGpuError("CONTRACT_INVALID", "chat request requires messages")
        payload = dict(payload)
        payload["model"] = deployment.upstream_model_id
        return self._invoke_model(deployment, payload)

    def _activate_model_deployment(
        self, request: ExecutionRequest, plaintexts: list[bytes]
    ) -> str | None:
        prefix = "model.deploy/"
        if not request.task_spec.workload_id.startswith(prefix):
            return None
        deployment_id = request.task_spec.workload_id.removeprefix(prefix)
        with self._lock:
            deployment = self._model_deployments.get(deployment_id)
            if deployment is None:
                raise CipherGpuError("MODEL_DEPLOYMENT_NOT_FOUND", "deployment was not registered")
            if not plaintexts:
                raise CipherGpuError("CONTRACT_INVALID", "deployment authorization has no encrypted material")
            self._clear_deployment_secret(deployment)
            if deployment.source_type == "OPENAI_COMPATIBLE":
                if len(plaintexts[0]) > 8192 or b"\n" in plaintexts[0] or b"\r" in plaintexts[0]:
                    raise CipherGpuError("CONTRACT_INVALID", "decrypted API credential is invalid")
                try:
                    plaintexts[0].decode("utf-8")
                except UnicodeDecodeError as failure:
                    raise CipherGpuError("CONTRACT_INVALID", "decrypted API credential is invalid") from failure
                deployment.secret = bytearray(plaintexts[0])
                deployment.status = "ONLINE"
                deployment.error_code = None
            else:
                # The existing execution contract supplies decrypted bytes.  The
                # control plane's streaming path writes the same archive without
                # retaining it in browser memory; this fallback keeps old small
                # package clients compatible while using the real runtime.
                archive = self._runtime.prepare_archive(deployment_id)
                with archive.open("wb") as output:
                    for plaintext in plaintexts:
                        output.write(plaintext)
                runtime = self._runtime.start(deployment_id, deployment.upstream_model_id,
                                              deployment.timeout_seconds)
                deployment.base_url = f"http://127.0.0.1:{runtime.port}/v1"
                deployment.status = "ONLINE"
                deployment.error_code = None
            deployment.session_id = request.session_id
            return deployment.status

    def _invoke_model(self, deployment: ModelDeployment, payload: dict[str, Any]) -> dict[str, Any]:
        if not deployment.base_url:
            raise CipherGpuError("MODEL_RUNTIME_UNAVAILABLE", "model runtime endpoint is not configured", 503)
        if deployment.source_type == "OPENAI_COMPATIBLE":
            self._validate_remote_url(deployment.base_url)
        url = deployment.base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if deployment.source_type == "OPENAI_COMPATIBLE":
            if deployment.secret is None:
                raise CipherGpuError("AUTHORIZATION_REQUIRED", "model credential is unavailable")
            headers["Authorization"] = "Bearer " + bytes(deployment.secret).decode("utf-8")
        try:
            with httpx.Client(
                timeout=deployment.timeout_seconds,
                follow_redirects=False,
                trust_env=False,
                limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
            ) as client:
                response = client.post(url, json=payload, headers=headers)
            if 300 <= response.status_code < 400:
                raise CipherGpuError("MODEL_PROVIDER_REDIRECT_DENIED", "model provider redirect was denied")
            if response.status_code < 200 or response.status_code >= 300:
                raise CipherGpuError("MODEL_PROVIDER_REJECTED", "upstream model rejected the request", 502)
            if len(response.content) > 10 * 1024 * 1024:
                raise CipherGpuError("MODEL_RESPONSE_TOO_LARGE", "upstream response exceeded the limit", 502)
            body = response.json()
            if not isinstance(body, dict):
                raise CipherGpuError("MODEL_PROVIDER_INVALID_RESPONSE", "upstream response is not an object", 502)
            return body
        except CipherGpuError:
            raise
        except Exception as failure:
            raise CipherGpuError("MODEL_PROVIDER_UNAVAILABLE", "upstream model is unavailable", 502) from failure

    def _validate_remote_url(self, value: str) -> None:
        try:
            parsed = httpx.URL(value)
            if parsed.scheme != "https" or not parsed.host or parsed.userinfo:
                raise ValueError("HTTPS URL required")
            if parsed.fragment:
                raise ValueError("URL fragment denied")
            for result in socket.getaddrinfo(parsed.host, parsed.port or 443, type=socket.SOCK_STREAM):
                address = ipaddress.ip_address(result[4][0])
                if not address.is_global:
                    raise ValueError("non-public address denied")
        except Exception as failure:
            raise CipherGpuError("MODEL_PROVIDER_URL_DENIED", "model provider URL is not allowed") from failure

    def _deployment_view(self, deployment: ModelDeployment) -> dict[str, object]:
        runtime_endpoint = self._runtime.endpoint(deployment.deployment_id)
        return {
            "deploymentId": deployment.deployment_id,
            "sourceType": deployment.source_type,
            "upstreamModelId": deployment.upstream_model_id,
            "securityProfile": "a100-sim",
            "simulated": True,
            "status": deployment.status,
            "sessionId": deployment.session_id,
            "errorCode": deployment.error_code,
            "runtimeActive": runtime_endpoint is not None,
            "runtimePid": self._runtime.pid(deployment.deployment_id),
            "runtimePort": int(runtime_endpoint.rsplit(":", 1)[1].split("/", 1)[0]) if runtime_endpoint else None,
        }

    def _clear_deployment_secret(self, deployment: ModelDeployment) -> None:
        if deployment.secret is not None:
            deployment.secret[:] = b"\x00" * len(deployment.secret)
            deployment.secret = None

    def _validate_grant(self, request: ExecutionRequest, session: Session, claims: dict[str, Any]) -> None:
        current = now()
        if parse_time(claims["nbf"]) > current + timedelta(seconds=30) or parse_time(claims["exp"]) <= current:
            raise CipherGpuError("GRANT_EXPIRED", "grant is outside its validity window")
        if claims["taskSpecDigest"] != request.task_spec_digest:
            raise CipherGpuError("TASK_DIGEST_MISMATCH", "grant is bound to another task")
        if claims["teeSessionId"] != session.session_id:
            raise CipherGpuError("KEY_MATCH_FAILED", "grant is bound to another TEE session")
        if claims["teeEphemeralPublicKeyHash"] != sha256(session.public_key):
            raise CipherGpuError("KEY_MATCH_FAILED", "grant TEK hash mismatch")
        expected_profile = {
            "securityProfile": request.task_spec.security_profile,
            "evidenceType": request.task_spec.evidence_type,
            "simulated": request.task_spec.simulated,
            "hardwareModel": request.task_spec.hardware_model,
            "runtimeSecurityRequirement": request.task_spec.runtime_security_requirement,
        }
        if any(claims[key] != value for key, value in expected_profile.items()):
            raise CipherGpuError("SECURITY_DOWNGRADE_DENIED", "grant security profile does not match task")
        if claims["assetVersionIds"] != request.task_spec.asset_version_ids:
            raise CipherGpuError("GRANT_SCOPE_INVALID", "grant input scope mismatch")
        if claims["outputRecipients"] != request.task_spec.output_recipients:
            raise CipherGpuError("OUTPUT_RECIPIENT_DENIED", "grant output scope mismatch")
        with self._lock:
            if claims["jti"] in self._consumed_jti:
                raise CipherGpuError("GRANT_REPLAYED", "grant jti was already consumed")

    def _purge(self) -> None:
        current = now()
        with self._lock:
            expired = [key for key, value in self._sessions.items() if value.expires_at <= current]
            for key in expired:
                del self._sessions[key]
            for deployment in self._model_deployments.values():
                if deployment.session_id in expired:
                    self._clear_deployment_secret(deployment)
                    deployment.session_id = None
                    deployment.status = "AUTHORIZATION_REQUIRED"
                    deployment.error_code = "ATTESTATION_EXPIRED"
