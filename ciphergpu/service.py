from __future__ import annotations

import ipaddress
import json
import os
import socket
import ssl
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

import httpx
from cryptography.exceptions import InvalidSignature
from pyhpke import PyHPKEError

from .crypto import (
    CONTRACT_VERSION,
    HPKE_DEK_INFO,
    HPKE_ODK_INFO,
    HPKE_REQUEST_KEY_INFO,
    aes_open,
    aes_seal,
    b64u,
    canonical,
    content_open,
    hpke_open,
    hpke_seal,
    json_bytes,
    new_hpke_key_pair,
    sha256,
    unb64u,
    verify_ed25519,
)
from .models import (
    AttestationRequest,
    AttestationResponse,
    ConfidentialInferenceRequest,
    ExecutionRequest,
    ExecutionResponse,
    ModelDeploymentRequest,
)


class CipherGpuError(Exception):
    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


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
        self._lock = threading.RLock()

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

    def offline_model_deployment(self, deployment_id: str) -> dict[str, object]:
        with self._lock:
            deployment = self._model_deployments.get(deployment_id)
            if deployment is None:
                raise CipherGpuError("MODEL_DEPLOYMENT_NOT_FOUND", "model deployment was not registered", 404)
            self._clear_deployment_secret(deployment)
            deployment.status = "OFFLINE"
            deployment.session_id = None
        return self._deployment_view(deployment)

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
            elif deployment.base_url:
                deployment.status = "ONLINE"
                deployment.error_code = None
            else:
                deployment.status = "RUNTIME_REQUIRED"
                deployment.error_code = "VLLM_ENDPOINT_NOT_CONFIGURED"
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
        return {
            "deploymentId": deployment.deployment_id,
            "sourceType": deployment.source_type,
            "upstreamModelId": deployment.upstream_model_id,
            "securityProfile": "a100-sim",
            "simulated": True,
            "status": deployment.status,
            "sessionId": deployment.session_id,
            "errorCode": deployment.error_code,
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
