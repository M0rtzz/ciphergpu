from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ContentEncryptionAlgorithm = Literal[
    "SM4-GCM",
    "AES-256-GCM",
    "AES-256-GCM-SIV",
    "CHACHA20-POLY1305",
    "XCHACHA20-POLY1305",
    "AES-256-SIV",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class AttestationRequest(StrictModel):
    domain_id: str = Field(alias="domainId", min_length=1, max_length=128)
    client_nonce: str = Field(alias="clientNonce", min_length=32, max_length=512)
    task_spec_digest: str = Field(alias="taskSpecDigest", pattern=r"^[0-9a-f]{64}$")
    expected_security_profile: Literal["a100-sim", "gpu-cc-prod"] = Field(alias="expectedSecurityProfile")
    runtime_security_requirement: Literal["gpu-cc", "controlled-sim-ok", "public"] = Field(
        alias="runtimeSecurityRequirement"
    )
    workload_digest: str = Field(alias="workloadDigest", min_length=1, max_length=256)
    policy_digest: str = Field(alias="policyDigest", min_length=1, max_length=256)
    ttl_seconds: int = Field(default=300, alias="ttlSeconds", ge=30, le=300)


class AttestationResponse(StrictModel):
    contract_version: str = Field(alias="contractVersion")
    session_id: str = Field(alias="sessionId")
    runtime_mode: Literal["SIMULATION"] = Field(alias="runtimeMode")
    attestation_verified: Literal[False] = Field(alias="attestationVerified")
    security_profile: Literal["a100-sim"] = Field(alias="securityProfile")
    evidence_type: Literal["SIMULATED_LAB_V1"] = Field(alias="evidenceType")
    simulated: Literal[True]
    hardware_model: Literal["NVIDIA A100"] = Field(alias="hardwareModel")
    tee_ephemeral_public_key: str = Field(alias="teeEphemeralPublicKey")
    evidence: dict[str, Any]
    evidence_signature: str = Field(alias="evidenceSignature")
    evidence_signing_public_key: str = Field(alias="evidenceSigningPublicKey")
    expires_at: str = Field(alias="expiresAt")


class TaskSpec(StrictModel):
    contract_version: str = Field(alias="contractVersion")
    task_id: str = Field(alias="taskId", min_length=1, max_length=128)
    domain_id: str = Field(alias="domainId", min_length=1, max_length=128)
    security_profile: Literal["a100-sim", "gpu-cc-prod"] = Field(alias="securityProfile")
    evidence_type: Literal["SIMULATED_LAB_V1", "GPU_CC_VENDOR"] = Field(alias="evidenceType")
    simulated: bool
    hardware_model: str = Field(alias="hardwareModel")
    runtime_security_requirement: Literal["gpu-cc", "controlled-sim-ok", "public"] = Field(
        alias="runtimeSecurityRequirement"
    )
    purpose: Literal["verify", "infer", "train"]
    workload_id: str = Field(alias="workloadId")
    asset_version_ids: list[str] = Field(alias="assetVersionIds", min_length=1)
    output_recipients: list[str] = Field(alias="outputRecipients", min_length=1)
    attestation_policy_id: str = Field(alias="attestationPolicyId")
    workload_digest: str = Field(alias="workloadDigest")
    policy_digest: str = Field(alias="policyDigest")
    egress_policy: Literal["deny-all"] = Field(alias="egressPolicy")
    issued_at: str = Field(alias="issuedAt")
    expires_at: str = Field(alias="expiresAt")


class GrantClaims(StrictModel):
    contract_version: str = Field(alias="contractVersion")
    grant_id: str = Field(alias="grantId")
    jti: str
    task_spec_digest: str = Field(alias="taskSpecDigest")
    tee_session_id: str = Field(alias="teeSessionId")
    tee_ephemeral_public_key_hash: str = Field(alias="teeEphemeralPublicKeyHash")
    security_profile: Literal["a100-sim", "gpu-cc-prod"] = Field(alias="securityProfile")
    evidence_type: Literal["SIMULATED_LAB_V1", "GPU_CC_VENDOR"] = Field(alias="evidenceType")
    simulated: bool
    hardware_model: str = Field(alias="hardwareModel")
    runtime_security_requirement: Literal["gpu-cc", "controlled-sim-ok", "public"] = Field(
        alias="runtimeSecurityRequirement"
    )
    asset_version_ids: list[str] = Field(alias="assetVersionIds")
    output_recipients: list[str] = Field(alias="outputRecipients")
    nbf: str
    exp: str
    max_uses: Literal[1] = Field(alias="maxUses")


class SignedGrant(StrictModel):
    claims: GrantClaims
    signing_public_key: str = Field(alias="signingPublicKey")
    signature: str


class SealedDek(StrictModel):
    asset_version_id: str = Field(alias="assetVersionId")
    enc: str
    ciphertext: str
    aad: str


# Kept as a separate model so a large weight envelope can carry one entry per
# chunk without changing the existing single-payload API-key shape.
class EncryptedInputChunk(StrictModel):
    format: Literal["ds-envelope/v1", "ds-envelope/v2"] = "ds-envelope/v1"
    envelope_id: str | None = Field(default=None, alias="envelopeId")
    implementation_version: Literal["1"] = Field(default="1", alias="implementationVersion")
    algorithm: ContentEncryptionAlgorithm
    nonce: str
    aad: dict[str, Any]
    ciphertext: str
    ciphertext_sha256: str = Field(alias="ciphertextSha256")


class EncryptedInput(StrictModel):
    asset_version_id: str = Field(alias="assetVersionId")
    format: Literal["ds-envelope/v1", "ds-envelope/v2"] = "ds-envelope/v1"
    envelope_id: str | None = Field(default=None, alias="envelopeId")
    implementation_version: Literal["1"] = Field(default="1", alias="implementationVersion")
    algorithm: ContentEncryptionAlgorithm
    nonce: str
    aad: dict[str, Any]
    ciphertext: str
    ciphertext_sha256: str = Field(alias="ciphertextSha256")
    # For model weights, the top-level fields remain the first chunk for
    # backwards compatibility while all chunks are carried in this list.
    chunks: list[EncryptedInputChunk] | None = None


class OutputRecipient(StrictModel):
    kid: str
    encryption_public_key: str = Field(alias="encryptionPublicKey")


class ExecutionRequest(StrictModel):
    task_spec: TaskSpec = Field(alias="taskSpec")
    task_spec_digest: str = Field(alias="taskSpecDigest")
    session_id: str = Field(alias="sessionId")
    grant: SignedGrant
    sealed_deks: list[SealedDek] = Field(alias="sealedDeks", min_length=1)
    encrypted_inputs: list[EncryptedInput] = Field(alias="encryptedInputs", min_length=1)
    output_recipients: list[OutputRecipient] = Field(alias="outputRecipients", min_length=1)
    scenario: Literal["NORMAL", "KEY_MISMATCH"] = "NORMAL"


class ExecutionResponse(StrictModel):
    contract_version: str = Field(alias="contractVersion")
    execution_id: str = Field(alias="executionId")
    status: Literal["SUCCEEDED"]
    runtime_mode: Literal["SIMULATION"] = Field(alias="runtimeMode")
    attestation_verified: Literal[False] = Field(alias="attestationVerified")
    encrypted_output: dict[str, Any] = Field(alias="encryptedOutput")
    key_envelopes: list[dict[str, Any]] = Field(alias="keyEnvelopes")
    receipt: dict[str, Any]


class ModelDeploymentRequest(StrictModel):
    deployment_id: str = Field(alias="deploymentId", min_length=1, max_length=128)
    source_type: Literal["LOCAL_WEIGHTS", "OPENAI_COMPATIBLE"] = Field(alias="sourceType")
    base_url: str | None = Field(default=None, alias="baseUrl", max_length=2048)
    upstream_model_id: str = Field(alias="upstreamModelId", min_length=1, max_length=256)
    timeout_seconds: int = Field(default=60, alias="timeoutSeconds", ge=5, le=300)
    security_profile: Literal["a100-sim"] = Field(alias="securityProfile")
    simulated: Literal[True]


class SealedRequestKey(StrictModel):
    enc: str
    ciphertext: str
    aad: str


class EncryptedInferencePayload(StrictModel):
    algorithm: Literal["AES-256-GCM"]
    nonce: str
    aad: dict[str, Any]
    ciphertext: str
    cipher_hash: str = Field(alias="cipherHash", pattern=r"^[0-9a-f]{64}$")
    sealed_request_key: SealedRequestKey = Field(alias="sealedRequestKey")


class ConfidentialInferenceRequest(StrictModel):
    deployment_id: str = Field(alias="deploymentId", min_length=1, max_length=128)
    session_id: str = Field(alias="sessionId", min_length=1, max_length=128)
    encrypted_request: EncryptedInferencePayload = Field(alias="encryptedRequest")
