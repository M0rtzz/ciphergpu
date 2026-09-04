from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Self

import httpx
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .crypto import (
    CONTRACT_VERSION,
    HPKE_REQUEST_KEY_INFO,
    b64u,
    canonical,
    hpke_seal,
    sha256,
    unb64u,
)


@dataclass(frozen=True)
class InferenceSession:
    session_id: str
    tee_ephemeral_public_key: str
    expires_at: str


class ConfidentialOpenAIClient:
    """Small OpenAI-shaped client that encrypts the complete request and response bodies."""

    def __init__(
        self,
        base_url: str,
        deployment_id: str,
        session: InferenceSession,
        *,
        headers: dict[str, str] | None = None,
        timeout: float = 120,
        verify: bool | str = True,
    ):
        if not base_url.startswith("https://") and not base_url.startswith("http://127.0.0.1"):
            raise ValueError("SDK base_url must use HTTPS; loopback HTTP is allowed only for development")
        self.deployment_id = deployment_id
        self.session = session
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=timeout,
            verify=verify,
            follow_redirects=False,
            trust_env=False,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def chat_completions_create(
        self, *, model: str, messages: list[dict[str, Any]], **parameters: Any
    ) -> dict[str, Any]:
        if self._expired():
            raise RuntimeError("inference authorization session expired; obtain a new attested TEK")
        body = {"model": model, "messages": messages, **parameters}
        request_key = bytearray(os.urandom(32))
        try:
            outer = self._encrypt_request(bytes(request_key), body)
            response = self._client.post(
                "/api/v1alpha1/confidential-inference/chat/completions", json=outer
            )
            response.raise_for_status()
            wrapped = response.json()
            if isinstance(wrapped, dict) and "status" in wrapped:
                status = wrapped.get("status") or {}
                if status.get("code", 0) not in (0, None):
                    raise RuntimeError(str(status.get("msg") or "confidential inference rejected"))
                wrapped = wrapped.get("data")
            if not isinstance(wrapped, dict):
                raise TypeError("confidential inference returned an invalid envelope")
            return self._decrypt_response(bytes(request_key), wrapped)
        finally:
            request_key[:] = b"\x00" * len(request_key)

    def _encrypt_request(self, request_key: bytes, body: dict[str, Any]) -> dict[str, Any]:
        aad = {
            "contractVersion": CONTRACT_VERSION,
            "deploymentId": self.deployment_id,
            "sessionId": self.session.session_id,
        }
        nonce = os.urandom(12)
        plaintext = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()
        ciphertext = AESGCM(request_key).encrypt(nonce, plaintext, canonical(aad))
        cipher_hash = sha256(ciphertext)
        key_aad = (
            f"inference|{self.deployment_id}|{self.session.session_id}|{cipher_hash}".encode()
        )
        sealed = hpke_seal(
            self.session.tee_ephemeral_public_key,
            request_key,
            key_aad,
            HPKE_REQUEST_KEY_INFO,
        )
        return {
            "deploymentId": self.deployment_id,
            "sessionId": self.session.session_id,
            "encryptedRequest": {
                "algorithm": "AES-256-GCM",
                "nonce": b64u(nonce),
                "aad": aad,
                "ciphertext": b64u(ciphertext),
                "cipherHash": cipher_hash,
                "sealedRequestKey": {
                    "enc": sealed["enc"],
                    "ciphertext": sealed["ciphertext"],
                    "aad": b64u(key_aad),
                },
            },
        }

    def _decrypt_response(self, request_key: bytes, response: dict[str, Any]) -> dict[str, Any]:
        if response.get("deploymentId") != self.deployment_id:
            raise RuntimeError("encrypted response deployment mismatch")
        if response.get("sessionId") != self.session.session_id:
            raise RuntimeError("encrypted response session mismatch")
        encrypted = response.get("encryptedResponse")
        if not isinstance(encrypted, dict):
            raise TypeError("encrypted response payload is missing")
        ciphertext = unb64u(str(encrypted["ciphertext"]))
        if sha256(ciphertext) != encrypted.get("ciphertextSha256"):
            raise RuntimeError("encrypted response digest mismatch")
        plaintext = AESGCM(request_key).decrypt(
            unb64u(str(encrypted["nonce"])), ciphertext, canonical(encrypted["aad"])
        )
        value = json.loads(plaintext)
        if not isinstance(value, dict):
            raise TypeError("decrypted OpenAI response is not an object")
        return value

    def _expired(self) -> bool:
        expiry = datetime.fromisoformat(self.session.expires_at)
        return expiry <= datetime.now(UTC)
