from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import rfc8785
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, AESGCMSIV, AESSIV, ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat, PublicFormat
from nacl.bindings import (
    crypto_aead_xchacha20poly1305_ietf_decrypt,
    crypto_aead_xchacha20poly1305_ietf_encrypt,
)
from pyhpke import AEADId, CipherSuite, KDFId, KEMId

CONTRACT_VERSION = "ds-confidential/v1"
HPKE_ALGORITHM = "HPKE-Base-X25519-HKDF-SHA256-AES-256-GCM"
HPKE_DEK_INFO = b"ds-confidential/v1/dek"
HPKE_ODK_INFO = b"ds-confidential/v1/odk"
HPKE_REQUEST_KEY_INFO = b"ds-confidential/v1/inference-request-key"

CONTENT_ENCRYPTION_ALGORITHMS = {
    "SM4-GCM": {"keySize": 16, "nonceSize": 12, "tagSize": 16},
    "AES-256-GCM": {"keySize": 32, "nonceSize": 12, "tagSize": 16},
    "AES-256-GCM-SIV": {"keySize": 32, "nonceSize": 12, "tagSize": 16},
    "CHACHA20-POLY1305": {"keySize": 32, "nonceSize": 12, "tagSize": 16},
    "XCHACHA20-POLY1305": {"keySize": 32, "nonceSize": 24, "tagSize": 16},
    "AES-256-SIV": {"keySize": 64, "nonceSize": 16, "tagSize": 16},
}


def b64u(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def unb64u(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def canonical(value: Any) -> bytes:
    return rfc8785.dumps(value)


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def hpke_suite() -> CipherSuite:
    return CipherSuite.new(
        KEMId.DHKEM_X25519_HKDF_SHA256,
        KDFId.HKDF_SHA256,
        AEADId.AES256_GCM,
    )


@dataclass
class HpkeKeyPair:
    private: Any
    public_raw: bytes


def new_hpke_key_pair() -> HpkeKeyPair:
    suite = hpke_suite()
    pair = suite.kem.derive_key_pair(os.urandom(32))
    return HpkeKeyPair(pair.private_key, pair.public_key.to_public_bytes())


def hpke_open(private_key: Any, enc: str, ciphertext: str, aad: bytes, info: bytes) -> bytes:
    suite = hpke_suite()
    context = suite.create_recipient_context(unb64u(enc), private_key, info=info)
    return context.open(unb64u(ciphertext), aad=aad)


def hpke_seal(public_key: str, plaintext: bytes, aad: bytes, info: bytes) -> dict[str, str]:
    suite = hpke_suite()
    recipient = suite.kem.deserialize_public_key(unb64u(public_key))
    enc, context = suite.create_sender_context(recipient, info=info)
    ciphertext = context.seal(plaintext, aad=aad)
    return {
        "algorithm": HPKE_ALGORITHM,
        "enc": b64u(enc),
        "ciphertext": b64u(ciphertext),
    }


def aes_open(key: bytes, nonce: str, ciphertext: str, aad: dict[str, Any]) -> bytes:
    return AESGCM(key).decrypt(unb64u(nonce), unb64u(ciphertext), canonical(aad))


def content_capabilities() -> list[dict[str, Any]]:
    return [
        {
            "algorithm": algorithm,
            **parameters,
            "keyDerivation": "HKDF-SHA256",
            "implementationVersion": "1",
            "enabled": True,
            "recommended": algorithm == "AES-256-GCM",
        }
        for algorithm, parameters in CONTENT_ENCRYPTION_ALGORITHMS.items()
    ]


def derive_content_key(dek: bytes, envelope_id: str, algorithm: str, implementation_version: str = "1") -> bytes:
    parameters = CONTENT_ENCRYPTION_ALGORITHMS.get(algorithm)
    if parameters is None:
        raise ValueError("unsupported content encryption algorithm")
    if len(dek) != 32:
        raise ValueError("DEK must contain 32 bytes")
    if implementation_version != "1":
        raise ValueError("unsupported content encryption implementation version")
    return HKDF(
        algorithm=hashes.SHA256(),
        length=parameters["keySize"],
        salt=f"ds-envelope/v2:{envelope_id}".encode(),
        info=f"content-key:{algorithm}:{implementation_version}".encode(),
    ).derive(dek)


def content_seal(
    dek: bytes,
    envelope_id: str,
    algorithm: str,
    nonce: bytes,
    plaintext: bytes,
    aad: Any,
    implementation_version: str = "1",
) -> bytes:
    parameters = CONTENT_ENCRYPTION_ALGORITHMS.get(algorithm)
    if parameters is None:
        raise ValueError("unsupported content encryption algorithm")
    if len(nonce) != parameters["nonceSize"]:
        raise ValueError("content encryption parameters are invalid")
    key = bytearray(derive_content_key(dek, envelope_id, algorithm, implementation_version))
    aad_bytes = canonical(aad)
    try:
        if algorithm == "SM4-GCM":
            encryptor = Cipher(algorithms.SM4(bytes(key)), modes.GCM(nonce)).encryptor()
            encryptor.authenticate_additional_data(aad_bytes)
            return encryptor.update(plaintext) + encryptor.finalize() + encryptor.tag
        if algorithm == "AES-256-GCM":
            return AESGCM(bytes(key)).encrypt(nonce, plaintext, aad_bytes)
        if algorithm == "AES-256-GCM-SIV":
            return AESGCMSIV(bytes(key)).encrypt(nonce, plaintext, aad_bytes)
        if algorithm == "CHACHA20-POLY1305":
            return ChaCha20Poly1305(bytes(key)).encrypt(nonce, plaintext, aad_bytes)
        if algorithm == "XCHACHA20-POLY1305":
            return crypto_aead_xchacha20poly1305_ietf_encrypt(plaintext, aad_bytes, nonce, bytes(key))
        return AESSIV(bytes(key)).encrypt(plaintext, [aad_bytes, nonce])
    finally:
        key[:] = b"\x00" * len(key)


def content_open(
    dek: bytes,
    envelope_id: str,
    algorithm: str,
    nonce: str,
    ciphertext: str,
    aad: Any,
    implementation_version: str = "1",
) -> bytes:
    parameters = CONTENT_ENCRYPTION_ALGORITHMS.get(algorithm)
    nonce_bytes = unb64u(nonce)
    if parameters is None:
        raise ValueError("unsupported content encryption algorithm")
    if len(nonce_bytes) != parameters["nonceSize"]:
        raise ValueError("content encryption parameters are invalid")
    key = bytearray(derive_content_key(dek, envelope_id, algorithm, implementation_version))
    aad_bytes = canonical(aad)
    ciphertext_bytes = unb64u(ciphertext)
    try:
        if algorithm == "SM4-GCM":
            if len(ciphertext_bytes) < 16:
                raise ValueError("SM4-GCM ciphertext is shorter than the authentication tag")
            decryptor = Cipher(algorithms.SM4(bytes(key)), modes.GCM(nonce_bytes, ciphertext_bytes[-16:])).decryptor()
            decryptor.authenticate_additional_data(aad_bytes)
            return decryptor.update(ciphertext_bytes[:-16]) + decryptor.finalize()
        if algorithm == "AES-256-GCM":
            return AESGCM(bytes(key)).decrypt(nonce_bytes, ciphertext_bytes, aad_bytes)
        if algorithm == "AES-256-GCM-SIV":
            return AESGCMSIV(bytes(key)).decrypt(nonce_bytes, ciphertext_bytes, aad_bytes)
        if algorithm == "CHACHA20-POLY1305":
            return ChaCha20Poly1305(bytes(key)).decrypt(nonce_bytes, ciphertext_bytes, aad_bytes)
        if algorithm == "XCHACHA20-POLY1305":
            return crypto_aead_xchacha20poly1305_ietf_decrypt(
                ciphertext_bytes, aad_bytes, nonce_bytes, bytes(key)
            )
        return AESSIV(bytes(key)).decrypt(ciphertext_bytes, [aad_bytes, nonce_bytes])
    finally:
        key[:] = b"\x00" * len(key)


def aes_seal(key: bytes, plaintext: bytes, aad: dict[str, Any]) -> dict[str, Any]:
    nonce = os.urandom(12)
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, canonical(aad))
    return {
        "algorithm": "AES-256-GCM",
        "nonce": b64u(nonce),
        "aad": aad,
        "ciphertext": b64u(ciphertext),
        "ciphertextSha256": sha256(ciphertext),
    }


def verify_ed25519(public_key: str, signature: str, value: Any) -> None:
    Ed25519PublicKey.from_public_bytes(unb64u(public_key)).verify(unb64u(signature), canonical(value))


class EvidenceSigner:
    def __init__(self, private_key: Ed25519PrivateKey):
        self._private_key = private_key

    @classmethod
    def load(cls, path: str | None) -> EvidenceSigner:
        if path:
            raw = Path(path).read_bytes()
            return cls(Ed25519PrivateKey.from_private_bytes(raw))
        return cls(Ed25519PrivateKey.generate())

    @property
    def public_key(self) -> str:
        raw = self._private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        return b64u(raw)

    def sign(self, payload: Any) -> str:
        return b64u(self._private_key.sign(canonical(payload)))

    def sign_evidence(self, payload: Any) -> str:
        return self.sign(payload)

    def sign_receipt(self, payload: Any) -> str:
        return self.sign(payload)

    def private_bytes(self) -> bytes:
        return self._private_key.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
