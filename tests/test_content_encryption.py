from __future__ import annotations

import os

import pytest
from cryptography.exceptions import InvalidTag
from fastapi.testclient import TestClient
from nacl.exceptions import CryptoError

from ciphergpu.app import create_app
from ciphergpu.crypto import (
    CONTENT_ENCRYPTION_ALGORITHMS,
    b64u,
    content_open,
    content_seal,
    sha256,
)

CROSS_LANGUAGE_VECTORS = {
    "SM4-GCM": "4658b6c6f010881eec7ca35462bd33fb7d61d1c3887baef4e7485b9025f77c13ba2074ec054a46",
    "AES-256-GCM": "61f9a60cec802ae73257be02c2e77cea9111e907ad9bab33b5f949207034741a36d3424f118396",
    "AES-256-GCM-SIV": "9f26ddc094d050f8d8f660c4a454807370f997d33cfa654e42145648447fac8807826412f0f07b",
    "CHACHA20-POLY1305": "41b89493ec73b56bfa32176f8b23a5cd382dafaa279f232305e00832b4a5c20b94820a0dfe178b",
    "XCHACHA20-POLY1305": "aaf5d11ab6713a85a28842203db6007503e57580ddb6d362adb6c57e8ed8879a95a2117c13df15",
    "AES-256-SIV": "8c7ff4754140b84ec2cce6f425bb5a068ff3d0a1365c9608b84f4ed76e6936c00639519c148f48",
}


@pytest.mark.parametrize("algorithm", CONTENT_ENCRYPTION_ALGORITHMS)
def test_content_algorithm_round_trip_and_tamper(algorithm: str) -> None:
    dek = bytes(range(32))
    envelope_id = "env_cross_language_vector_1"
    parameters = CONTENT_ENCRYPTION_ALGORITHMS[algorithm]
    nonce = bytes(range(parameters["nonceSize"]))
    aad = {
        "format": "ds-envelope/v2",
        "envelopeId": envelope_id,
        "contentEncryptionAlgorithm": algorithm,
        "implementationVersion": "1",
        "chunkIndex": 0,
        "plaintextLength": 24,
    }
    plaintext = b"confidential model data"
    ciphertext = content_seal(dek, envelope_id, algorithm, nonce, plaintext, aad)
    assert ciphertext.hex() == CROSS_LANGUAGE_VECTORS[algorithm]

    assert content_open(
        dek,
        envelope_id,
        algorithm,
        b64u(nonce),
        b64u(ciphertext),
        aad,
    ) == plaintext

    tampered = bytearray(ciphertext)
    tampered[-1] ^= 1
    with pytest.raises((InvalidTag, CryptoError, ValueError)):
        content_open(
            dek,
            envelope_id,
            algorithm,
            b64u(nonce),
            b64u(bytes(tampered)),
            aad,
        )


def test_unknown_algorithm_and_nonce_length_fail_closed() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        content_seal(os.urandom(32), "env-1", "AES-CBC", os.urandom(16), b"data", {})
    with pytest.raises(ValueError, match="parameters"):
        content_seal(os.urandom(32), "env-1", "AES-256-GCM", os.urandom(8), b"data", {})


def test_capabilities_publish_authenticated_algorithms() -> None:
    response = TestClient(create_app()).get("/v1/crypto/capabilities")
    assert response.status_code == 200
    body = response.json()
    assert body["format"] == "ds-envelope/v2"
    assert body["defaultAlgorithm"] == "AES-256-GCM"
    assert [item["algorithm"] for item in body["contentEncryptionAlgorithms"]] == list(
        CONTENT_ENCRYPTION_ALGORITHMS
    )
    assert sha256(str(body).encode())
