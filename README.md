# CipherGPU

CipherGPU is the confidential data-plane agent for `ds-confidential/v1`. The current implementation
is a software simulation: it exercises ephemeral X25519 session keys, HPKE key release, encrypted
execution, and encrypted output, but it does not claim hardware attestation or GPU confidential
computing.

Run locally:

```bash
uv run --extra test pytest
uv run ciphergpu
```

Production-style startup requires `CIPHERGPU_TLS_KEY`, `CIPHERGPU_TLS_CERT`, and
`CIPHERGPU_TLS_CA`; the server then requires client certificates. The evidence signing key is a raw
32-byte Ed25519 private key at `CIPHERGPU_EVIDENCE_SIGNING_KEY`. No private key or plaintext is
written to application logs.

The development deployment runs the simulation trust service separately:

```bash
uv run uvicorn ciphergpu.sim_attestation:app --host 0.0.0.0 --port 9100
SIM_ATTESTATION_URL=http://127.0.0.1:9100 uv run ciphergpu
```

Both services always identify evidence as `a100-sim`, `SIMULATED_LAB_V1`, and `simulated=true`.
They reject assets whose runtime requirement is `gpu-cc`.

The Python client keeps the inner request compatible with OpenAI chat completions while encrypting
the entire HTTP body to a short-lived, attested TEK session:

```python
from ciphergpu.client import ConfidentialOpenAIClient, InferenceSession

session = InferenceSession("tees_xxx", "base64url-tek-public-key", "2026-09-02T01:00:00Z")
with ConfidentialOpenAIClient("https://platform.example", "deploy_xxx", session) as client:
    result = client.chat_completions_create(
        model="model-a", messages=[{"role": "user", "content": "hello"}]
    )
```

The remote model provider still receives plaintext after CipherGPU decrypts the request. A100 does
not protect runtime process or GPU memory from a privileged host administrator.
