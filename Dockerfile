# vLLM supplies the CUDA-enabled runtime used by every confidential deployment.
# Keep the control agent in the same container so the plaintext model directory
# never needs to be mounted into a Docker daemon or a second privileged service.
FROM vllm/vllm-openai:v0.10.2

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN groupadd --system --gid 10001 ciphergpu \
    && useradd --system --uid 10001 --gid ciphergpu --home-dir /nonexistent --shell /usr/sbin/nologin ciphergpu

WORKDIR /app
COPY --chown=ciphergpu:ciphergpu pyproject.toml README.md ./
COPY --chown=ciphergpu:ciphergpu ciphergpu ./ciphergpu
RUN pip install ".[training]"

USER 10001:10001
EXPOSE 9000
ENTRYPOINT ["/usr/bin/python3", "-m", "ciphergpu.main"]
