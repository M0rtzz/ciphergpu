FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN groupadd --system --gid 10001 ciphergpu \
    && useradd --system --uid 10001 --gid ciphergpu --home-dir /nonexistent --shell /usr/sbin/nologin ciphergpu

WORKDIR /app
COPY --chown=ciphergpu:ciphergpu pyproject.toml README.md ./
COPY --chown=ciphergpu:ciphergpu ciphergpu ./ciphergpu
RUN pip install .

USER 10001:10001
EXPOSE 9000
ENTRYPOINT ["python", "-m", "ciphergpu.main"]
