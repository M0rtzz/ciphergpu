import os

import uvicorn


def run() -> None:
    uvicorn.run(
        "ciphergpu.app:app",
        host=os.environ.get("CIPHERGPU_HOST", "0.0.0.0"),
        port=int(os.environ.get("CIPHERGPU_PORT", "9000")),
        ssl_keyfile=os.environ.get("CIPHERGPU_TLS_KEY"),
        ssl_certfile=os.environ.get("CIPHERGPU_TLS_CERT"),
        ssl_ca_certs=os.environ.get("CIPHERGPU_TLS_CA"),
        ssl_cert_reqs=2 if os.environ.get("CIPHERGPU_TLS_CA") else 0,
        access_log=False,
    )


if __name__ == "__main__":
    run()
