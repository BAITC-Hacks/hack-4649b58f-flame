"""Privacy guard: production meeting data may only be processed locally."""

import os
from urllib.parse import urlparse


def validate_local_model_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Ollama endpoint must use http on localhost")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Ollama endpoint credentials are not allowed")
    if parsed.query or parsed.fragment or "?" in url or "#" in url:
        raise ValueError("Ollama endpoint must not contain a query or fragment")
    # Accessing port validates both its numeric format and range.
    if parsed.port == 0:
        raise ValueError("Ollama endpoint port must be positive")
    return url.rstrip("/")


def local_model_url() -> str:
    url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    return validate_local_model_url(url)
