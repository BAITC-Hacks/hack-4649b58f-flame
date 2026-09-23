"""Privacy guard: production meeting data may only be processed locally."""

import os


def local_model_url() -> str:
    url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    if not url.startswith(("http://127.0.0.1:", "http://localhost:")):
        raise ValueError("OLLAMA_BASE_URL must point to localhost for meeting data")
    return url
