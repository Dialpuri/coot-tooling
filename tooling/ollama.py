"""Thread-local Ollama host management for multi-instance load splitting."""
from __future__ import annotations

import os
import threading

_DEFAULT_HOSTS = [
    "http://127.0.0.1:11434",
    "http://127.0.0.1:11435",
]

_env = os.environ.get("OLLAMA_HOSTS")
OLLAMA_HOSTS: list[str] = [h.strip() for h in _env.split(",") if h.strip()] if _env else _DEFAULT_HOSTS

_local = threading.local()


def set_host(host: str) -> None:
    _local.host = host


def get_host() -> str:
    return getattr(_local, "host", OLLAMA_HOSTS[0])


def chat_url() -> str:
    return get_host() + "/api/chat"


def generate_url() -> str:
    return get_host() + "/api/generate"
