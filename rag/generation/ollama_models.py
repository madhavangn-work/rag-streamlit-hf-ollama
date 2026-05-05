from __future__ import annotations

from ollama import Client

from rag.generation.exceptions import OllamaDiscoveryError


def list_installed_ollama_models(*, host: str | None = None) -> list[str]:
    """Return model tags reported by the local Ollama daemon (``ollama list`` API)."""

    try:
        resp = Client(host=host).list()
    except ConnectionError as exc:
        msg = (
            "Could not reach Ollama. Start the daemon (e.g. `ollama serve`) "
            "or set `OLLAMA_HOST` / pass `host=` if it runs elsewhere."
        )
        raise OllamaDiscoveryError(msg) from exc

    tags: list[str] = []
    for entry in getattr(resp, "models", ()) or []:
        name = getattr(entry, "model", None)
        if isinstance(name, str) and name.strip():
            tags.append(name.strip())
    return sorted(set(tags))

