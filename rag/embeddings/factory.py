from __future__ import annotations

from typing import Any, Literal

from langchain_core.embeddings import Embeddings

from rag.embeddings.huggingface import create_huggingface_embeddings
from rag.embeddings.ollama import create_ollama_nomic_embeddings


def build_embeddings(
    backend: Literal["ollama", "huggingface"],
    *,
    ollama_options: dict[str, Any] | None = None,
    huggingface_options: dict[str, Any] | None = None,
) -> Embeddings:
    """Construct a LangChain ``Embeddings`` implementation for the chosen backend.

    Pass only the options dict that matches ``backend``; the other must be empty or ``None``
    so misconfiguration fails fast.
    """
    oo = dict(ollama_options or {})
    ho = dict(huggingface_options or {})

    if backend == "ollama":
        if ho:
            msg = "`huggingface_options` must be empty when backend is ollama"
            raise ValueError(msg)
        return create_ollama_nomic_embeddings(**oo)

    if oo:
        msg = "`ollama_options` must be empty when backend is huggingface"
        raise ValueError(msg)
    return create_huggingface_embeddings(**ho)
