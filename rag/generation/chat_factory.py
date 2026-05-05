from __future__ import annotations

from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel

from rag.generation.hf_chat import create_chat_huggingface_local
from rag.generation.ollama_chat import create_chat_ollama


def build_chat_model(
    backend: Literal["ollama", "huggingface"],
    *,
    ollama_options: dict[str, Any] | None = None,
    huggingface_options: dict[str, Any] | None = None,
) -> BaseChatModel:
    """Instantiate the configured LangChain ``BaseChatModel`` for answer generation."""

    oo = dict(ollama_options or {})
    ho = dict(huggingface_options or {})

    if backend == "ollama":
        if ho:
            msg = "`huggingface_options` must be empty when backend is ollama"
            raise ValueError(msg)
        return create_chat_ollama(**oo)

    if oo:
        msg = "`ollama_options` must be empty when backend is huggingface"
        raise ValueError(msg)
    return create_chat_huggingface_local(**ho)
