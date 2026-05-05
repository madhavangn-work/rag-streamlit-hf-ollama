from __future__ import annotations

from typing import Any

from langchain_ollama.chat_models import ChatOllama


def create_chat_ollama(
    *,
    model: str,
    base_url: str | None = None,
    validate_model_on_init: bool = False,
    temperature: float | None = None,
    num_predict: int | None = None,
    client_kwargs: dict[str, Any] | None = None,
    sync_client_kwargs: dict[str, Any] | None = None,
    async_client_kwargs: dict[str, Any] | None = None,
) -> ChatOllama:
    """LangChain chat model backed by a local or remote Ollama server."""

    kw: dict[str, Any] = {
        "model": model,
        "base_url": base_url,
        "validate_model_on_init": validate_model_on_init,
    }
    if temperature is not None:
        kw["temperature"] = temperature
    if num_predict is not None:
        kw["num_predict"] = num_predict
    if client_kwargs is not None:
        kw["client_kwargs"] = client_kwargs
    if sync_client_kwargs is not None:
        kw["sync_client_kwargs"] = sync_client_kwargs
    if async_client_kwargs is not None:
        kw["async_client_kwargs"] = async_client_kwargs

    return ChatOllama(**kw)
