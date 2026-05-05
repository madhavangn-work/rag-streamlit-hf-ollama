from __future__ import annotations

from langchain_ollama.embeddings import OllamaEmbeddings

from rag.embeddings.constants import DEFAULT_OLLAMA_EMBED_MODEL


def create_ollama_nomic_embeddings(
    *,
    model: str | None = None,
    base_url: str | None = None,
    validate_model_on_init: bool = False,
    dimensions: int | None = None,
    keep_alive: int | None = None,
    client_kwargs: dict | None = None,
    sync_client_kwargs: dict | None = None,
    async_client_kwargs: dict | None = None,
) -> OllamaEmbeddings:
    """LangChain embeddings client targeting Ollama (default model: ``nomic-embed-text``)."""
    kw: dict = {
        "model": model if model is not None else DEFAULT_OLLAMA_EMBED_MODEL,
        "base_url": base_url,
        "validate_model_on_init": validate_model_on_init,
    }
    if dimensions is not None:
        kw["dimensions"] = dimensions
    if keep_alive is not None:
        kw["keep_alive"] = keep_alive
    if client_kwargs is not None:
        kw["client_kwargs"] = client_kwargs
    if sync_client_kwargs is not None:
        kw["sync_client_kwargs"] = sync_client_kwargs
    if async_client_kwargs is not None:
        kw["async_client_kwargs"] = async_client_kwargs

    return OllamaEmbeddings(**kw)
