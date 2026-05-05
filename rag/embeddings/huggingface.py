from __future__ import annotations

from typing import Any

from langchain_huggingface import HuggingFaceEmbeddings

from rag.embeddings.constants import DEFAULT_HF_EMBED_MODEL


def create_huggingface_embeddings(
    *,
    model_name: str | None = None,
    cache_folder: str | None = None,
    model_kwargs: dict[str, Any] | None = None,
    encode_kwargs: dict[str, Any] | None = None,
    query_encode_kwargs: dict[str, Any] | None = None,
    multi_process: bool = False,
    show_progress: bool = False,
) -> HuggingFaceEmbeddings:
    """Local HF Hub embeddings via ``sentence_transformers`` (LangChain wrapper).

    For Nomic checkpoints under ``nomic-ai/``, ``trust_remote_code=True`` is set on the
    SentenceTransformer unless you already supplied it under ``model_kwargs``.
    """

    resolved_name = model_name if model_name is not None else DEFAULT_HF_EMBED_MODEL

    merged_model_kw = dict(model_kwargs or {})
    if (
        resolved_name.startswith("nomic-ai/")
        and "trust_remote_code" not in merged_model_kw
    ):
        merged_model_kw["trust_remote_code"] = True

    kw: dict[str, Any] = {
        "model_name": resolved_name,
        "model_kwargs": merged_model_kw,
        "encode_kwargs": dict(encode_kwargs or {}),
        "query_encode_kwargs": dict(query_encode_kwargs or {}),
        "multi_process": multi_process,
        "show_progress": show_progress,
    }
    if cache_folder is not None:
        kw["cache_folder"] = cache_folder

    return HuggingFaceEmbeddings(**kw)
