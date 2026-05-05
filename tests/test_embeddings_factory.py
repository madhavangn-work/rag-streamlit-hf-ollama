from __future__ import annotations

import pytest
from langchain_ollama import OllamaEmbeddings

from rag.embeddings import build_embeddings


def test_build_embeddings_ollama_returns_ollama() -> None:
    emb = build_embeddings("ollama", ollama_options={"model": "nomic-embed-text"})
    assert isinstance(emb, OllamaEmbeddings)


def test_build_embeddings_ollama_rejects_hf_sidecar() -> None:
    with pytest.raises(ValueError, match="huggingface_options"):
        build_embeddings(
            "ollama",
            huggingface_options={"model_name": "x"},
            ollama_options={},
        )


def test_build_embeddings_huggingface_rejects_ollama_sidecar() -> None:
    with pytest.raises(ValueError, match="ollama_options"):
        build_embeddings(
            "huggingface",
            ollama_options={"model": "x"},
            huggingface_options={"model_name": "sentence-transformers/all-MiniLM-L6-v2"},
        )


def test_build_embeddings_huggingface_routes_options(monkeypatch) -> None:
    import rag.embeddings.huggingface as hf_pkg

    created: dict = {}

    class _Fake:
        def __init__(self, **kw) -> None:
            created.update(kw)

    monkeypatch.setattr(hf_pkg, "HuggingFaceEmbeddings", _Fake)

    build_embeddings(
        "huggingface",
        huggingface_options={"model_name": "my-model"},
    )

    assert created["model_name"] == "my-model"
