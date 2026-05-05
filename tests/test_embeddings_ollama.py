from __future__ import annotations

from rag.embeddings import DEFAULT_OLLAMA_EMBED_MODEL, create_ollama_nomic_embeddings


def test_create_ollama_nomic_embeddings_defaults() -> None:
    emb = create_ollama_nomic_embeddings()
    assert emb.model == DEFAULT_OLLAMA_EMBED_MODEL


def test_create_ollama_nomic_embeddings_explicit_model_override() -> None:
    emb = create_ollama_nomic_embeddings(model="other-embed-model")
    assert emb.model == "other-embed-model"
