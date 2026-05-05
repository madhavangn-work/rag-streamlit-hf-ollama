from __future__ import annotations

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from rag.vectorstore import build_chroma_from_documents


def test_build_chroma_from_documents_similarity_roundtrip() -> None:
    docs = [
        Document(
            page_content="The capital of France is Paris.",
            metadata={"topic": "geography"},
        ),
        Document(
            page_content="Python is a programming language.",
            metadata={"topic": "tech"},
        ),
    ]
    emb = DeterministicFakeEmbedding(size=32)
    store = build_chroma_from_documents(docs, emb, collection_name="test-flat")
    hits = store.similarity_search(
        query="Tell me what language people use for data science?",
        k=len(docs),
    )
    corpus = "".join(h.page_content for h in hits)
    assert "Python" in corpus


def test_build_chroma_from_documents_requires_nonempty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        build_chroma_from_documents(
            [],
            DeterministicFakeEmbedding(size=16),
            collection_name="empty",
        )
