from __future__ import annotations

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from rag.embeddings import embed_documents_serial


def test_embed_documents_serial_matches_flat_embed_documents() -> None:
    docs = [
        Document(page_content="alpha", metadata={"i": "0"}),
        Document(page_content="beta", metadata={"i": "1"}),
        Document(page_content="gamma", metadata={"i": "2"}),
    ]
    emb = DeterministicFakeEmbedding(size=8)
    batch_result = embed_documents_serial(emb, docs, batch_size=2)
    flat_result = emb.embed_documents(["alpha", "beta", "gamma"])
    assert batch_result == flat_result


def test_embed_documents_serial_rejects_zero_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        embed_documents_serial(
            DeterministicFakeEmbedding(size=4),
            [Document(page_content="x")],
            batch_size=0,
        )
