from __future__ import annotations

from langchain_core.documents import Document

from rag.chunking import ChunkConfig, chunk_documents


def test_chunk_documents_splits_long_text() -> None:
    text = "word " * 800
    docs = [
        Document(
            page_content=text,
            metadata={"source": "test.txt"},
        ),
    ]
    cfg = ChunkConfig(chunk_size=40, chunk_overlap=4)
    chunks = chunk_documents(docs, cfg)
    assert len(chunks) > 1
    assert all(doc.metadata["chunk_level"] == "standard" for doc in chunks)
    assert all("chunk_index" in doc.metadata for doc in chunks)
    assert all(doc.metadata["source"] == "test.txt" for doc in chunks)


def test_chunk_documents_preserves_sequence_indices() -> None:
    chunks = chunk_documents(
        [
            Document(page_content="aaaaaaaaaa " * 20, metadata={}),
            Document(page_content="bbbbbbbbbb " * 20, metadata={}),
        ],
        ChunkConfig(chunk_size=30, chunk_overlap=5),
    )
    indexes = [c.metadata["chunk_index"] for c in chunks]
    assert indexes == list(range(len(indexes)))
