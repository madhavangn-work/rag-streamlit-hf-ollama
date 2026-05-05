from __future__ import annotations

from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


def embed_documents_serial(
    embedder: Embeddings,
    documents: Sequence[Document],
    *,
    batch_size: int = 32,
) -> list[list[float]]:
    """Produce one embedding vector per document (uses ``page_content`` only)."""

    if batch_size < 1:
        msg = "batch_size must be at least 1"
        raise ValueError(msg)

    texts = [doc.page_content for doc in documents]
    out: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        out.extend(embedder.embed_documents(chunk))
    return out
