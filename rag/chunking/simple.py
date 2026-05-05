from __future__ import annotations

from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.chunking.config import ChunkConfig

_DEFAULT_CHUNK_LEVEL = "standard"


def chunk_documents(
    documents: Sequence[Document],
    config: ChunkConfig | None = None,
) -> list[Document]:
    """Split documents with a recursive character TextSplitter."""
    cfg = config or ChunkConfig()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )
    chunks = splitter.split_documents(list(documents))
    out: list[Document] = []
    for idx, doc in enumerate(chunks):
        meta = {**doc.metadata, "chunk_index": idx, "chunk_level": _DEFAULT_CHUNK_LEVEL}
        out.append(Document(page_content=doc.page_content, metadata=meta))
    return out
