from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag.vectorstore.constants import DEFAULT_CHROMA_COLLECTION
from rag.vectorstore.serialization import sanitize_chroma_metadata


def _sanitize_document(doc: Document) -> Document:
    safe_meta = sanitize_chroma_metadata(doc.metadata or {})
    return Document(page_content=doc.page_content, metadata=safe_meta, id=doc.id)


def build_chroma_from_documents(
    documents: Sequence[Document],
    embedding: Embeddings,
    *,
    collection_name: str = DEFAULT_CHROMA_COLLECTION,
    persist_directory: str | Path | None = None,
    ids: Sequence[str] | None = None,
    **chromaclient_kwargs,
) -> Chroma:
    """Embed ``documents`` and persist them into a Chroma vector store."""

    docs = list(documents)
    if not docs:
        msg = "documents must be non-empty"
        raise ValueError(msg)

    prepared = [_sanitize_document(d) for d in docs]
    persist = (
        None if persist_directory is None else str(Path(persist_directory).expanduser())
    )
    id_list = list(ids) if ids is not None else None

    return Chroma.from_documents(
        documents=prepared,
        embedding=embedding,
        collection_name=collection_name,
        persist_directory=persist,
        ids=id_list,
        **chromaclient_kwargs,
    )
