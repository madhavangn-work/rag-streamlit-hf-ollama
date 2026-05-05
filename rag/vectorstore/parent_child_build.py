from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag.chunking.parent_child import ParentChildSplitResult
from rag.vectorstore.chroma_build import build_chroma_from_documents
from rag.vectorstore.constants import DEFAULT_CHROMA_CHILD_COLLECTION


@dataclass
class BuiltParentChildIndex:
    """Child chunks in Chroma; parent bodies keyed by ``parent_doc_id`` for augmentation."""

    child_vectorstore: Chroma
    parents_by_id: dict[str, Document]


def build_parent_child_index(
    split: ParentChildSplitResult,
    embedding: Embeddings,
    *,
    collection_name: str = DEFAULT_CHROMA_CHILD_COLLECTION,
    persist_directory: str | Path | None = None,
    **chromaclient_kwargs,
) -> BuiltParentChildIndex:
    """Index only **child** chunks in Chroma; keep parent documents aside for retrieval."""
    parents_by_id: dict[str, Document] = {}
    for parent in split.parent_documents:
        pid = parent.metadata.get("parent_doc_id")
        if pid is None:
            msg = "Every parent chunk must expose metadata key 'parent_doc_id'"
            raise ValueError(msg)
        if pid in parents_by_id:
            msg = f"Duplicate parent_doc_id encountered: {pid!r}"
            raise ValueError(msg)
        parents_by_id[str(pid)] = parent

    if not split.child_documents:
        msg = "split.child_documents must be non-empty"
        raise ValueError(msg)

    child_store = build_chroma_from_documents(
        split.child_documents,
        embedding,
        collection_name=collection_name,
        persist_directory=persist_directory,
        **chromaclient_kwargs,
    )

    return BuiltParentChildIndex(
        child_vectorstore=child_store,
        parents_by_id=parents_by_id,
    )
