"""Vector index construction (Chroma). Logic separate from retrieval and UI."""

from rag.vectorstore.chroma_build import build_chroma_from_documents
from rag.vectorstore.constants import (
    DEFAULT_CHROMA_CHILD_COLLECTION,
    DEFAULT_CHROMA_COLLECTION,
)
from rag.vectorstore.parent_child_build import BuiltParentChildIndex, build_parent_child_index
from rag.vectorstore.serialization import sanitize_chroma_metadata

__all__ = [
    "BuiltParentChildIndex",
    "DEFAULT_CHROMA_CHILD_COLLECTION",
    "DEFAULT_CHROMA_COLLECTION",
    "build_chroma_from_documents",
    "build_parent_child_index",
    "sanitize_chroma_metadata",
]
