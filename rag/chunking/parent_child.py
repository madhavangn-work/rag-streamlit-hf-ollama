from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.chunking.config import ParentChildChunkConfig


@dataclass(frozen=True, slots=True)
class ParentChildSplitResult:
    """Parents for contextual retrieval / docstore; children for vector lookup."""

    parent_documents: list[Document]
    child_documents: list[Document]


def split_parent_child(
    documents: Sequence[Document],
    config: ParentChildChunkConfig | None = None,
) -> ParentChildSplitResult:
    """Create parent chunks, then subdivide each parent into embedding-sized children."""
    cfg = config or ParentChildChunkConfig()

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.parent_chunk_size,
        chunk_overlap=cfg.parent_chunk_overlap,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.child_chunk_size,
        chunk_overlap=cfg.child_chunk_overlap,
    )

    parent_chunks = parent_splitter.split_documents(list(documents))

    parent_documents: list[Document] = []
    child_documents: list[Document] = []

    for p_idx, parent in enumerate(parent_chunks):
        parent_id = uuid.uuid4().hex
        parent_meta = {
            **parent.metadata,
            "parent_doc_id": parent_id,
            "chunk_level": "parent",
            "parent_chunk_index": p_idx,
        }
        parent_documents.append(
            Document(page_content=parent.page_content, metadata=parent_meta),
        )

        child_source = Document(
            page_content=parent.page_content,
            metadata={
                **parent.metadata,
                "parent_doc_id": parent_id,
                "chunk_level": "child",
                "parent_chunk_index": p_idx,
            },
        )
        splits = child_splitter.split_documents([child_source])
        for c_idx, child in enumerate(splits):
            merged = {
                **child.metadata,
                "parent_doc_id": parent_id,
                "chunk_level": "child",
                "parent_chunk_index": p_idx,
                "child_chunk_index": c_idx,
            }
            child_documents.append(
                Document(page_content=child.page_content, metadata=merged),
            )

    return ParentChildSplitResult(
        parent_documents=parent_documents,
        child_documents=child_documents,
    )
