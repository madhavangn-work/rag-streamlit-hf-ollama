from __future__ import annotations

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from rag.chunking import ParentChildChunkConfig, split_parent_child
from rag.chunking.parent_child import ParentChildSplitResult
from rag.vectorstore import build_parent_child_index


def test_parent_child_index_exposes_parents_and_children() -> None:
    body = ("Neptune is far from the Sun. " * 30) + "It is an ice giant planet."
    split = split_parent_child(
        [Document(page_content=body, metadata={"source": "planets.txt"})],
        ParentChildChunkConfig(
            parent_chunk_size=400,
            parent_chunk_overlap=0,
            child_chunk_size=80,
            child_chunk_overlap=0,
        ),
    )
    assert split.child_documents

    emb = DeterministicFakeEmbedding(size=24)
    index = build_parent_child_index(
        split,
        emb,
        collection_name="test-parent-child",
    )

    assert index.parents_by_id
    for pid, parent_doc in index.parents_by_id.items():
        assert parent_doc.metadata.get("parent_doc_id") == pid

    hits = index.child_vectorstore.similarity_search(
        "ice giant Neptune",
        k=len(split.child_documents),
    )
    assert hits
    parents_hit = {
        index.parents_by_id[h.metadata["parent_doc_id"]].page_content
        for h in hits
        if h.metadata.get("parent_doc_id")
    }
    assert any("Neptune" in text for text in parents_hit)


def test_parent_child_duplicate_parent_doc_id_raises() -> None:
    pid = "dup-id"
    p1 = Document(
        page_content="a",
        metadata={"parent_doc_id": pid, "chunk_level": "parent"},
    )
    p2 = Document(
        page_content="b",
        metadata={"parent_doc_id": pid, "chunk_level": "parent"},
    )
    kid = Document(
        page_content="child body",
        metadata={"parent_doc_id": pid, "chunk_level": "child"},
    )
    split = ParentChildSplitResult(parent_documents=[p1, p2], child_documents=[kid])
    with pytest.raises(ValueError, match="Duplicate parent_doc_id"):
        build_parent_child_index(split, DeterministicFakeEmbedding(size=16))
