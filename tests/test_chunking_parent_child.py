from __future__ import annotations

from langchain_core.documents import Document

from rag.chunking import ParentChildChunkConfig, split_parent_child


def test_split_parent_child_links_children_to_parents() -> None:
    body = "para " * 400
    docs = [Document(page_content=body, metadata={"source": "book.md"})]
    cfg = ParentChildChunkConfig(
        parent_chunk_size=800,
        parent_chunk_overlap=0,
        child_chunk_size=120,
        child_chunk_overlap=10,
    )
    result = split_parent_child(docs, cfg)
    assert result.parent_documents
    assert result.child_documents
    parent_ids = {p.metadata["parent_doc_id"] for p in result.parent_documents}
    assert all(p.metadata["chunk_level"] == "parent" for p in result.parent_documents)
    assert all(c.metadata["chunk_level"] == "child" for c in result.child_documents)
    for child in result.child_documents:
        assert child.metadata["parent_doc_id"] in parent_ids
        assert child.metadata["source"] == "book.md"


def test_split_parent_child_each_parent_has_unique_id() -> None:
    text = "x" * 5000
    docs = [Document(page_content=text, metadata={})]
    result = split_parent_child(
        docs,
        ParentChildChunkConfig(
            parent_chunk_size=300,
            parent_chunk_overlap=0,
            child_chunk_size=80,
            child_chunk_overlap=0,
        ),
    )
    ids = [p.metadata["parent_doc_id"] for p in result.parent_documents]
    assert len(ids) == len(set(ids))
