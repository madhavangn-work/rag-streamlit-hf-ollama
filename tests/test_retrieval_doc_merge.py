from __future__ import annotations

from langchain_core.documents import Document

from rag.retrieval.doc_merge import dedupe_documents


def test_dedupe_documents_prefers_stable_order_and_id() -> None:
    d0 = Document(page_content="identical snippet", metadata={"src": "a"}, id="x")
    d_dup = Document(page_content="identical snippet", metadata={"src": "a"}, id="x")
    other = Document(page_content="distinct body", metadata={"src": "a"}, id="y")
    merged = dedupe_documents([d0, d_dup, other])
    assert len(merged) == 2
