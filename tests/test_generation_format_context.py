from __future__ import annotations

from langchain_core.documents import Document

from rag.generation import format_documents_for_prompt


def test_format_documents_includes_numbers_and_truncation() -> None:
    long = Document(page_content=("x" * 500), metadata={"source": "a.txt"})
    out_full = format_documents_for_prompt([long])
    assert "[1 source=a.txt]" in out_full
    out_trim = format_documents_for_prompt([long], max_chars_per_doc=10)
    assert len(out_trim) < len(out_full)
