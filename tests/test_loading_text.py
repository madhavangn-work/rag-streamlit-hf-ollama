from __future__ import annotations

from pathlib import Path

import pytest

from rag.loading.text import is_plain_text_suffix, load_plain_text_documents


def test_is_plain_text_suffix() -> None:
    assert is_plain_text_suffix(".md")
    assert is_plain_text_suffix(".MD")
    assert not is_plain_text_suffix(".pdf")


def test_load_plain_text_documents_txt(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("Hello RAG\n", encoding="utf-8")
    docs = load_plain_text_documents(path)
    assert len(docs) == 1
    assert "Hello RAG" in docs[0].page_content
    assert docs[0].metadata.get("source") == str(path.resolve())


def test_load_plain_text_documents_rejects_pdf(tmp_path: Path) -> None:
    path = tmp_path / "wrong.pdf"
    path.write_bytes(b"%PDF-1.4")
    with pytest.raises(ValueError, match="Unsupported text extension"):
        load_plain_text_documents(path)
