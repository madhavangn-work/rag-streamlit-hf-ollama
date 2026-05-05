from __future__ import annotations

from pathlib import Path

import pytest

from rag.loading.pdf import load_pdf_documents


def test_load_pdf_documents_ok(minimal_pdf: Path) -> None:
    docs = load_pdf_documents(minimal_pdf)
    assert len(docs) >= 1
    assert docs[0].metadata.get("source") == str(minimal_pdf.resolve())


def test_load_pdf_documents_rejects_non_pdf(tmp_path: Path) -> None:
    not_pdf = tmp_path / "fake.pdf.txt"
    not_pdf.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected a .pdf file"):
        load_pdf_documents(not_pdf)
