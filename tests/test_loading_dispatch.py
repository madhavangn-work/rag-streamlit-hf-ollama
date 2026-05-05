from __future__ import annotations

from pathlib import Path

import pytest

from rag.loading.dispatch import load_file_documents, load_path_list_documents


def test_load_file_documents_pdf(minimal_pdf: Path) -> None:
    docs = load_file_documents(minimal_pdf)
    assert len(docs) >= 1


def test_load_file_documents_txt(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_text("dispatch", encoding="utf-8")
    docs = load_file_documents(path)
    assert len(docs) == 1
    assert "dispatch" in docs[0].page_content


def test_load_file_documents_unsupported(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_file_documents(path)


def test_load_path_list_documents_merges_order(
    tmp_path: Path,
    minimal_pdf: Path,
) -> None:
    t1 = tmp_path / "one.txt"
    t1.write_text("first", encoding="utf-8")
    t2 = tmp_path / "two.md"
    t2.write_text("second", encoding="utf-8")
    docs = load_path_list_documents([t1, minimal_pdf, t2])
    assert len(docs) >= 3
    texts = [d.page_content for d in docs]
    assert any("first" in (x or "") for x in texts)
    assert any("second" in (x or "") for x in texts)
