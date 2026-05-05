from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from rag.loading.directory import DirectoryLoadOptions, load_directory_documents


def test_load_directory_documents_recursive_collects_nested(
    tmp_path: Path,
    minimal_pdf: Path,
) -> None:
    sub = tmp_path / "nested"
    sub.mkdir()
    nested_txt = sub / "deep.txt"
    nested_txt.write_text("deep content", encoding="utf-8")
    top_txt = tmp_path / "top.md"
    top_txt.write_text("top content", encoding="utf-8")
    pdf_copy = tmp_path / "copied.pdf"
    shutil.copy2(minimal_pdf, pdf_copy)

    docs = load_directory_documents(tmp_path, recursive=True)
    joined = "\n".join(d.page_content or "" for d in docs)
    assert "deep content" in joined
    assert "top content" in joined
    sources = {d.metadata.get("source") for d in docs}
    assert str(top_txt.resolve()) in sources


def test_load_directory_documents_flat_only_top_level(tmp_path: Path) -> None:
    sub = tmp_path / "nested"
    sub.mkdir()
    (sub / "hidden_from_flat.txt").write_text("nested only", encoding="utf-8")
    (tmp_path / "visible.txt").write_text("top level", encoding="utf-8")

    docs = load_directory_documents(tmp_path, recursive=False)
    contents = "".join(d.page_content or "" for d in docs)
    assert "top level" in contents
    assert "nested only" not in contents


def test_load_directory_documents_skips_hidden_subpaths(tmp_path: Path) -> None:
    hidden_dir = tmp_path / ".venvish"
    hidden_dir.mkdir()
    (hidden_dir / "secret.txt").write_text("secret", encoding="utf-8")
    (tmp_path / "public.txt").write_text("public", encoding="utf-8")

    docs = load_directory_documents(tmp_path, recursive=True)
    contents = "".join(d.page_content or "" for d in docs)
    assert "public" in contents
    assert "secret" not in contents


def test_load_directory_documents_pdf_only_filter(
    tmp_path: Path,
    minimal_pdf: Path,
) -> None:
    shutil.copy2(minimal_pdf, tmp_path / "doc.pdf")
    (tmp_path / "note.txt").write_text("notes", encoding="utf-8")

    opts = DirectoryLoadOptions(include_pdf=True, include_plain_text=False)
    docs = load_directory_documents(tmp_path, recursive=True, options=opts)
    assert all(
        Path(str(d.metadata.get("source"))).suffix.lower() == ".pdf"
        for d in docs
        if d.metadata.get("source")
    )


def test_load_directory_not_a_directory(minimal_pdf: Path) -> None:
    with pytest.raises(NotADirectoryError):
        load_directory_documents(minimal_pdf)


def test_load_directory_documents_missing_path() -> None:
    with pytest.raises(FileNotFoundError, match="Path does not exist"):
        load_directory_documents("/this/path/should/not/exist/rag-test-999")
