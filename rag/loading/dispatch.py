from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from langchain_core.documents import Document

from rag.loading._paths import resolve_existing_path
from rag.loading.pdf import load_pdf_documents
from rag.loading.text import is_plain_text_suffix, load_plain_text_documents


def load_file_documents(path: str | Path) -> list[Document]:
    """Route a single file to the correct loader (.pdf or plain text/markdown)."""
    file_path = resolve_existing_path(path)
    suf = file_path.suffix.lower()
    if suf == ".pdf":
        return load_pdf_documents(file_path)
    if is_plain_text_suffix(suf):
        return load_plain_text_documents(file_path)
    msg = (
        f"Unsupported file type {suf!r} for {file_path}. "
        "Supported: .pdf, .txt, .md, .markdown"
    )
    raise ValueError(msg)


def load_path_list_documents(paths: Sequence[str | Path]) -> list[Document]:
    """Load many files (mixed types supported) into one flattened document list."""
    out: list[Document] = []
    for raw in paths:
        out.extend(load_file_documents(raw))
    return out
