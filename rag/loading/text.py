from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders.text import TextLoader
from langchain_core.documents import Document

from rag.loading._paths import resolve_existing_path

_TEXT_EXTENSIONS = frozenset({".txt", ".md", ".markdown"})


def is_plain_text_suffix(suffix: str) -> bool:
    return suffix.lower() in _TEXT_EXTENSIONS


def load_plain_text_documents(
    path: str | Path,
    *,
    encoding: str | None = None,
    autodetect_encoding: bool = True,
) -> list[Document]:
    """Load a UTF-8 (or explicitly encoded) plain-text / Markdown file."""
    file_path = resolve_existing_path(path)
    if not is_plain_text_suffix(file_path.suffix):
        msg = f"Unsupported text extension {file_path.suffix!r} for {file_path}"
        raise ValueError(msg)
    kwargs: dict = {"autodetect_encoding": autodetect_encoding}
    if encoding is not None:
        kwargs["encoding"] = encoding
    loader = TextLoader(str(file_path), **kwargs)
    return loader.load()
