from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document

from rag.loading._paths import resolve_existing_path
from rag.loading.dispatch import load_file_documents


@dataclass(frozen=True, slots=True)
class DirectoryLoadOptions:
    """Options for scanning a directory for loadable documents."""

    include_pdf: bool = True
    include_plain_text: bool = True


def load_directory_documents(
    directory: str | Path,
    *,
    recursive: bool = True,
    options: DirectoryLoadOptions | None = None,
) -> list[Document]:
    """Load all PDF and plain-text/markdown files under a directory."""
    root = resolve_existing_path(directory)
    if not root.is_dir():
        msg = f"Not a directory: {root}"
        raise NotADirectoryError(msg)

    opts = options or DirectoryLoadOptions()
    return _collect_and_load(root, recursive, opts)


def _collect_and_load(
    root: Path,
    recursive: bool,
    opts: DirectoryLoadOptions,
) -> list[Document]:
    walk = root.rglob("*") if recursive else root.iterdir()
    candidates: list[Path] = []
    for path in walk:
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        if any(part.startswith(".") for part in rel.parts):
            continue
        suf = path.suffix.lower()
        if opts.include_pdf and suf == ".pdf":
            candidates.append(path)
        elif opts.include_plain_text and suf in {".txt", ".md", ".markdown"}:
            candidates.append(path)

    candidates.sort()
    docs: list[Document] = []
    for path in candidates:
        docs.extend(load_file_documents(path))
    return docs
