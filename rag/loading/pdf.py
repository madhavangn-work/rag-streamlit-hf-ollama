from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_core.documents import Document

from rag.loading._paths import resolve_existing_path


def load_pdf_documents(path: str | Path) -> list[Document]:
    """Load one PDF file into LangChain Documents (one per page where applicable)."""
    file_path = resolve_existing_path(path)
    if file_path.suffix.lower() != ".pdf":
        msg = f"Expected a .pdf file, got: {file_path}"
        raise ValueError(msg)
    loader = PyPDFLoader(str(file_path))
    return loader.load()
