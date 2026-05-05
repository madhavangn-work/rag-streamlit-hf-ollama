"""Document ingestion: LangChain Documents from paths and folders."""

from rag.loading.directory import DirectoryLoadOptions, load_directory_documents
from rag.loading.dispatch import load_file_documents, load_path_list_documents
from rag.loading.pdf import load_pdf_documents
from rag.loading.text import load_plain_text_documents

__all__ = [
    "DirectoryLoadOptions",
    "load_directory_documents",
    "load_file_documents",
    "load_path_list_documents",
    "load_pdf_documents",
    "load_plain_text_documents",
]
