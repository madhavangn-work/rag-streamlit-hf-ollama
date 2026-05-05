"""Embedding backends: Ollama (default ``nomic-embed-text``) and Hugging Face Hub."""

from rag.embeddings.batch import embed_documents_serial
from rag.embeddings.constants import (
    DEFAULT_HF_EMBED_MODEL,
    DEFAULT_OLLAMA_EMBED_MODEL,
)
from rag.embeddings.factory import build_embeddings
from rag.embeddings.huggingface import create_huggingface_embeddings
from rag.embeddings.ollama import create_ollama_nomic_embeddings

__all__ = [
    "DEFAULT_HF_EMBED_MODEL",
    "DEFAULT_OLLAMA_EMBED_MODEL",
    "build_embeddings",
    "create_huggingface_embeddings",
    "create_ollama_nomic_embeddings",
    "embed_documents_serial",
]
