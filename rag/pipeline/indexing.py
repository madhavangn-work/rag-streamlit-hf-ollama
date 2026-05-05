from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from rag.chunking import ChunkConfig, ParentChildChunkConfig, chunk_documents, split_parent_child
from rag.embeddings.factory import build_embeddings
from rag.loading.dispatch import load_path_list_documents
from rag.vectorstore import (
    DEFAULT_CHROMA_CHILD_COLLECTION,
    DEFAULT_CHROMA_COLLECTION,
    build_chroma_from_documents,
    build_parent_child_index,
)
from rag.vectorstore.parent_child_build import BuiltParentChildIndex


@dataclass
class IndexedCorpus:
    """Vector index ready for retrieval (flat or parent-augmented)."""

    embeddings: Embeddings
    child_vectorstore: Chroma
    parent_index: BuiltParentChildIndex | None


def ingest_paths_to_index(
    source_paths: Sequence[str | Path],
    *,
    persist_directory: str | Path,
    embedding_backend: Literal["ollama", "huggingface"],
    embedding_ollama_options: dict[str, Any] | None = None,
    embedding_hf_options: dict[str, Any] | None = None,
    use_parent_document_retrieval: bool = False,
    chunk_config: ChunkConfig | None = None,
    parent_child_chunk_config: ParentChildChunkConfig | None = None,
    flat_collection_name: str = DEFAULT_CHROMA_COLLECTION,
    child_collection_name: str = DEFAULT_CHROMA_CHILD_COLLECTION,
) -> IndexedCorpus:
    """Load files from disk paths, chunk, embed, and persist into Chroma under ``persist_directory``."""

    paths = list(source_paths)
    if not paths:
        msg = "source_paths must be non-empty"
        raise ValueError(msg)

    persist = Path(persist_directory).expanduser().resolve()
    persist.mkdir(parents=True, exist_ok=True)

    raw_docs = load_path_list_documents(paths)

    embeddings = build_embeddings(
        embedding_backend,
        ollama_options=dict(embedding_ollama_options or {}),
        huggingface_options=dict(embedding_hf_options or {}),
    )

    cc = chunk_config if chunk_config is not None else ChunkConfig()
    pc = (
        parent_child_chunk_config
        if parent_child_chunk_config is not None
        else ParentChildChunkConfig()
    )

    if use_parent_document_retrieval:
        split = split_parent_child(raw_docs, pc)
        parent_bundle = build_parent_child_index(
            split,
            embeddings,
            persist_directory=persist / "chroma_children",
            collection_name=child_collection_name,
        )
        return IndexedCorpus(
            embeddings=embeddings,
            child_vectorstore=parent_bundle.child_vectorstore,
            parent_index=parent_bundle,
        )

    leaf_chunks = chunk_documents(raw_docs, cc)
    chroma_flat = build_chroma_from_documents(
        leaf_chunks,
        embeddings,
        collection_name=flat_collection_name,
        persist_directory=persist / "chroma_flat",
    )
    return IndexedCorpus(
        embeddings=embeddings,
        child_vectorstore=chroma_flat,
        parent_index=None,
    )
