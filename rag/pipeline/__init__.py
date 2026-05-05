"""High-level ingestion + indexing compositions for apps (Streamlit hooks here)."""

from rag.pipeline.indexing import IndexedCorpus, ingest_paths_to_index

__all__ = ["IndexedCorpus", "ingest_paths_to_index"]
