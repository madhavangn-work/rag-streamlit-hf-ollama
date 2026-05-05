from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from rag.chunking import ChunkConfig
from rag.pipeline import ingest_paths_to_index


@pytest.fixture
def deterministic_embeddings(monkeypatch):
    monkeypatch.setattr(
        "rag.pipeline.indexing.build_embeddings",
        lambda *_a, **_kw: DeterministicFakeEmbedding(size=24),
    )


def test_flat_ingest_on_disk_txt(tmp_path: Path, deterministic_embeddings) -> None:
    doc = tmp_path / "notes.txt"
    doc.write_text("TinyLlama is compact and usable for demos.", encoding="utf-8")
    persist = tmp_path / "persist"

    corpus = ingest_paths_to_index(
        [doc],
        persist_directory=persist,
        embedding_backend="huggingface",
        embedding_hf_options={"model_name": "stub-for-build"},
        use_parent_document_retrieval=False,
        chunk_config=ChunkConfig(chunk_size=80, chunk_overlap=10),
    )
    assert corpus.parent_index is None
    hits = corpus.child_vectorstore.similarity_search("TinyLlama compact demos", k=24)
    pooled = "".join(h.page_content for h in hits)
    assert "TinyLlama" in pooled


def test_parent_ingest_provides_parent_index(tmp_path: Path, deterministic_embeddings) -> None:
    doc = tmp_path / "blob.txt"
    doc.write_text(("Sentence alpha. ") * 40 + "Rare marker PDQ_PARENT_TEST.", encoding="utf-8")
    persist = tmp_path / "persist_parent"

    corpus = ingest_paths_to_index(
        [doc],
        persist_directory=persist,
        embedding_backend="ollama",
        embedding_ollama_options={"model": "fake"},
        use_parent_document_retrieval=True,
    )
    assert corpus.parent_index is not None
    assert corpus.parent_index.parents_by_id

