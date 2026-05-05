from __future__ import annotations

import pytest

from rag.chunking.config import ChunkConfig, ParentChildChunkConfig


def test_chunk_config_valid_defaults() -> None:
    c = ChunkConfig()
    assert c.chunk_size == 512


def test_chunk_config_rejects_overlap_too_large() -> None:
    with pytest.raises(ValueError, match="chunk_overlap must be smaller"):
        ChunkConfig(chunk_size=100, chunk_overlap=100)


def test_parent_child_config_rejects_child_larger_than_parent() -> None:
    with pytest.raises(ValueError, match="child_chunk_size must not exceed"):
        ParentChildChunkConfig(
            parent_chunk_size=200,
            parent_chunk_overlap=0,
            child_chunk_size=400,
            child_chunk_overlap=0,
        )
