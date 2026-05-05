from __future__ import annotations

from dataclasses import dataclass


def _ensure_overlap_below_size(chunk_size: int, chunk_overlap: int, *, label: str) -> None:
    if chunk_size <= 0:
        msg = f"{label}.chunk_size must be positive"
        raise ValueError(msg)
    if chunk_overlap < 0:
        msg = f"{label}.chunk_overlap must be non-negative"
        raise ValueError(msg)
    if chunk_overlap >= chunk_size:
        msg = f"{label}.chunk_overlap must be smaller than chunk_size"
        raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ChunkConfig:
    """Settings for recursive character splitting (single level)."""

    chunk_size: int = 512
    chunk_overlap: int = 64

    def __post_init__(self) -> None:
        _ensure_overlap_below_size(self.chunk_size, self.chunk_overlap, label="ChunkConfig")


@dataclass(frozen=True, slots=True)
class ParentChildChunkConfig:
    """Large parent chunks subdivided into small child chunks for embeddings."""

    parent_chunk_size: int = 2000
    parent_chunk_overlap: int = 200
    child_chunk_size: int = 400
    child_chunk_overlap: int = 40

    def __post_init__(self) -> None:
        _ensure_overlap_below_size(
            self.parent_chunk_size,
            self.parent_chunk_overlap,
            label="ParentChildChunkConfig parent",
        )
        _ensure_overlap_below_size(
            self.child_chunk_size,
            self.child_chunk_overlap,
            label="ParentChildChunkConfig child",
        )
        if self.child_chunk_size > self.parent_chunk_size:
            msg = "child_chunk_size must not exceed parent_chunk_size"
            raise ValueError(msg)
