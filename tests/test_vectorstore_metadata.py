from __future__ import annotations

from rag.vectorstore import sanitize_chroma_metadata


def test_sanitize_chroma_metadata_drops_none() -> None:
    out = sanitize_chroma_metadata({"a": 1, "b": None, "flag": True})
    assert set(out.keys()) == {"a", "flag"}
    assert out["a"] == 1


def test_sanitize_chroma_metadata_stringifies_unknown_type() -> None:
    class _Odd:
        def __repr__(self) -> str:
            return "odd-x"

    out = sanitize_chroma_metadata({"x": _Odd()})  # type: ignore[dict-item]
    assert out["x"] == "odd-x"
