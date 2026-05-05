from __future__ import annotations

from rag.retrieval.query_parsers import split_numbered_llm_lines


def test_split_numbered_primary_path() -> None:
    txt = """1. alpha variant query\n2. beta wording\n4. gamma final\nnoise line"""
    qs = split_numbered_llm_lines(txt, limit=5)
    assert qs == ["alpha variant query", "beta wording", "gamma final"]


def test_split_numbered_fallback_lines() -> None:
    qs = split_numbered_llm_lines("only-one-line\nanother", limit=10)
    assert qs == ["only-one-line", "another"]
