from __future__ import annotations

import pytest

from rag.loading._paths import resolve_existing_path


def test_resolve_existing_path_ok(tmp_path) -> None:
    f = tmp_path / "exists.txt"
    f.write_text("x", encoding="utf-8")
    resolved = resolve_existing_path(f)
    assert resolved.is_file()


def test_resolve_existing_path_missing() -> None:
    with pytest.raises(FileNotFoundError, match="Path does not exist"):
        resolve_existing_path("/nonexistent/dir/987654321/nope.pdf")
