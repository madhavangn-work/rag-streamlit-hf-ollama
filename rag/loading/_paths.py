from __future__ import annotations

from pathlib import Path


def resolve_existing_path(path: str | Path) -> Path:
    """Return an absolute Path; raise FileNotFoundError if missing."""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        msg = f"Path does not exist: {p}"
        raise FileNotFoundError(msg)
    return p
