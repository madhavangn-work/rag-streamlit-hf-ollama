from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_ACCEPTED_SCALAR = (str, int, float, bool)


def sanitize_chroma_metadata(metadata: Mapping[str, Any]) -> dict[str, str | int | float | bool]:
    """Flatten metadata values for Chroma (scalars only; omit ``None``)."""
    out: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, _ACCEPTED_SCALAR):
            out[str(key)] = value
        else:
            out[str(key)] = str(value)
    return out
