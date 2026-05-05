"""Minimal helpers for uploaded files → temp paths (keeps Streamlit page short)."""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from pathlib import Path


def save_uploads_to_tempdir(
    upload_names: Sequence[str],
    upload_bytes: Sequence[bytes],
) -> list[Path]:
    tmp = Path(tempfile.mkdtemp(prefix="rag_upload_"))
    paths: list[Path] = []
    for name, blob in zip(upload_names, upload_bytes, strict=True):
        safe = Path(name).name.replace(" ", "_")
        destination = tmp / safe
        destination.write_bytes(blob)
        paths.append(destination)
    return paths
