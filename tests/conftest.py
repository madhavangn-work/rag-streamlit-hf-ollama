from __future__ import annotations

from pathlib import Path

import pytest
from pypdf import PdfWriter


@pytest.fixture
def minimal_pdf(tmp_path: Path) -> Path:
    """Single blank-page PDF usable by PyPDFLoader."""
    pdf_path = tmp_path / "sample.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with pdf_path.open("wb") as f:
        writer.write(f)
    return pdf_path
