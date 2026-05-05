from __future__ import annotations

from collections.abc import Sequence

from langchain_core.documents import Document


def _doc_fingerprint(doc: Document) -> tuple[str | None, str | None, str]:
    cid = getattr(doc, "id", None)
    sid = doc.metadata.get("source") if doc.metadata else None
    prefix = doc.page_content[:240] if doc.page_content else ""
    return (str(cid) if cid else None, str(sid) if sid else None, prefix)


def dedupe_documents(documents: Sequence[Document]) -> list[Document]:
    """Drop near-duplicate retrievals while preserving stable order."""

    ordered: list[Document] = []
    seen: set[tuple[str | None, str | None, str]] = set()
    for doc in documents:
        key = _doc_fingerprint(doc)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(doc)
    return ordered
