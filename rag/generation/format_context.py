from __future__ import annotations

from collections.abc import Sequence

from langchain_core.documents import Document


def format_documents_for_prompt(
    documents: Sequence[Document],
    *,
    max_chars_per_doc: int | None = None,
) -> str:
    """Flatten retrieved documents into numbered context blocks for the chat prompt."""

    blocks: list[str] = []
    for i, doc in enumerate(documents, start=1):
        body = doc.page_content or ""
        if max_chars_per_doc is not None:
            body = body[:max_chars_per_doc]
        meta = doc.metadata if doc.metadata else {}
        sid = meta.get("source", "")
        head = f"[{i}]" if not sid else f"[{i} source={sid}]"
        blocks.append(f"{head}\n{body.strip()}")
    return "\n\n---\n\n".join(blocks) if blocks else "(No context passages retrieved.)"

