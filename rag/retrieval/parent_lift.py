from __future__ import annotations

from collections.abc import Mapping, Sequence

from langchain_core.documents import Document


def lift_parents_from_child_hits(
    child_documents: Sequence[Document],
    parents_by_id: Mapping[str, Document],
    *,
    include_child_if_no_parent_link: bool = True,
) -> list[Document]:
    """Collapse child retrieval hits onto parent passages when ``parent_doc_id`` metadata exists."""

    parents_out: list[Document] = []
    seen_parents: set[str] = set()
    extra_children: list[Document] = []

    for child in child_documents:
        pid_raw = None
        if child.metadata:
            pid_raw = child.metadata.get("parent_doc_id")

        pid = None if pid_raw is None else str(pid_raw)

        if pid and pid in parents_by_id:
            if pid not in seen_parents:
                seen_parents.add(pid)
                parents_out.append(parents_by_id[pid])
            continue

        if include_child_if_no_parent_link:
            extra_children.append(child)

    return parents_out + extra_children

