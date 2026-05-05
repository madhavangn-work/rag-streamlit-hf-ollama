"""Retrieval primitives: HyDE vectors, multi-query expansion, optional parent uplift."""

from rag.retrieval.adapters import adapt_prompt_with_runnable
from rag.retrieval.doc_merge import dedupe_documents
from rag.retrieval.engine import gather_child_documents, retrieve_bundle
from rag.retrieval.parent_lift import lift_parents_from_child_hits
from rag.retrieval.prompts import HYDE_PROMPT, MULTI_QUERY_PROMPT
from rag.retrieval.types import RetrievalBundle, RetrievalInputs

__all__ = [
    "HYDE_PROMPT",
    "MULTI_QUERY_PROMPT",
    "RetrievalBundle",
    "RetrievalInputs",
    "adapt_prompt_with_runnable",
    "dedupe_documents",
    "gather_child_documents",
    "lift_parents_from_child_hits",
    "retrieve_bundle",
]
