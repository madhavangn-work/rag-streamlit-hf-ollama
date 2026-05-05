from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag.vectorstore.parent_child_build import BuiltParentChildIndex

LLMTurnPrompt = Callable[[Mapping[str, str | int]], str]


@dataclass(frozen=True, slots=True)
class RetrievalBundle:
    """Retrieval outcomes for downstream grounding."""

    merged_child_hits: list[Document]
    context_documents: list[Document]


@dataclass(frozen=True, slots=True)
class RetrievalInputs:
    """Everything required to assemble context for a query."""

    child_vectorstore: Chroma
    embeddings: Embeddings
    llm_invoke_multi_query: LLMTurnPrompt | None = None
    llm_invoke_hyde: LLMTurnPrompt | None = None
    multi_query_variant_target: int = 4
    k_per_multi_query: int = 4
    k_hyde: int = 6
    k_direct: int = 8
    parent_index: BuiltParentChildIndex | None = None
    dedupe_parents: bool = True
