from __future__ import annotations

from langchain_core.documents import Document

from rag.retrieval.doc_merge import dedupe_documents
from rag.retrieval.parent_lift import lift_parents_from_child_hits
from rag.retrieval.query_parsers import split_numbered_llm_lines
from rag.retrieval.types import RetrievalBundle, RetrievalInputs


def gather_child_documents(question: str, *, inp: RetrievalInputs) -> list[Document]:
    """Vector search(es) executed against stored **child** chunks."""

    pooled: list[Document] = []
    ran_multi = inp.llm_invoke_multi_query is not None
    ran_hyde = inp.llm_invoke_hyde is not None

    if ran_multi:
        assert inp.llm_invoke_multi_query is not None  # narrowing
        mq_text = inp.llm_invoke_multi_query(
            {
                "question": question,
                "n_queries": inp.multi_query_variant_target,
            },
        )
        variants = split_numbered_llm_lines(
            mq_text,
            limit=inp.multi_query_variant_target,
        )
        trimmed = [v for v in variants if v.strip()]
        lookup_order = list(dict.fromkeys([question, *trimmed]))

        for q in lookup_order:
            pooled.extend(
                inp.child_vectorstore.similarity_search(
                    q,
                    k=inp.k_per_multi_query,
                ),
            )

    if ran_hyde:
        assert inp.llm_invoke_hyde is not None
        hypo = inp.llm_invoke_hyde({"question": question}).strip()
        vec = inp.embeddings.embed_query(hypo)
        pooled.extend(
            inp.child_vectorstore.similarity_search_by_vector(
                vec,
                k=inp.k_hyde,
            ),
        )

    if not ran_multi and not ran_hyde:
        pooled.extend(
            inp.child_vectorstore.similarity_search(
                question,
                k=inp.k_direct,
            ),
        )

    return dedupe_documents(pooled)


def retrieve_bundle(question: str, *, inp: RetrievalInputs) -> RetrievalBundle:
    """Full retrieval incl. dedupe plus optional promotion to parents."""

    merged_children = gather_child_documents(question, inp=inp)

    idx = inp.parent_index
    if idx is None:
        return RetrievalBundle(
            merged_child_hits=merged_children,
            context_documents=merged_children,
        )

    parent_docs = lift_parents_from_child_hits(
        merged_children,
        idx.parents_by_id,
        include_child_if_no_parent_link=True,
    )
    contexts = dedupe_documents(parent_docs) if inp.dedupe_parents else parent_docs
    return RetrievalBundle(
        merged_child_hits=merged_children,
        context_documents=contexts,
    )
