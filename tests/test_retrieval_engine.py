from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from rag.chunking import ParentChildChunkConfig, split_parent_child
from rag.retrieval.engine import gather_child_documents, retrieve_bundle
from rag.retrieval.types import RetrievalInputs
from rag.vectorstore import build_chroma_from_documents, build_parent_child_index


def _mk_store(embedder: DeterministicFakeEmbedding) -> tuple:
    docs = [
        Document(
            page_content="Rare token QUARK99 describes subatomic physics matter.",
            metadata={"label": "physics"},
        ),
        Document(
            page_content="The Rare token BOTANY77 covers plant morphology facts.",
            metadata={"label": "biology"},
        ),
    ]
    return build_chroma_from_documents(docs, embedder, collection_name="retrieve-store"), docs


def test_gather_child_documents_direct_fallback() -> None:
    embedder = DeterministicFakeEmbedding(size=16)
    store, docs = _mk_store(embedder)
    inp = RetrievalInputs(
        child_vectorstore=store,
        embeddings=embedder,
    )
    ctx = gather_child_documents("Discuss QUARK99 and fundamental particles.", inp=inp)
    pooled = "".join(d.page_content for d in ctx)
    assert docs[0].page_content.strip()[:20] in pooled or "QUARK99" in pooled


def test_gather_child_documents_multi_query_and_hyde() -> None:
    embedder = DeterministicFakeEmbedding(size=16)
    store, docs = _mk_store(embedder)

    def mq(_: dict) -> str:
        return (
            "1. Questions about BOTANY77 plant morphology\n"
            "2. Follow ups on QUARK99 physics\n"
        )

    def hyde(_: dict) -> str:
        # Hypothetical answer text should embed nearer to BOTANY passage.
        return "BOTANY77 plants structures leaves stems roots textbook summary."

    inp = RetrievalInputs(
        child_vectorstore=store,
        embeddings=embedder,
        llm_invoke_multi_query=mq,
        llm_invoke_hyde=hyde,
        k_per_multi_query=2,
        k_hyde=4,
        k_direct=8,
    )
    merged = gather_child_documents("anything", inp=inp)
    content = "".join(d.page_content for d in merged)
    assert docs[1].metadata["label"] == "biology"
    assert "BOTANY77" in content
    assert "QUARK99" in content


def test_retrieve_bundle_resolves_parents() -> None:
    embedder = DeterministicFakeEmbedding(size=12)
    long_body = (
        "Rare Neptune token ZZT9 anchors observations. "
        + ("Neptune planetary science detail. " * 30)
    )
    split = split_parent_child(
        [
            Document(
                page_content=long_body,
                metadata={"source": "facts.txt"},
            ),
        ],
        ParentChildChunkConfig(
            parent_chunk_size=380,
            parent_chunk_overlap=0,
            child_chunk_size=72,
            child_chunk_overlap=0,
        ),
    )
    idx = build_parent_child_index(split, embedder, collection_name="retrieve-parent-store")

    inp = RetrievalInputs(
        child_vectorstore=idx.child_vectorstore,
        embeddings=embedder,
        parent_index=idx,
        k_direct=len(split.child_documents),
    )

    bundle = retrieve_bundle("Tell me about ZZT9 and Neptune science.", inp=inp)
    assert bundle.context_documents
    joined_parents = "".join(d.page_content for d in bundle.context_documents)
    assert "ZZT9" in joined_parents


def test_parent_lift_keeps_children_without_parent_link() -> None:
    from rag.retrieval.parent_lift import lift_parents_from_child_hits

    parent = Document(
        page_content="parent big",
        metadata={"parent_doc_id": "p1"},
    )
    child = Document(
        page_content="child small",
        metadata={"parent_doc_id": "p1"},
    )
    orphans = Document(page_content="orphan chunk", metadata={})
    out = lift_parents_from_child_hits(
        [child, orphans],
        {"p1": parent},
        include_child_if_no_parent_link=True,
    )
    assert any("parent big" in d.page_content for d in out)
    assert any("orphan chunk" in d.page_content for d in out)

