"""Streamlit shell: configures backends and calls modular ``rag`` pipeline code only."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from rag.chunking import ChunkConfig, ParentChildChunkConfig
from rag.embeddings import DEFAULT_HF_EMBED_MODEL, DEFAULT_OLLAMA_EMBED_MODEL
from rag.generation import (
    DEFAULT_HF_GENERATION_MODEL_ID,
    OllamaDiscoveryError,
    adapt_prompt_with_runnable,
    build_chat_model,
    generate_rag_answer,
    list_installed_ollama_models,
)
from rag.pipeline import IndexedCorpus, ingest_paths_to_index
from rag.retrieval import (
    HYDE_PROMPT,
    MULTI_QUERY_PROMPT,
    RetrievalInputs,
    retrieve_bundle,
)
from streamlit_helpers.io import save_uploads_to_tempdir

st.set_page_config(page_title="RAG (Ollama / Hugging Face)", layout="wide")

st.title("Modular RAG demo")
st.caption(
    "Indexing, retrieval, and generation run in `rag/`; this file only wires Streamlit widgets.",
)

if "indexed_corpus" not in st.session_state:
    st.session_state.indexed_corpus: IndexedCorpus | None = None
if "index_persist_dir" not in st.session_state:
    st.session_state.index_persist_dir: str | None = None


def _ollama_model_choices() -> list[str]:
    try:
        return list_installed_ollama_models()
    except OllamaDiscoveryError:
        return []


with st.sidebar:
    st.header("Embedding (index time)")
    emb_backend = st.radio("Embedding backend", ["ollama", "huggingface"], horizontal=True)
    ollama_embed_models = _ollama_model_choices()
    if emb_backend == "ollama":
        if ollama_embed_models:
            emb_ollama_model = st.selectbox(
                "Ollama embedding model",
                ollama_embed_models,
                index=(
                    ollama_embed_models.index(DEFAULT_OLLAMA_EMBED_MODEL)
                    if DEFAULT_OLLAMA_EMBED_MODEL in ollama_embed_models
                    else 0
                ),
            )
        else:
            st.info("Ollama not reachable — enter a model name (e.g. `nomic-embed-text`).")
            emb_ollama_model = st.text_input(
                "Ollama embedding model",
                value=DEFAULT_OLLAMA_EMBED_MODEL,
            )
        emb_hf_model = DEFAULT_HF_EMBED_MODEL
    else:
        emb_hf_model = st.text_input("HF embedding model id", value=DEFAULT_HF_EMBED_MODEL)
        emb_ollama_model = DEFAULT_OLLAMA_EMBED_MODEL

    st.header("Chat (answers + HyDE / multi-query)")
    gen_backend = st.radio("Chat backend", ["ollama", "huggingface"], horizontal=True)
    gen_ollama_models = _ollama_model_choices()
    if gen_backend == "ollama":
        if gen_ollama_models:
            gen_ollama_model = st.selectbox("Ollama chat model", gen_ollama_models)
        else:
            gen_ollama_model = st.text_input("Ollama chat model", value="llama3.2:latest")
        gen_hf_model = DEFAULT_HF_GENERATION_MODEL_ID
    else:
        gen_hf_model = st.text_input(
            "HF chat model (≤2B recommended)",
            value=DEFAULT_HF_GENERATION_MODEL_ID,
        )
        gen_ollama_model = "llama3.2:latest"

    st.header("Chunking & retrieval")
    use_parent = st.checkbox("Parent-document retrieval (larger parent context)", value=False)
    if not use_parent:
        c_size = st.number_input("Chunk size", min_value=64, value=512, step=32)
        c_overlap = st.number_input("Chunk overlap", min_value=0, value=64, step=8)
    else:
        c_size, c_overlap = 512, 64
        with st.expander("Parent / child split sizes"):
            p_cs = st.number_input("Parent chunk size", value=2000, step=100)
            p_ov = st.number_input("Parent overlap", value=200, step=50)
            ch_cs = st.number_input("Child chunk size", value=400, step=50)
            ch_ov = st.number_input("Child overlap", value=40, step=10)

    use_mq = st.checkbox("Multi-query retrieval", value=True)
    use_hyde = st.checkbox("HyDE retrieval", value=False)

    st.header("Index")
    uploads = st.file_uploader(
        "Upload .pdf / .txt / .md",
        type=["pdf", "txt", "md", "markdown"],
        accept_multiple_files=True,
    )
    build = st.button("Build / rebuild index", type="primary")

if build:
    if not uploads:
        st.error("Upload at least one document first.")
    else:
        names = [u.name for u in uploads]
        blobs = [u.getvalue() for u in uploads]
        try:
            tmp_paths = save_uploads_to_tempdir(names, blobs)
            persist = Path(tempfile.mkdtemp(prefix="rag_chroma_"))
            emb_oo = {}
            emb_ho = {}
            if emb_backend == "ollama":
                emb_oo = {"model": emb_ollama_model}
            else:
                emb_ho = {"model_name": emb_hf_model}

            cc = None if use_parent else ChunkConfig(chunk_size=int(c_size), chunk_overlap=int(c_overlap))
            pc = None
            if use_parent:
                pc = ParentChildChunkConfig(
                    parent_chunk_size=int(p_cs),
                    parent_chunk_overlap=int(p_ov),
                    child_chunk_size=int(ch_cs),
                    child_chunk_overlap=int(ch_ov),
                )

            with st.spinner("Indexing (load → chunk → embed → Chroma)…"):
                ic = ingest_paths_to_index(
                    tmp_paths,
                    persist_directory=persist,
                    embedding_backend=emb_backend,
                    embedding_ollama_options=emb_oo,
                    embedding_hf_options=emb_ho,
                    use_parent_document_retrieval=use_parent,
                    chunk_config=cc,
                    parent_child_chunk_config=pc,
                )
            st.session_state.indexed_corpus = ic
            st.session_state.index_persist_dir = str(persist)
            st.success("Index ready.")
        except Exception as exc:  # noqa: BLE001 — surface to user
            st.exception(exc)

st.divider()
query = st.text_area("Question", placeholder="Ask something about your documents…")
ask = st.button("Answer", type="primary")

if ask:
    idx = st.session_state.indexed_corpus
    if idx is None:
        st.warning("Build an index from the sidebar first.")
    elif not query.strip():
        st.warning("Enter a question.")
    else:
        try:
            gen_oo = {}
            gen_ho = {}
            if gen_backend == "ollama":
                gen_oo = {"model": gen_ollama_model}
            else:
                gen_ho = {"model_id": gen_hf_model}

            with st.spinner("Loading chat model…"):
                chat = build_chat_model(
                    gen_backend,
                    ollama_options=gen_oo,
                    huggingface_options=gen_ho,
                )

            mq_fn = adapt_prompt_with_runnable(MULTI_QUERY_PROMPT, chat) if use_mq else None
            hyde_fn = adapt_prompt_with_runnable(HYDE_PROMPT, chat) if use_hyde else None

            r_in = RetrievalInputs(
                child_vectorstore=idx.child_vectorstore,
                embeddings=idx.embeddings,
                llm_invoke_multi_query=mq_fn,
                llm_invoke_hyde=hyde_fn,
                parent_index=idx.parent_index,
            )

            with st.spinner("Retrieving + generating…"):
                bundle = retrieve_bundle(query.strip(), inp=r_in)
                answer = generate_rag_answer(
                    query.strip(),
                    bundle.context_documents,
                    chat,
                )

            st.subheader("Answer")
            st.write(answer)
            with st.expander("Context used"):
                for i, doc in enumerate(bundle.context_documents, start=1):
                    src = (doc.metadata or {}).get("source", "")
                    st.markdown(f"**[{i}]** `{src}`")
                    st.text((doc.page_content or "")[:4000])
        except Exception as exc:  # noqa: BLE001
            st.exception(exc)

if st.session_state.index_persist_dir:
    st.caption(f"Last index persist directory: `{st.session_state.index_persist_dir}`")
