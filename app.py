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

_CSS = """
    <style>
    :root {
        --muted: rgb(107, 114, 128);
        --border: rgb(229, 231, 235);
        --accent: rgb(59, 130, 246);
    }
    .streamlit-expanderHeader { font-weight: 600; }
    section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
    .product-hero-title {
        font-size: clamp(1.65rem, 2.8vw, 2.05rem);
        font-weight: 650;
        letter-spacing: -0.025em;
        line-height: 1.25;
        margin-bottom: 0.35rem;
        color: rgb(17, 24, 39);
    }
    .product-hero-lead {
        font-size: 1rem;
        color: rgb(75, 85, 99);
        max-width: 52rem;
        line-height: 1.55;
        margin-bottom: 0;
    }
    .status-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.8125rem;
        font-weight: 500;
        padding: 0.2rem 0.65rem;
        border-radius: 999px;
        border: 1px solid var(--border);
        background: rgb(249, 250, 251);
        color: rgb(55, 65, 81);
        margin-bottom: 1rem;
    }
    .status-chip.ready { border-color: rgb(187, 247, 208); background: rgb(240, 253, 244); color: rgb(22, 101, 52); }
    .status-chip.pending { border-color: rgb(254, 240, 138); background: rgb(254, 252, 232); color: rgb(133, 77, 14); }
    hr.product-rule { margin: 1.25rem 0 1.5rem; border: none; border-top: 1px solid var(--border); }
    .section-label {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: rgb(156, 163, 175);
        margin-bottom: 0.5rem;
    }
    .footer-muted { font-size: 0.75rem; color: var(--muted); }
    div[data-testid="stMarkdownContainer"] p { margin-bottom: 0.5rem; }
    </style>
"""


def _inject_styles() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


st.set_page_config(
    page_title="Retriever Studio",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

_inject_styles()

st.markdown('<p class="product-hero-title">Retriever Studio</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="product-hero-lead">Ingest organizational documents, configure embedding and '
    "language backends, then run grounded question answering with configurable retrieval strategies. "
    "All reasoning and storage run in isolated pipeline modules—this interface gathers settings only.</p>",
    unsafe_allow_html=True,
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
    st.markdown('<p class="section-label">Indexing · Embeddings</p>', unsafe_allow_html=True)
    emb_backend = st.radio(
        "Vector embedding provider",
        ["ollama", "huggingface"],
        horizontal=True,
    )
    ollama_embed_models = _ollama_model_choices()
    if emb_backend == "ollama":
        if ollama_embed_models:
            emb_ollama_model = st.selectbox(
                "Embedding model tag",
                ollama_embed_models,
                index=(
                    ollama_embed_models.index(DEFAULT_OLLAMA_EMBED_MODEL)
                    if DEFAULT_OLLAMA_EMBED_MODEL in ollama_embed_models
                    else 0
                ),
            )
        else:
            st.info(
                "The Ollama registry could not be queried from this host. "
                "Confirm the service is reachable, then enter a model tag manually.",
            )
            emb_ollama_model = st.text_input(
                "Embedding model tag",
                value=DEFAULT_OLLAMA_EMBED_MODEL,
            )
        emb_hf_model = DEFAULT_HF_EMBED_MODEL
    else:
        emb_hf_model = st.text_input(
            "Hugging Face model repository",
            value=DEFAULT_HF_EMBED_MODEL,
        )
        emb_ollama_model = DEFAULT_OLLAMA_EMBED_MODEL

    st.divider()

    st.markdown('<p class="section-label">Answering · Language model</p>', unsafe_allow_html=True)
    gen_backend = st.radio(
        "Response provider",
        ["ollama", "huggingface"],
        horizontal=True,
    )
    gen_ollama_models = _ollama_model_choices()
    if gen_backend == "ollama":
        if gen_ollama_models:
            gen_ollama_model = st.selectbox("Chat-capable model", gen_ollama_models)
        else:
            gen_ollama_model = st.text_input("Chat-capable model", value="llama3.2:latest")
        gen_hf_model = DEFAULT_HF_GENERATION_MODEL_ID
    else:
        gen_hf_model = st.text_input(
            "Chat model repository",
            help="Use a suitably sized instruct model for your hardware (≤2B parameters recommended).",
            value=DEFAULT_HF_GENERATION_MODEL_ID,
        )
        gen_ollama_model = "llama3.2:latest"

    st.divider()

    st.markdown('<p class="section-label">Corpus segmentation</p>', unsafe_allow_html=True)
    use_parent = st.checkbox(
        "Hierarchical segmentation (retrieve children, cite parents)",
        value=False,
    )
    if not use_parent:
        c_size = st.number_input(
            "Target chunk length (characters)",
            min_value=64,
            value=512,
            step=32,
        )
        c_overlap = st.number_input(
            "Adjacent chunk overlap (characters)",
            min_value=0,
            value=64,
            step=8,
        )
    else:
        c_size, c_overlap = 512, 64
        with st.expander("Parent and child spans"):
            p_cs = st.number_input("Parent chunk size", value=2000, step=100)
            p_ov = st.number_input("Parent overlap", value=200, step=50)
            ch_cs = st.number_input("Child chunk size", value=400, step=50)
            ch_ov = st.number_input("Child overlap", value=40, step=10)

    st.markdown('<p class="section-label">Query-time retrieval</p>', unsafe_allow_html=True)
    use_mq = st.checkbox("Multi-query rewriting", value=True)
    use_hyde = st.checkbox("Hypothetical document embeddings (HyDE)", value=False)

indexed_ready = st.session_state.indexed_corpus is not None
if indexed_ready:
    st.markdown(
        '<span class="status-chip ready">Search index synchronized</span>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<span class="status-chip pending">Awaiting corpus build</span>',
        unsafe_allow_html=True,
    )

left, right = st.columns((1, 1), gap="large")

with left:
    st.markdown('<p class="section-label">Corpus ingestion</p>', unsafe_allow_html=True)
    with st.container(border=True):
        uploads = st.file_uploader(
            "Select files",
            type=["pdf", "txt", "md", "markdown"],
            accept_multiple_files=True,
            help="Portable document and plain-text markdown sources are normalized for segmentation.",
        )
        build = st.button(
            "Build or rebuild search index",
            type="primary",
            use_container_width=True,
        )

with right:
    st.markdown('<p class="section-label">Grounded question answering</p>', unsafe_allow_html=True)
    with st.container(border=True):
        query = st.text_area(
            "Natural-language question",
            height=136,
            placeholder="Ask a factual question constrained to your uploaded corpus…",
            label_visibility="visible",
        )
        ask = st.button(
            "Generate answer",
            type="primary",
            use_container_width=True,
        )

if build:
    if not uploads:
        st.error("Add at least one supported file before building the search index.")
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

            with st.spinner("Constructing segmented vectors and persistence layer…"):
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
            st.success("Search index operational—proceed to questioning.")
        except Exception as exc:  # noqa: BLE001 — surface to user
            st.exception(exc)

st.markdown('<hr class="product-rule" />', unsafe_allow_html=True)

if ask:
    idx = st.session_state.indexed_corpus
    if idx is None:
        st.warning("Complete ingestion and indexing before submitting questions.")
    elif not query.strip():
        st.warning("Enter a substantive question.")
    else:
        try:
            gen_oo = {}
            gen_ho = {}
            if gen_backend == "ollama":
                gen_oo = {"model": gen_ollama_model}
            else:
                gen_ho = {"model_id": gen_hf_model}

            with st.spinner("Provisioning language session…"):
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

            with st.spinner("Resolving citations and drafting response…"):
                bundle = retrieve_bundle(query.strip(), inp=r_in)
                answer = generate_rag_answer(
                    query.strip(),
                    bundle.context_documents,
                    chat,
                )

            st.markdown("#### Synthesized response")
            st.write(answer)
            with st.expander("Evidence passages", expanded=False):
                for i, doc in enumerate(bundle.context_documents, start=1):
                    src = (doc.metadata or {}).get("source", "")
                    st.markdown(f"**Passage {i}** · `{src}`")
                    st.text((doc.page_content or "")[:4000])
        except Exception as exc:  # noqa: BLE001
            st.exception(exc)

if st.session_state.index_persist_dir:
    st.markdown(
        '<p class="footer-muted">Persisted corpus storage: '
        f'<code>{st.session_state.index_persist_dir}</code></p>',
        unsafe_allow_html=True,
    )
