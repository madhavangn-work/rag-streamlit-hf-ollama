# Retriever Studio

A modular **retrieval-augmented generation (RAG)** system with a **Streamlit** front end. Core pipeline code lives under `rag/` and is kept separate from UI concerns. You can run embeddings and chat models through **Ollama** or **Hugging Face** (local `sentence-transformers` / transformers pipelines), use **Chroma** as the vector store, and enable **multi-query retrieval**, **HyDE**, and **parent–document** context expansion.

---

## Table of contents

- [Features](#features)
- [Requirements](#requirements)
- [Reproducing the project from GitHub](#reproducing-the-project-from-github)
- [Running the application](#running-the-application)
- [Using the UI](#using-the-ui)
- [Configuration overview](#configuration-overview)
- [Running tests](#running-tests)
- [Project layout](#project-layout)
- [Environment variables](#environment-variables)
- [Troubleshooting](#troubleshooting)

---

## Features

| Area | Details |
|------|---------|
| **Document loading** | PDF (via PyPDF / LangChain loaders), plain text, Markdown; single files, path lists, or directory scans. |
| **Chunking** | Recursive character splitting; optional **parent / child** chunks for parent-document retrieval. |
| **Embeddings** | **Ollama** (`nomic-embed-text` default) or **Hugging Face Hub** via LangChain `HuggingFaceEmbeddings` (default Nomic v1.5–style checkpoint; `einops` included for models that need it). |
| **Vector store** | **Chroma** with optional on-disk persistence under a per-build directory. |
| **Retrieval** | Optional **multi-query** expansion and **HyDE** (hypothetical document embeddings), with deduplication and optional **parent lifting**. |
| **Generation** | **Ollama** chat or **Hugging Face** local chat pipeline; defaults favor small instruct models (e.g. TinyLlama ~1.1B for local chat). |
| **UI** | Landing page, dedicated **Ollama setup** help (documentation only—no installs from the app), and a workspace for indexing and Q&A. |

Pipeline logic is **not** embedded in Streamlit handlers beyond wiring; import the `rag` package from scripts, notebooks, or tests the same way the app does.

---

## Requirements

- **Python** `>= 3.14` (see [`pyproject.toml`](pyproject.toml)).
- **Git** and a **virtual environment** tool. This repo is configured for **[uv](https://github.com/astral-sh/uv)** (`uv sync` installs locked dependencies).
- Optional but typical for this stack:
  - **[Ollama](https://ollama.com)** for local embedding and chat models (install and pull models on **your** machine—see in-app *Ollama setup* and [ollama.com/download](https://ollama.com/download)).
  - **GPU / RAM** sized to your chosen Hugging Face models if you use the HF backends locally.

---

## Reproducing the project from GitHub

These steps assume you are starting from a clean clone of this repository (for example after pushing it to GitHub and opening it elsewhere).

### 1. Clone the repository

```bash
git clone https://github.com/<your-org-or-user>/<your-repo-name>.git
cd <your-repo-name>
```

Replace the URL with your actual GitHub remote.

### 2. Install dependencies with uv (recommended)

[Install uv](https://docs.astral.sh/uv/getting-started/installation/) if you do not have it, then from the repository root:

```bash
uv sync
```

This creates or updates a `.venv` and installs runtime dependencies from `pyproject.toml`. Committing **`uv.lock`** (recommended) lets `uv sync` install the same resolved dependency versions on any clone. To include development dependencies (pytest):

```bash
uv sync --group dev
```

Activate the environment, or prefix commands with `uv run` (examples below use `uv run` so activation is optional).

### 3. Alternative: pip / venv

If you prefer not to use uv:

```bash
python3.14 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install .
pip install pytest          # optional, for tests
```

Adjust the Python executable if your `3.14` binary is named differently on your OS.

### 4. Verify the install

```bash
uv run python -c "import rag; import streamlit; print('ok')"
```

---

## Running the application

From the repository root:

```bash
uv run streamlit run app.py
```

Then open the URL Streamlit prints (typically `http://localhost:8501`). The first screen is the **landing** page; open **Workspace** to upload documents, build the index, and ask questions.

A convenience message is also available via:

```bash
uv run python main.py
```

That only prints how to launch Streamlit; the app itself is `app.py`.

---

## Using the UI

1. **Landing** — Choose **Open workspace** or **Ollama setup** (instructional-only page with links to official Ollama docs).
2. **Workspace (sidebar)** — Choose embedding provider (Ollama vs Hugging Face), chat provider, chunking options, and retrieval options (multi-query, HyDE, hierarchical parent–child indexing). If Ollama is running, model tags may populate from `ollama list`; otherwise enter tags manually.
3. **Corpus ingestion** — Upload supported files (PDF, TXT, Markdown), then **Build or rebuild search index**. A new Chroma persist directory is created per successful build (path shown at the bottom when applicable).
4. **Questions** — Enter a query and **Generate answer**; expand **Evidence passages** to inspect grounded context.

Navigation uses **Home**, **Workspace**, and **Ollama setup** in the sidebar without discarding an existing in-session index unless you rebuild.

---

## Configuration overview

| Concern | Ollama | Hugging Face |
|---------|--------|----------------|
| **Embeddings** | Default embedding model name aligns with `nomic-embed-text` (Ollama). | Default Hub id for embeddings is set in `rag/embeddings/constants.py` (Nomic-compatible checkpoint). Some models require `trust_remote_code` (handled for `nomic-ai/*` ids). |
| **Chat** | Any chat-capable tag you have pulled (e.g. `llama3.2:latest`). | Default generation model id is a small instruct model (~1.1B parameters); override in the UI for other ≤2B-class checkpoints suitable for your hardware. |
| **Vector store** | Chroma persists under the temporary directory created at index time (displayed in the UI). | Same. |

The application does **not** download Ollama or pull models automatically; follow the in-app **Ollama setup** page or official Ollama documentation.

---

## Running tests

With the dev group installed:

```bash
uv sync --group dev
uv run pytest tests/
```

Pytest options live in [`pyproject.toml`](pyproject.toml) (`testpaths`, `pythonpath`, and filters for known third-party deprecation warnings).

---

## Project layout

```
.
├── app.py                 # Streamlit entry (landing, workspace, Ollama help)
├── main.py                # Prints launch hint for Streamlit
├── pyproject.toml         # Dependencies and tool config
├── streamlit_helpers/     # Small helpers (e.g. upload → temp paths)
├── rag/
│   ├── loading/          # Document loaders (PDF, text, directory, dispatch)
│   ├── chunking/          # Flat and parent/child splitting
│   ├── embeddings/        # Ollama + Hugging Face embedding factories
│   ├── vectorstore/       # Chroma build + parent index bundle
│   ├── retrieval/        # Multi-query, HyDE, parent lift, merge
│   ├── generation/       # Chat factories + RAG answer chain
│   └── pipeline/         # End-to-end ingest → index composition
└── tests/                 # Pytest suite mirroring rag modules
```

Import examples for programmatic use:

```python
from rag.pipeline import ingest_paths_to_index
from rag.retrieval import retrieve_bundle, RetrievalInputs
from rag.generation import generate_rag_answer, build_chat_model
```

---

## Environment variables

| Variable | Purpose |
|----------|---------|
| `OLLAMA_HOST` | When Ollama runs on another host/port, point clients to it per [Ollama networking docs](https://github.com/ollama/ollama/blob/main/docs/faq.md). |
| `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` | Required for gated Hugging Face models; set in your shell or `.env` tooling before starting Streamlit (not read automatically unless you add that integration). |

---

## Troubleshooting

- **`ImportError: ... einops`** — Some Hub embedding models declare `einops`; it is listed in `pyproject.toml`. Run `uv sync` after pulling latest `main`.
- **Ollama model list empty in the UI** — Ensure `ollama serve` (or the system service) is running and reachable; confirm with `ollama list` in your own terminal, then use **Ollama setup** in the app for install/pull instructions.
- **Large HF model downloads** — First run of a Hugging Face backend may download gigabytes of weights; ensure disk space and acceptable Hugging Face Hub terms.
- **Python version errors** — This project declares `requires-python >= 3.14`. Install a matching interpreter or adjust constraints only if you fully understand dependency compatibility.

---
