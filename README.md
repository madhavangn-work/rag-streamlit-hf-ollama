# rag-streamlit-hf-ollama

A Streamlit-based RAG app using Hugging Face embeddings and local LLM inference with Ollama.

Repository: <https://github.com/madhavangn-work/rag-streamlit-hf-ollama>

## Highlights

- Streamlit UI for document upload, indexing, and Q&A.
- Modular `rag/` package (loading, chunking, embeddings, retrieval, generation, pipeline).
- Embeddings and generation via Ollama and/or Hugging Face local models.
- Chroma-backed vector storage.
- Retrieval enhancements including HyDE, multi-query expansion, and parent-document flows.

## Quick Start

### 1) Clone

```bash
git clone https://github.com/madhavangn-work/rag-streamlit-hf-ollama.git
cd rag-streamlit-hf-ollama
```

### 2) Install dependencies

Recommended (uv):

```bash
uv sync
```

Include dev tools:

```bash
uv sync --group dev
```

Alternative (pip/venv):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install .
pip install pytest
```

### 3) Run app

```bash
uv run streamlit run app.py
```

Open the local Streamlit URL (usually `http://localhost:8501`).

## Requirements

- Python `>= 3.14`
- Git
- Optional: [Ollama](https://ollama.com/download) for local embedding/chat models
- Optional: adequate CPU/GPU RAM for chosen Hugging Face local models

## Basic Usage

1. Open the workspace page in Streamlit.
2. Choose embeddings/chat backends (Ollama or Hugging Face).
3. Upload files (PDF/TXT/Markdown).
4. Build or rebuild the index.
5. Ask questions and inspect retrieved evidence.

## Run Tests

```bash
uv sync --group dev
uv run pytest tests/
```

## Project Structure

```text
.
├── app.py
├── main.py
├── pyproject.toml
├── rag/
│   ├── loading/
│   ├── chunking/
│   ├── embeddings/
│   ├── vectorstore/
│   ├── retrieval/
│   ├── generation/
│   └── pipeline/
└── tests/
```

## Environment Variables

- `OLLAMA_HOST`: set when Ollama runs on a non-default host/port.
- `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN`: required for gated Hugging Face models.

## Troubleshooting

- If Ollama models are not detected, ensure `ollama serve` is running.
- If Hugging Face model loading fails, verify token access for gated repos.
- If dependency errors appear, rerun `uv sync` (or reinstall in a fresh virtualenv).

## License

MIT
