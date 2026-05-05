from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_ollama import ChatOllama

from rag.generation import (
    DEFAULT_HF_GENERATION_MODEL_ID,
    build_chat_model,
)


def test_build_chat_model_ollama_returns_chat_ollama() -> None:
    chat = build_chat_model("ollama", ollama_options={"model": "test-model:latest"})
    assert isinstance(chat, ChatOllama)
    assert chat.model == "test-model:latest"


def test_build_chat_model_ollama_rejects_hf_sidecar() -> None:
    with pytest.raises(ValueError, match="huggingface_options"):
        build_chat_model(
            "ollama",
            huggingface_options={"model_id": "x"},
            ollama_options={"model": "m"},
        )


def test_build_chat_model_huggingface_rejects_ollama_sidecar() -> None:
    with pytest.raises(ValueError, match="ollama_options"):
        build_chat_model(
            "huggingface",
            ollama_options={"model": "x"},
            huggingface_options={},
        )


def test_build_chat_model_huggingface_routes_options(monkeypatch) -> None:
    import rag.generation.chat_factory as cf_pkg

    created: dict = {}

    def _stub(**kw) -> MagicMock:
        created.update(kw)
        return MagicMock()

    monkeypatch.setattr(cf_pkg, "create_chat_huggingface_local", _stub)

    build_chat_model(
        "huggingface",
        huggingface_options={
            "model_id": DEFAULT_HF_GENERATION_MODEL_ID,
            "pipeline_kwargs": {"max_new_tokens": 12},
        },
    )

    assert created["model_id"] == DEFAULT_HF_GENERATION_MODEL_ID
    assert created["pipeline_kwargs"]["max_new_tokens"] == 12

