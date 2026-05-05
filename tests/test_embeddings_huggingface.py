from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rag.embeddings import DEFAULT_HF_EMBED_MODEL, create_huggingface_embeddings

_PATCH_ST = "sentence_transformers.SentenceTransformer"


@pytest.fixture
def mock_sentence_transformer() -> MagicMock:
    mock_client = MagicMock()

    def _encode(texts, **kw):
        del kw  # pooled args from LangChain caller
        return np.zeros((len(texts), 16), dtype=np.float32)

    mock_client.encode = _encode

    with patch(_PATCH_ST) as ctor:
        ctor.return_value = mock_client
        yield ctor


def test_hf_default_model_and_trust_remote_code_for_nomic(
    mock_sentence_transformer: MagicMock,
) -> None:
    emb = create_huggingface_embeddings()
    assert emb.model_name == DEFAULT_HF_EMBED_MODEL

    mock_sentence_transformer.assert_called_once()
    _name, kw = mock_sentence_transformer.call_args[0][0], mock_sentence_transformer.call_args.kwargs
    assert _name == DEFAULT_HF_EMBED_MODEL
    assert kw.get("trust_remote_code") is True


def test_hf_no_implicit_trust_remote_code_outside_nomic_family(
    mock_sentence_transformer: MagicMock,
) -> None:
    small = "sentence-transformers/all-MiniLM-L6-v2"
    emb = create_huggingface_embeddings(model_name=small)
    assert emb.model_name == small

    assert "trust_remote_code" not in mock_sentence_transformer.call_args.kwargs


def test_hf_embed_documents_uses_stub_encode(
    mock_sentence_transformer: MagicMock,
) -> None:
    emb = create_huggingface_embeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectors = emb.embed_documents(["hello", "world"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 16
