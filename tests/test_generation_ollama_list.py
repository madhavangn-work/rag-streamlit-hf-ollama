from __future__ import annotations

from types import SimpleNamespace
import pytest

from rag.generation import OllamaDiscoveryError, list_installed_ollama_models


def test_list_installed_ollama_models_sorted_unique(monkeypatch) -> None:
    import rag.generation.ollama_models as pkg

    class _FakeClient:
        def list(self):
            return SimpleNamespace(
                models=[
                    SimpleNamespace(model="zephyr:latest"),
                    SimpleNamespace(model="llama:7b"),
                    SimpleNamespace(model="llama:7b"),
                    SimpleNamespace(model=None),
                ],
            )

    monkeypatch.setattr(pkg, "Client", lambda host=None: _FakeClient())

    names = list_installed_ollama_models()
    assert names == ["llama:7b", "zephyr:latest"]


def test_list_installed_maps_connection_error(monkeypatch) -> None:
    import rag.generation.ollama_models as pkg

    class _BadClient:
        def list(self):
            raise ConnectionError("boom")

    monkeypatch.setattr(pkg, "Client", lambda host=None: _BadClient())

    with pytest.raises(OllamaDiscoveryError, match="Could not reach Ollama"):
        list_installed_ollama_models()
