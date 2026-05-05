"""Answer generation: Ollama or Hugging Face chat models + RAG prompt chain."""

from rag.generation.chat_factory import build_chat_model
from rag.generation.constants import DEFAULT_HF_GENERATION_MODEL_ID
from rag.generation.exceptions import OllamaDiscoveryError
from rag.generation.format_context import format_documents_for_prompt
from rag.generation.generate import build_rag_answer_chain, generate_rag_answer
from rag.generation.hf_chat import create_chat_huggingface_local
from rag.generation.ollama_chat import create_chat_ollama
from rag.generation.ollama_models import list_installed_ollama_models
from rag.generation.prompts import RAG_GENERATION_PROMPT
from rag.retrieval.adapters import adapt_prompt_with_runnable

__all__ = [
    "adapt_prompt_with_runnable",
    "DEFAULT_HF_GENERATION_MODEL_ID",
    "OllamaDiscoveryError",
    "RAG_GENERATION_PROMPT",
    "build_chat_model",
    "build_rag_answer_chain",
    "create_chat_huggingface_local",
    "create_chat_ollama",
    "format_documents_for_prompt",
    "generate_rag_answer",
    "list_installed_ollama_models",
]
