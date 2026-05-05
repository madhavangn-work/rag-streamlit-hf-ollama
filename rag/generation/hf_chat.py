from __future__ import annotations

from typing import Any

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

from rag.generation.constants import DEFAULT_HF_GENERATION_MODEL_ID


def create_chat_huggingface_local(
    *,
    model_id: str | None = None,
    task: str = "text-generation",
    pipeline_kwargs: dict[str, Any] | None = None,
    model_kwargs: dict[str, Any] | None = None,
    device_map: str | None = None,
    device: int | None = None,
    chat_kwargs: dict[str, Any] | None = None,
) -> ChatHuggingFace:
    """Local HF causal-LM stack (transformers pipeline) wrapped as a chat model.

    Default ``model_id`` is ≈1.1B parameters (under the project 2B cap). Choose a
    different chat checkpoint explicitly if you need another small model.
    """

    resolved = model_id if model_id is not None else DEFAULT_HF_GENERATION_MODEL_ID
    pipe_kw = {"max_new_tokens": 256, **(pipeline_kwargs or {})}

    llm = HuggingFacePipeline.from_model_id(
        model_id=resolved,
        task=task,
        model_kwargs=dict(model_kwargs or {}),
        pipeline_kwargs=pipe_kw,
        device_map=device_map,
        device=device,
    )

    return ChatHuggingFace(llm=llm, **(chat_kwargs or {}))
