from __future__ import annotations

from collections.abc import Mapping

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from rag.retrieval.types import LLMTurnPrompt


def adapt_prompt_with_runnable(
    prompt: ChatPromptTemplate,
    llm_tail: Runnable,
) -> LLMTurnPrompt:
    """Wire a formatted chat prompt plus any LangChain runnable (e.g. chat model)."""

    chain = prompt | llm_tail | StrOutputParser()

    def _invoke(variables: Mapping[str, str | int]) -> str:
        return str(chain.invoke(dict(variables))).strip()

    return _invoke
