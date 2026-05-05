from __future__ import annotations

from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnablePassthrough

from rag.generation.format_context import format_documents_for_prompt
from rag.generation.prompts import RAG_GENERATION_PROMPT


def build_rag_answer_chain(
    llm: BaseChatModel,
    *,
    max_chars_per_context_doc: int | None = None,
) -> Runnable:
    """LCEL chain: ``{question, context_documents}`` → assistant string."""

    return (
        RunnablePassthrough.assign(
            context=lambda x: format_documents_for_prompt(
                x["context_documents"],
                max_chars_per_doc=max_chars_per_context_doc,
            ),
        )
        | RAG_GENERATION_PROMPT
        | llm
        | StrOutputParser()
    )


def generate_rag_answer(
    question: str,
    context_documents: Sequence[Document],
    llm: BaseChatModel,
    *,
    max_chars_per_context_doc: int | None = None,
) -> str:
    """Run grounded generation over retrieved ``context_documents``."""

    chain = build_rag_answer_chain(
        llm,
        max_chars_per_context_doc=max_chars_per_context_doc,
    )
    return str(
        chain.invoke(
            {
                "question": question,
                "context_documents": list(context_documents),
            },
        ),
    ).strip()
