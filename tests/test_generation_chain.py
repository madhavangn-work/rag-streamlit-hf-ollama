from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from rag.generation import generate_rag_answer


def test_generate_rag_answer_uses_context_fake_llm() -> None:
    llm = FakeListChatModel(responses=["Answer: TOK_ABC_MARKER context mentioned."])

    docs = [
        Document(
            page_content="Rare token TOK_ABC_MARKER lives only here.",
            metadata={"source": "doc.md"},
        ),
    ]
    ans = generate_rag_answer("Where is TOK_ABC_MARKER?", docs, llm)
    assert "TOK_ABC_MARKER" in ans
