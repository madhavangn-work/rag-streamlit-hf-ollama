from langchain_core.prompts import ChatPromptTemplate

RAG_GENERATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer using ONLY the passages in Context.\n"

            "- If Context is insufficient, say you do not have enough evidence.\n"

            "- Prefer concise, factual answers; mention uncertainty when appropriate.\n"

            "- Do not invent sources or quotation marks for passages not present.",
        ),
        (
            "human",
            "Context:\n"
            "{context}\n\n"
            "Question:\n"
            "{question}",
        ),
    ],
)
