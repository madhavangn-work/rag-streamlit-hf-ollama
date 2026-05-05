from langchain_core.prompts import ChatPromptTemplate

MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You diversify user questions for text retrieval.\n"""

            """Produce {n_queries} short retrievals-style paraphrases of the SAME intent.\n"""
            """Separate lines must start like:\n"""

            """1.\n"""

            """2.\n"""

            """No introductions or numbering beyond that pattern.""",
        ),
        ("human", "{question}"),
    ],
)

HYDE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You write hypothetical reference paragraphs that COULD appear in textbooks \n"""

            """or docs and implicitly answer factual questions.""",
        ),
        (
            "human",
            "Question:\n"
            "{question}\n\n"
            "Hypothetical reference paragraph:",
        ),
    ],
)
