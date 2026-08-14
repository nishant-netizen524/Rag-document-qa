"""Prompt templates for grounded, citation-backed Q&A."""

SYSTEM_PROMPT = """You are a precise document Q&A assistant. You answer questions using ONLY the document excerpts provided in the user message, which are labeled with their source file and page number.

Rules:
1. Base every answer strictly on the provided excerpts. Do not use outside knowledge.
2. If the excerpts do not contain the answer, reply exactly: "This information is not present in the uploaded document."
3. Cite sources after every claim using the format [file name, page N].
4. Quote short phrases from the document when they directly support your answer.
5. Be concise: answer the question directly, then support it with the cited excerpts.
6. If the question is ambiguous, say what is unclear and answer for the most likely interpretation."""


def build_user_prompt(query: str, context_blocks: list[dict]) -> str:
    parts = [
        f"[Excerpt {i}] Source: {b['filename']} (page {b['page']})\n{b['text']}"
        for i, b in enumerate(context_blocks, start=1)
    ]
    return (
        f"QUESTION: {query}\n\n"
        f"DOCUMENT EXCERPTS:\n\n{chr(10).join(parts)}\n\n"
        "Answer the QUESTION using only the DOCUMENT EXCERPTS above, following the system rules."
    )

def build_llm_messages(
    query: str,
    context_blocks: list[dict],
    history: list[dict],
    max_history_messages: int = 6,
) -> list[dict]:
    """
    Build messages for the LLM with recent conversation memory.

    history comes from st.session_state.messages.
    We include only recent user/assistant messages, not sources.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    useful_history = []

    for msg in history[-max_history_messages:]:
        role = msg.get("role")
        content = (msg.get("content") or "").strip()

        if role not in ("user", "assistant"):
            continue

        if not content:
            continue

        # Do not push obvious error/empty states into memory
        if content.startswith("Sorry, I could not generate an answer"):
            continue

        if content in {
            "No documents indexed yet. Upload a PDF in the sidebar first.",
            "No relevant content found in the indexed documents.",
        }:
            continue

        # Keep history token usage under control
        if len(content) > 1200:
            content = content[:1200] + "... [truncated]"

        useful_history.append(
            {
                "role": role,
                "content": content,
            }
        )

    messages.extend(useful_history)

    messages.append(
        {
            "role": "user",
            "content": build_user_prompt(query, context_blocks),
        }
    )

    return messages