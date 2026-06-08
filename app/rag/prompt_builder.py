"""
Prompt builder for the RAG pipeline.

Constructs the message list passed to the LLM provider. The message list
follows the OpenAI chat format (role/content pairs), which is also supported
by Anthropic's messages API.

The system prompt is the primary defence against hallucination. It explicitly
instructs the model to:
    - Answer ONLY from the provided context
    - Never use general knowledge
    - Acknowledge when the context does not contain the answer

The system prompt is a module-level constant. It is not configurable via
environment variables -- operators weakening or removing it would silently
degrade answer quality and accuracy without any visible error.

Message structure:
    [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": "<CONTEXT>\n...\n</CONTEXT>\n\n{query}"},
    ]

The context is wrapped in XML-style tags to give the LLM a clear visual
boundary between retrieved document content and the user's question.
This boundary has been shown to improve grounding accuracy in empirical
evaluations across multiple model families.
"""

from app.logging.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# System prompt -- DO NOT make configurable without adding validation.
# Weakening this prompt allows the LLM to use general knowledge instead of
# the user's documents, producing plausible-sounding but potentially wrong
# answers.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a precise document assistant. "
    "Answer the user's question using ONLY the information provided in the "
    "<CONTEXT> section below. "
    "Do not use any knowledge from your training data. "
    "If the context does not contain sufficient information to answer the "
    "question, respond with: "
    "'The provided documents do not contain enough information to answer this question.' "
    "When answering, cite the source document and chunk number where the "
    "information was found, using the format [Source: filename, chunk N]. "
    "Be concise and accurate. Do not speculate or extrapolate beyond what "
    "the context explicitly states."
)

# Template for the user message. The context is wrapped in XML-style tags.
USER_MESSAGE_TEMPLATE = "<CONTEXT>\n{context}\n</CONTEXT>\n\n{query}"

# Type alias for a chat message dict
Message = dict[str, str]


def build_messages(context_text: str, query: str) -> list[Message]:
    """
    Build the message list for the LLM provider.

    Produces a two-message list: system prompt + user message containing
    the retrieved context and the query.

    Args:
        context_text: The assembled context string from context_assembler.
                      Each chunk is labelled with its source and index.
        query:        The user's original natural language query.

    Returns:
        list[Message]: A list of role/content dicts ready for the LLM API.
                       Format: [{"role": "system", "content": ...},
                                {"role": "user",   "content": ...}]
    """
    user_content = USER_MESSAGE_TEMPLATE.format(
        context=context_text,
        query=query,
    )

    messages: list[Message] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    logger.debug(
        "prompt built",
        extra={
            "system_prompt_chars": len(SYSTEM_PROMPT),
            "context_chars": len(context_text),
            "query_chars": len(query),
            "user_message_chars": len(user_content),
            "total_prompt_chars": len(SYSTEM_PROMPT) + len(user_content),
        },
    )

    return messages


def estimate_prompt_tokens(messages: list[Message]) -> int:
    """
    Estimate total prompt tokens from the message list.

    Uses the character approximation: tokens ≈ chars / 4.
    This is a rough estimate. For production-accurate counting,
    replace with tiktoken when it is added as a dependency.

    Args:
        messages: The message list from build_messages().

    Returns:
        int: Estimated token count for the full prompt.
    """
    total_chars = sum(len(m.get("content", "")) for m in messages)
    return total_chars // 4
