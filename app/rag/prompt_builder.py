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
from app.rag.token_utils import estimate_messages_tokens

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# System prompt -- DO NOT make configurable without adding validation.
# Weakening this prompt allows the LLM to use general knowledge instead of
# the user's documents, producing plausible-sounding but potentially wrong
# answers.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an expert resume and document analysis assistant. "

    "Answer ONLY using information present in the provided context. "

    "Use ALL relevant chunks when generating an answer. "

    "If the query asks for a summary, candidate profile, overview, "
    "or resume summary, combine information across multiple chunks and include: "
    "education, skills, projects, technologies, achievements, and experience "
    "whenever available in the context. "

    "Do not focus on a single chunk if multiple chunks contain relevant information. "

    "If information is missing from the context, explicitly state that it is not available. "

    "Always provide a structured and concise answer. "

    "Always cite sources using the format "
    "[Source: filename, chunk N]."
)

# System prompt for DIRECT routing -- general knowledge answers.
# Deliberately different from SYSTEM_PROMPT: no context block is provided,
# and the model is permitted to use its training knowledge.
# Applied when the Router Agent decides DIRECT (no documents, or strong
# general-knowledge signal).
DIRECT_SYSTEM_PROMPT = (
    "You are a knowledgeable and helpful assistant. "
    "Answer the user's question accurately and concisely using your general knowledge. "
    "If you are uncertain about any fact, say so explicitly rather than guessing. "
    "Do not fabricate information."
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


def build_direct_messages(query: str) -> list[Message]:
    """
    Build the message list for a DIRECT (no-retrieval) LLM call.

    Used by RoutedQueryService when the Router Agent decides DIRECT.
    No context block is included -- the model answers from general knowledge.

    The system prompt is DIRECT_SYSTEM_PROMPT, which is deliberately
    different from SYSTEM_PROMPT. SYSTEM_PROMPT forbids general knowledge;
    DIRECT_SYSTEM_PROMPT permits it.

    Args:
        query: The user's original natural language query.

    Returns:
        list[Message]: Two messages: [system (direct prompt), user (query)].
                       No <CONTEXT> tags -- the user message is just the query.
    """
    messages: list[Message] = [
        {"role": "system", "content": DIRECT_SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    logger.debug(
        "direct prompt built",
        extra={
            "system_prompt_chars": len(DIRECT_SYSTEM_PROMPT),
            "query_chars": len(query),
            "total_prompt_chars": len(DIRECT_SYSTEM_PROMPT) + len(query),
        },
    )

    return messages


def estimate_prompt_tokens(messages: list[Message]) -> int:
    """
    Estimate total prompt tokens from the message list.

    Delegates to app.rag.token_utils.estimate_messages_tokens().
    To switch to exact tokenisation, update token_utils only.

    Args:
        messages: The message list from build_messages().

    Returns:
        int: Estimated token count for the full prompt.
    """
    return estimate_messages_tokens(messages)
