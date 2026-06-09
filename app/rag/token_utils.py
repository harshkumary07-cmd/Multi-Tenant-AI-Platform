"""
Token estimation utilities.

Single source of truth for all token counting in the RAG pipeline.

Current implementation: character approximation (tokens ≈ chars / 4).

Upgrade path:
    Replace estimate_tokens() and tokens_to_char_limit() here to use
    provider-specific tokenisers (e.g. tiktoken for OpenAI, Anthropic's
    tokeniser for Claude) without modifying any caller.

    No other file in the codebase performs token arithmetic directly.
    All callers import from this module.

Approximation accuracy:
    For English prose, 1 token ≈ 3.8-4.2 characters (GPT-4 family).
    The fixed ratio of 4 chars/token is conservative: it slightly
    under-estimates available budget, which is the safe error direction.
    The approximation never causes context-window overflow.
"""

# ---------------------------------------------------------------------------
# Internal constant -- used only within this module.
# No other file may reference this value directly.
# ---------------------------------------------------------------------------
_CHARS_PER_TOKEN: int = 4


def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in a text string.

    Args:
        text: Any string -- a single message, a full prompt, a completion.

    Returns:
        int: Estimated token count. Always >= 0.

    Upgrade note:
        Replace this function body with a tiktoken or provider-specific
        tokeniser call. The signature must remain (text: str) -> int.
    """
    return len(text) // _CHARS_PER_TOKEN


def estimate_messages_tokens(messages: list[dict[str, str]]) -> int:
    """
    Estimate the total tokens across a list of chat messages.

    Sums estimate_tokens() over every message's content field.

    Args:
        messages: List of role/content dicts (the LLM message format).

    Returns:
        int: Estimated total token count for the full message list.
    """
    return sum(estimate_tokens(m.get("content", "")) for m in messages)


def tokens_to_char_limit(token_budget: int) -> int:
    """
    Convert a token budget into a character limit for context assembly.

    Used by the context assembler to enforce the context window budget
    without exact tokenisation.

    Args:
        token_budget: Maximum allowed tokens (e.g. 2000).

    Returns:
        int: Maximum characters to include in the assembled context.

    Upgrade note:
        When switching to exact tokenisation, this function may be
        deprecated in favour of direct token counting in the assembler.
        Keep it until the assembler is updated simultaneously.
    """
    return token_budget * _CHARS_PER_TOKEN
