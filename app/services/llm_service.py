"""
LLM provider abstraction layer.

Defines LLMProvider as an abstract base class and provides three implementations:

    LocalProvider     -- development/CI mode; no external API call required
    OpenAIProvider    -- production OpenAI (gpt-4o, gpt-4o-mini, etc.)
    AnthropicProvider -- production Anthropic (claude-3-haiku, claude-3-5-sonnet, etc.)

Provider selection is controlled by settings.LLM_PROVIDER:
    "local"     -> LocalProvider
    "openai"    -> OpenAIProvider
    "anthropic" -> AnthropicProvider

LLMResponse carries a standardised output regardless of which provider
generated it. Business logic (QueryService) never imports provider-specific
classes -- it only imports LLMProvider and LLMResponse.

Dependency isolation:
    OpenAIProvider imports openai inside its generate() method.
    AnthropicProvider imports anthropic inside its generate() method.
    This means the platform starts and runs without those packages installed,
    as long as settings.LLM_PROVIDER is "local" or an untested provider is
    not instantiated. Missing packages produce ImportError at generate() time,
    not at application startup.

Token counting:
    All three providers report token usage. LocalProvider estimates based
    on character counts. OpenAI and Anthropic report exact counts from their
    API responses.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.logging.logger import get_logger
from app.models.exceptions import LLMProviderError, LLMTimeoutError
from app.models.query_result import TokenUsage
from app.rag.prompt_builder import Message, estimate_prompt_tokens

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# LLM response dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LLMResponse:
    """
    Standardised response from any LLM provider.

    Attributes:
        content:     The text content of the model's response.
        token_usage: Token consumption breakdown.
        model:       The exact model identifier used (from the API response).
        provider:    The provider name ("local", "openai", "anthropic").
        latency_ms:  Time from API call to response, in milliseconds.
    """

    content: str
    token_usage: TokenUsage
    model: str
    provider: str
    latency_ms: int


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class LLMProvider(ABC):
    """
    Abstract base class for LLM provider implementations.

    All concrete providers must implement generate(). Business logic
    depends on this interface only -- never on concrete implementations.

    Args:
        model_name:      Model identifier string (passed to the API).
        api_key:         Provider API key. May be empty for LocalProvider.
        timeout_seconds: Maximum seconds to wait for an API response.
    """

    def __init__(
        self,
        model_name: str,
        api_key: str,
        timeout_seconds: int,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    @abstractmethod
    def generate(self, messages: list[Message]) -> LLMResponse:
        """
        Generate a response from the provided message list.

        Args:
            messages: Role/content message list from prompt_builder.

        Returns:
            LLMResponse: Standardised response with content and token usage.

        Raises:
            LLMTimeoutError:    If the provider does not respond in time.
            LLMProviderError:   If the provider returns an error response.
        """
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the canonical provider name string."""
        ...


# ---------------------------------------------------------------------------
# Local provider -- development and CI use
# ---------------------------------------------------------------------------


class LocalProvider(LLMProvider):
    """
    Development LLM provider that requires no external API call.

    Generates a deterministic response by summarising the context provided
    in the message list. This exercises the full query pipeline (embedding,
    retrieval, context assembly, prompt construction) without an API key.

    Use cases:
        - Development without API credentials
        - CI/CD pipeline test execution
        - Integration tests that verify the pipeline end-to-end

    Token reporting:
        Estimates based on character counts using the 1 token ≈ 4 chars rule.
        Reports are approximate but structurally correct.
    """

    @property
    def provider_name(self) -> str:
        return "local"

    def generate(self, messages: list[Message]) -> LLMResponse:
        """
        Generate a response by extracting key content from the context.

        Parses the user message to find the context block and query,
        then produces a structured response indicating context was found.
        """
        start = time.monotonic()

        user_message = next(
            (m["content"] for m in messages if m.get("role") == "user"), ""
        )

        # Extract context section
        context_section = ""
        if "<CONTEXT>" in user_message and "</CONTEXT>" in user_message:
            start_idx = user_message.index("<CONTEXT>") + len("<CONTEXT>")
            end_idx = user_message.index("</CONTEXT>")
            context_section = user_message[start_idx:end_idx].strip()

        # Extract query
        query = ""
        if "</CONTEXT>" in user_message:
            after_context = user_message.split("</CONTEXT>", 1)[-1].strip()
            query = after_context.strip()

        # Build a minimal, honest response
        if context_section:
            # Count sources mentioned
            source_lines = [
                line for line in context_section.splitlines()
                if line.startswith("[Source:")
            ]
            source_count = len(source_lines)
            response_text = (
                f"[LocalProvider] Based on {source_count} context passage(s) "
                f"retrieved for your query: '{query[:80]}{'...' if len(query) > 80 else ''}'. "
                "Context was successfully assembled and passed to this provider. "
                "In production, replace LocalProvider with OpenAIProvider or "
                "AnthropicProvider to receive a real AI-generated answer."
            )
        else:
            response_text = (
                "[LocalProvider] No context was provided. "
                "This indicates a pipeline issue -- context should always be "
                "present when the LLM is called."
            )

        latency_ms = int((time.monotonic() - start) * 1000)
        prompt_tokens = estimate_prompt_tokens(messages)
        completion_tokens = len(response_text) // 4

        logger.info(
            "llm response generated",
            extra={
                "event": "LLM_RESPONSE",
                "provider": self.provider_name,
                "model": self.model_name,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "latency_ms": latency_ms,
            },
        )

        return LLMResponse(
            content=response_text,
            token_usage=TokenUsage.from_counts(prompt_tokens, completion_tokens),
            model=self.model_name,
            provider=self.provider_name,
            latency_ms=latency_ms,
        )


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------


class OpenAIProvider(LLMProvider):
    """
    OpenAI LLM provider.

    Uses the openai Python SDK. The SDK is imported lazily inside generate()
    so that the application starts without the package installed (as long as
    this provider is not actually called).

    Compatible models: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo.

    Raises:
        LLMTimeoutError:  If the API does not respond within timeout_seconds.
        LLMProviderError: If the OpenAI API returns any error response,
                          or if the openai package is not installed.
    """

    @property
    def provider_name(self) -> str:
        return "openai"

    def generate(self, messages: list[Message]) -> LLMResponse:
        start = time.monotonic()

        try:
            import openai  # type: ignore[import-untyped]
        except ImportError as exc:
            raise LLMProviderError(
                "The openai package is not installed. "
                "Add openai to requirements.txt and run pip install."
            ) from exc

        try:
            client = openai.OpenAI(
                api_key=self.api_key,
                timeout=float(self.timeout_seconds),
            )
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,  # type: ignore[arg-type]
            )
        except openai.APITimeoutError as exc:
            raise LLMTimeoutError(
                f"OpenAI API did not respond within {self.timeout_seconds}s: {exc}"
            ) from exc
        except openai.APIError as exc:
            raise LLMProviderError(
                f"OpenAI API error: {exc}"
            ) from exc
        except Exception as exc:
            raise LLMProviderError(
                f"Unexpected error calling OpenAI API: {exc}"
            ) from exc

        latency_ms = int((time.monotonic() - start) * 1000)
        content = response.choices[0].message.content or ""

        usage = response.usage
        token_usage = TokenUsage.from_counts(
            prompt=usage.prompt_tokens if usage else 0,
            completion=usage.completion_tokens if usage else 0,
        )

        logger.info(
            "llm response generated",
            extra={
                "event": "LLM_RESPONSE",
                "provider": self.provider_name,
                "model": response.model,
                "prompt_tokens": token_usage.prompt_tokens,
                "completion_tokens": token_usage.completion_tokens,
                "latency_ms": latency_ms,
            },
        )

        return LLMResponse(
            content=content,
            token_usage=token_usage,
            model=response.model,
            provider=self.provider_name,
            latency_ms=latency_ms,
        )


# ---------------------------------------------------------------------------
# Anthropic provider
# ---------------------------------------------------------------------------


class AnthropicProvider(LLMProvider):
    """
    Anthropic LLM provider.

    Uses the anthropic Python SDK. The SDK is imported lazily inside generate()
    so that the application starts without the package installed.

    Compatible models: claude-3-5-sonnet-20241022, claude-3-haiku-20240307,
                       claude-3-opus-20240229.

    Note: Anthropic's messages API uses a slightly different structure --
    the system prompt is a separate parameter, not a message. This provider
    handles that translation internally.

    Raises:
        LLMTimeoutError:  If the API does not respond within timeout_seconds.
        LLMProviderError: If the Anthropic API returns any error response,
                          or if the anthropic package is not installed.
    """

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def generate(self, messages: list[Message]) -> LLMResponse:
        start = time.monotonic()

        try:
            import anthropic  # type: ignore[import-untyped]
        except ImportError as exc:
            raise LLMProviderError(
                "The anthropic package is not installed. "
                "Add anthropic to requirements.txt and run pip install."
            ) from exc

        # Extract system message and user messages separately
        system_content = ""
        user_messages: list[Message] = []
        for msg in messages:
            if msg.get("role") == "system":
                system_content = msg["content"]
            else:
                user_messages.append(msg)

        try:
            client = anthropic.Anthropic(
                api_key=self.api_key,
                timeout=float(self.timeout_seconds),
            )
            response = client.messages.create(
                model=self.model_name,
                max_tokens=1024,
                system=system_content,
                messages=user_messages,  # type: ignore[arg-type]
            )
        except anthropic.APITimeoutError as exc:
            raise LLMTimeoutError(
                f"Anthropic API did not respond within {self.timeout_seconds}s: {exc}"
            ) from exc
        except anthropic.APIError as exc:
            raise LLMProviderError(
                f"Anthropic API error: {exc}"
            ) from exc
        except Exception as exc:
            raise LLMProviderError(
                f"Unexpected error calling Anthropic API: {exc}"
            ) from exc

        latency_ms = int((time.monotonic() - start) * 1000)

        content_blocks = response.content
        content = "".join(
            block.text for block in content_blocks
            if hasattr(block, "text")
        )

        token_usage = TokenUsage.from_counts(
            prompt=response.usage.input_tokens,
            completion=response.usage.output_tokens,
        )

        logger.info(
            "llm response generated",
            extra={
                "event": "LLM_RESPONSE",
                "provider": self.provider_name,
                "model": response.model,
                "prompt_tokens": token_usage.prompt_tokens,
                "completion_tokens": token_usage.completion_tokens,
                "latency_ms": latency_ms,
            },
        )

        return LLMResponse(
            content=content,
            token_usage=token_usage,
            model=response.model,
            provider=self.provider_name,
            latency_ms=latency_ms,
        )


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------


def create_llm_provider(
    provider_name: str,
    model_name: str,
    api_key: str,
    timeout_seconds: int,
) -> LLMProvider:
    """
    Factory function for LLM providers.

    Instantiates the correct LLMProvider subclass based on provider_name.
    Called from QueryService.__init__() using values from Settings.

    Args:
        provider_name:   "local", "openai", or "anthropic".
        model_name:      Model identifier string.
        api_key:         API key (from settings.LLM_API_KEY.get_secret_value()).
        timeout_seconds: Request timeout (from settings.LLM_TIMEOUT_SECONDS).

    Returns:
        LLMProvider: Concrete provider instance.

    Raises:
        LLMProviderError: If provider_name is not recognised.
    """
    providers: dict[str, type[LLMProvider]] = {
        "local": LocalProvider,
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
    }

    if provider_name not in providers:
        raise LLMProviderError(
            f"Unknown LLM provider '{provider_name}'. "
            f"Supported providers: {sorted(providers.keys())}. "
            "Check the LLM_PROVIDER setting."
        )

    provider_class = providers[provider_name]

    logger.info(
        "llm provider created",
        extra={
            "event": "LLM_PROVIDER_CREATED",
            "provider": provider_name,
            "model": model_name,
        },
    )

    return provider_class(
        model_name=model_name,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
    )
