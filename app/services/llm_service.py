"""
LLM provider abstraction layer.

Defines LLMProvider as an abstract base class and provides four implementations:

    LocalProvider     -- development/CI mode; no external API call required
    OpenAIProvider    -- production OpenAI (gpt-4o, gpt-4o-mini, etc.)
    AnthropicProvider -- production Anthropic (claude-3-haiku, claude-3-5-sonnet, etc.)
    OllamaProvider    -- local Ollama instance (llama3, mistral, etc.)

Provider selection is controlled by settings.LLM_PROVIDER:
    "local"     -> LocalProvider
    "openai"    -> OpenAIProvider
    "anthropic" -> AnthropicProvider
    "ollama"    -> OllamaProvider

LLMResponse carries a standardised output regardless of which provider
generated it. Business logic (QueryService) never imports provider-specific
classes -- it only imports LLMProvider and LLMResponse.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import httpx

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
        provider:    The provider name ("local", "openai", "anthropic", "ollama").
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
    """

    @property
    def provider_name(self) -> str:
        return "local"

    def generate(self, messages: list[Message]) -> LLMResponse:
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

        if context_section:
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
            raise LLMProviderError(f"OpenAI API error: {exc}") from exc
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
            raise LLMProviderError(f"Anthropic API error: {exc}") from exc
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
# Ollama provider
# ---------------------------------------------------------------------------


class OllamaProvider(LLMProvider):
    """
    Ollama local LLM provider.

    Calls a locally-running Ollama instance via its REST API using the
    /api/chat endpoint (chat-style messages format).

    The Ollama host URL is configurable via the OLLAMA_BASE_URL constructor
    argument (mapped from settings.OLLAMA_BASE_URL).

    Compatible models: any model pulled via `ollama pull <model>`,
    e.g. mistral, llama3, codellama, etc.

    Raises:
        LLMTimeoutError:  If Ollama does not respond within timeout_seconds.
        LLMProviderError: If Ollama returns an error or is unreachable.
    """

    def __init__(
        self,
        model_name: str,
        api_key: str,
        timeout_seconds: int,
        base_url: str = "http://localhost:11434",
    ) -> None:
        super().__init__(model_name, api_key, timeout_seconds)
        # Normalise: strip trailing slash
        self._base_url = base_url.rstrip("/")

    @property
    def provider_name(self) -> str:
        return "ollama"

    def generate(self, messages: list[Message]) -> LLMResponse:
        start = time.monotonic()

        # Use /api/chat for proper chat-style message handling
        url = f"{self._base_url}/api/chat"

        try:
            response = httpx.post(
                url,
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "stream": False,
                },
                timeout=float(self.timeout_seconds),
            )
            response.raise_for_status()
            data = response.json()

        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(
                f"Ollama did not respond within {self.timeout_seconds}s. "
                f"Check that Ollama is running at {self._base_url}."
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise LLMProviderError(
                f"Ollama returned HTTP {exc.response.status_code}: "
                f"{exc.response.text[:200]}"
            ) from exc
        except Exception as exc:
            raise LLMProviderError(
                f"Ollama request failed: {exc}. "
                f"Ensure Ollama is reachable at {self._base_url} and "
                f"model '{self.model_name}' is pulled."
            ) from exc

        latency_ms = int((time.monotonic() - start) * 1000)

        # /api/chat response: {"message": {"role": "assistant", "content": "..."}}
        message_obj = data.get("message", {})
        content = message_obj.get("content", "")

        if not content:
            # Fallback: some Ollama versions use "response" key
            content = data.get("response", "")

        # Ollama reports token counts in eval_count / prompt_eval_count
        prompt_tokens = data.get("prompt_eval_count", estimate_prompt_tokens(messages))
        completion_tokens = data.get("eval_count", len(content) // 4)

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
            content=content,
            token_usage=TokenUsage.from_counts(prompt_tokens, completion_tokens),
            model=self.model_name,
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
    ollama_base_url: str = "http://localhost:11434",
) -> LLMProvider:
    """
    Factory function for LLM providers.

    Args:
        provider_name:    "local", "openai", "anthropic", or "ollama".
        model_name:       Model identifier string.
        api_key:          API key (from settings.LLM_API_KEY.get_secret_value()).
        timeout_seconds:  Request timeout (from settings.LLM_TIMEOUT_SECONDS).
        ollama_base_url:  Base URL for Ollama (from settings.OLLAMA_BASE_URL).

    Returns:
        LLMProvider: Concrete provider instance.

    Raises:
        LLMProviderError: If provider_name is not recognised.
    """
    if provider_name == "ollama":
        logger.info(
            "llm provider created",
            extra={
                "event": "LLM_PROVIDER_CREATED",
                "provider": provider_name,
                "model": model_name,
                "ollama_base_url": ollama_base_url,
            },
        )
        return OllamaProvider(
            model_name=model_name,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            base_url=ollama_base_url,
        )

    providers: dict[str, type[LLMProvider]] = {
        "local": LocalProvider,
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
    }

    if provider_name not in providers:
        raise LLMProviderError(
            f"Unknown LLM provider '{provider_name}'. "
            f"Supported providers: {sorted(list(providers.keys()) + ['ollama'])}. "
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
