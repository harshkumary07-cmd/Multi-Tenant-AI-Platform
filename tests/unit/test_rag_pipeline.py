"""
Unit tests for Module 6 -- RAG Query Engine.

Tests cover every component with mocked dependencies.
No infrastructure (ChromaDB, LLM APIs, embedding model) required.

Test classes:
    TestTokenUsage           -- TokenUsage dataclass
    TestSourceReference      -- SourceReference dataclass
    TestQueryResult          -- QueryResult properties
    TestContextAssembler     -- filtering, ranking, budget, deduplication
    TestPromptBuilder        -- message construction, token estimation
    TestLLMProviderFactory   -- provider creation and error handling
    TestLocalProvider        -- local provider response generation
    TestOpenAIProvider       -- OpenAI provider with mocked SDK
    TestAnthropicProvider    -- Anthropic provider with mocked SDK
    TestQueryService         -- full pipeline orchestration
    TestQueryServiceNoResult -- no-result path when threshold not met
"""

from unittest.mock import MagicMock, patch

import pytest

from app.models.chunk import ChunkResult
from app.models.exceptions import (
    LLMProviderError,
    LLMTimeoutError,
    NoRelevantChunksError,
    VectorStoreError,
)
from app.models.query_result import QueryResult, SourceReference, TokenUsage
from app.rag.context_assembler import (
    AssembledContext,
    assemble_context,
)
from app.rag.prompt_builder import (
    SYSTEM_PROMPT,
    build_messages,
    estimate_prompt_tokens,
)
from app.rag.token_utils import estimate_messages_tokens, estimate_tokens, tokens_to_char_limit
from app.services.llm_service import (
    AnthropicProvider,
    LLMProvider,
    LLMResponse,
    LocalProvider,
    OpenAIProvider,
    create_llm_provider,
)
from app.services.query_service import QueryService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_chunk_result(
    score: float = 0.85,
    doc_id: str = "doc_abc",
    source: str = "report.pdf",
    chunk_index: int = 0,
    text: str = "This is a relevant chunk of text with useful information.",
) -> ChunkResult:
    return ChunkResult(
        chunk_id=f"{doc_id}_chunk_{chunk_index:03d}",
        doc_id=doc_id,
        source=source,
        chunk_index=chunk_index,
        text=text,
        score=score,
    )


def make_mock_repository(
    chunks: list[ChunkResult] | None = None,
    doc_count: int = 1,
) -> MagicMock:
    repo = MagicMock()
    repo.search_chunks.return_value = chunks if chunks is not None else [make_chunk_result()]
    repo.count_documents.return_value = doc_count
    return repo


def make_mock_settings(
    top_k: int = 5,
    threshold: float = 0.35,
) -> MagicMock:
    s = MagicMock()
    s.RETRIEVAL_TOP_K = top_k
    s.RETRIEVAL_CONFIDENCE_THRESHOLD = threshold
    return s


def make_local_provider() -> LocalProvider:
    return LocalProvider(
        model_name="local",
        api_key="",
        timeout_seconds=30,
    )


# ---------------------------------------------------------------------------
# TokenUsage
# ---------------------------------------------------------------------------


class TestTokenUsage:

    def test_zero_returns_all_zeros(self) -> None:
        u = TokenUsage.zero()
        assert u.prompt_tokens == 0
        assert u.completion_tokens == 0
        assert u.total_tokens == 0

    def test_from_counts_sums_correctly(self) -> None:
        u = TokenUsage.from_counts(100, 50)
        assert u.prompt_tokens == 100
        assert u.completion_tokens == 50
        assert u.total_tokens == 150

    def test_is_frozen(self) -> None:
        import dataclasses
        u = TokenUsage.zero()
        with pytest.raises(dataclasses.FrozenInstanceError):
            u.prompt_tokens = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SourceReference
# ---------------------------------------------------------------------------


class TestSourceReference:

    def test_fields_accessible(self) -> None:
        ref = SourceReference(
            doc_id="doc_abc",
            source="report.pdf",
            chunk_count=3,
            top_score=0.91,
        )
        assert ref.doc_id == "doc_abc"
        assert ref.source == "report.pdf"
        assert ref.chunk_count == 3
        assert ref.top_score == 0.91

    def test_is_frozen(self) -> None:
        import dataclasses
        ref = SourceReference("a", "b.pdf", 1, 0.5)
        with pytest.raises(dataclasses.FrozenInstanceError):
            ref.doc_id = "modified"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# QueryResult
# ---------------------------------------------------------------------------


class TestQueryResult:

    def _make(
        self,
        answer: str | None = "The revenue was $2.4B.",
        no_result_reason: str | None = None,
    ) -> QueryResult:
        return QueryResult(
            query="What was the revenue?",
            user_id="u_test",
            answer=answer,
            sources=[SourceReference("doc_a", "report.pdf", 2, 0.9)],
            route="RETRIEVE",
            chunks_retrieved=5,
            chunks_used=2,
            token_usage=TokenUsage.from_counts(200, 80),
            latency_ms=1240,
            no_result_reason=no_result_reason,
        )

    def test_has_answer_true_when_answer_present(self) -> None:
        result = self._make(answer="some answer")
        assert result.has_answer is True

    def test_has_answer_false_when_none(self) -> None:
        result = self._make(answer=None)
        assert result.has_answer is False

    def test_is_no_result_true_when_no_answer_and_reason(self) -> None:
        result = self._make(answer=None, no_result_reason="NO_RELEVANT_CHUNKS")
        assert result.is_no_result is True

    def test_is_no_result_false_when_answer_present(self) -> None:
        result = self._make(answer="answer here")
        assert result.is_no_result is False

    def test_is_frozen(self) -> None:
        import dataclasses
        result = self._make()
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.answer = "modified"  # type: ignore[misc]

    def test_timestamp_defaults_to_now(self) -> None:
        from datetime import UTC, datetime, timedelta

        before = datetime.now(tz=UTC)
        result = self._make()
        after = datetime.now(tz=UTC)
        assert before - timedelta(seconds=1) <= result.timestamp <= after


# ---------------------------------------------------------------------------
# Token utilities
# ---------------------------------------------------------------------------


class TestTokenUtils:
    """token_utils is the single source of truth for all token arithmetic."""

    def test_estimate_tokens_empty_string(self) -> None:
        """Empty string has zero tokens."""
        assert estimate_tokens("") == 0

    def test_estimate_tokens_returns_integer(self) -> None:
        """estimate_tokens always returns an int."""
        result = estimate_tokens("some text here")
        assert isinstance(result, int)

    def test_estimate_tokens_scales_with_length(self) -> None:
        """Longer text produces a higher estimate."""
        short = estimate_tokens("hi")
        long = estimate_tokens("A" * 400)
        assert long > short

    def test_estimate_tokens_non_negative(self) -> None:
        """Token estimate is always >= 0."""
        assert estimate_tokens("x") >= 0

    def test_estimate_tokens_four_chars_per_token(self) -> None:
        """Internal approximation: 4 chars = 1 token."""
        assert estimate_tokens("A" * 400) == 100
        assert estimate_tokens("A" * 4) == 1
        assert estimate_tokens("A" * 8) == 2

    def test_estimate_messages_tokens_empty_list(self) -> None:
        """Empty message list has zero tokens."""
        assert estimate_messages_tokens([]) == 0

    def test_estimate_messages_tokens_sums_content(self) -> None:
        """Sums estimate_tokens over all message contents."""
        messages = [
            {"role": "system", "content": "A" * 400},
            {"role": "user",   "content": "A" * 400},
        ]
        assert estimate_messages_tokens(messages) == 200

    def test_estimate_messages_tokens_ignores_missing_content(self) -> None:
        """Messages without a content key contribute zero tokens."""
        messages = [{"role": "system"}]
        assert estimate_messages_tokens(messages) == 0

    def test_tokens_to_char_limit_scales_linearly(self) -> None:
        """Character limit is proportional to the token budget."""
        assert tokens_to_char_limit(100) == tokens_to_char_limit(50) * 2

    def test_tokens_to_char_limit_zero(self) -> None:
        """Zero token budget produces zero character limit."""
        assert tokens_to_char_limit(0) == 0

    def test_tokens_to_char_limit_positive(self) -> None:
        """Positive token budget produces positive character limit."""
        assert tokens_to_char_limit(2000) > 0

    def test_prompt_builder_estimate_delegates_to_token_utils(self) -> None:
        """estimate_prompt_tokens in prompt_builder routes through token_utils."""
        messages = build_messages("context text here", "what is revenue?")
        # Both functions must agree
        assert estimate_prompt_tokens(messages) == estimate_messages_tokens(messages)


# ---------------------------------------------------------------------------
# Context assembler
# ---------------------------------------------------------------------------


class TestContextAssembler:

    def test_chunks_below_threshold_are_excluded(self) -> None:
        chunks = [
            make_chunk_result(score=0.9),
            make_chunk_result(score=0.1, chunk_index=1),  # below threshold
        ]
        result = assemble_context(chunks, threshold=0.35)
        assert result.chunk_count == 1
        assert all(c.score >= 0.35 for c in result.chunks_used)

    def test_all_chunks_above_threshold_are_included(self) -> None:
        chunks = [make_chunk_result(score=0.5 + i * 0.1, chunk_index=i) for i in range(3)]
        result = assemble_context(chunks, threshold=0.35)
        assert result.chunk_count == 3

    def test_no_chunks_above_threshold_raises(self) -> None:
        chunks = [make_chunk_result(score=0.1)]
        with pytest.raises(NoRelevantChunksError):
            assemble_context(chunks, threshold=0.35)

    def test_empty_chunks_raises(self) -> None:
        with pytest.raises(NoRelevantChunksError):
            assemble_context([], threshold=0.35)

    def test_chunks_sorted_by_score_descending(self) -> None:
        chunks = [
            make_chunk_result(score=0.5, chunk_index=0),
            make_chunk_result(score=0.9, chunk_index=1),
            make_chunk_result(score=0.7, chunk_index=2),
        ]
        result = assemble_context(chunks, threshold=0.35)
        scores = [c.score for c in result.chunks_used]
        assert scores == sorted(scores, reverse=True)

    def test_context_text_contains_source_headers(self) -> None:
        chunks = [make_chunk_result(score=0.9, source="annual.pdf", chunk_index=3)]
        result = assemble_context(chunks, threshold=0.35)
        assert "annual.pdf" in result.context_text
        assert "[Source:" in result.context_text

    def test_context_text_contains_chunk_text(self) -> None:
        chunks = [make_chunk_result(score=0.9, text="Revenue grew by 34 percent.")]
        result = assemble_context(chunks, threshold=0.35)
        assert "Revenue grew by 34 percent." in result.context_text

    def test_budget_limits_chunks_included(self) -> None:
        # Create chunks where each is ~500 chars; token_budget=100 -> char limit=400
        long_text = "A" * 450
        chunks = [make_chunk_result(score=0.9 - i * 0.01, chunk_index=i, text=long_text)
                  for i in range(5)]
        # With token_budget=100 (400 chars), at most 1 chunk fits
        result = assemble_context(chunks, threshold=0.35, token_budget=100)
        # At most 1 chunk due to budget -- possibly 0 if header overhead fills it
        assert result.chunk_count <= 2

    def test_sources_deduplicated_by_doc_id(self) -> None:
        chunks = [
            make_chunk_result(score=0.9, doc_id="doc_a", chunk_index=0),
            make_chunk_result(score=0.8, doc_id="doc_a", chunk_index=1),  # same doc
            make_chunk_result(score=0.7, doc_id="doc_b", chunk_index=0),
        ]
        result = assemble_context(chunks, threshold=0.35)
        assert len(result.sources) == 2  # doc_a and doc_b only
        doc_ids = {s.doc_id for s in result.sources}
        assert doc_ids == {"doc_a", "doc_b"}

    def test_source_chunk_count_correct(self) -> None:
        chunks = [
            make_chunk_result(score=0.9, doc_id="doc_a", chunk_index=0),
            make_chunk_result(score=0.85, doc_id="doc_a", chunk_index=1),
        ]
        result = assemble_context(chunks, threshold=0.35)
        doc_a_source = next(s for s in result.sources if s.doc_id == "doc_a")
        assert doc_a_source.chunk_count == 2

    def test_source_top_score_is_maximum(self) -> None:
        chunks = [
            make_chunk_result(score=0.9, doc_id="doc_a", chunk_index=0),
            make_chunk_result(score=0.7, doc_id="doc_a", chunk_index=1),
        ]
        result = assemble_context(chunks, threshold=0.35)
        doc_a = next(s for s in result.sources if s.doc_id == "doc_a")
        assert abs(doc_a.top_score - 0.9) < 1e-6

    def test_assembled_context_returns_correct_type(self) -> None:
        chunks = [make_chunk_result()]
        result = assemble_context(chunks, threshold=0.35)
        assert isinstance(result, AssembledContext)

    def test_char_count_matches_context_text(self) -> None:
        chunks = [make_chunk_result()]
        result = assemble_context(chunks, threshold=0.35)
        assert result.char_count == len(result.context_text)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


class TestPromptBuilder:

    def test_build_messages_returns_two_messages(self) -> None:
        messages = build_messages("context text here", "What is revenue?")
        assert len(messages) == 2

    def test_first_message_is_system(self) -> None:
        messages = build_messages("context", "query")
        assert messages[0]["role"] == "system"

    def test_second_message_is_user(self) -> None:
        messages = build_messages("context", "query")
        assert messages[1]["role"] == "user"

    def test_system_prompt_is_constant(self) -> None:
        messages = build_messages("context", "query")
        assert messages[0]["content"] == SYSTEM_PROMPT

    def test_user_message_contains_context(self) -> None:
        messages = build_messages("My important context text.", "What is revenue?")
        assert "My important context text." in messages[1]["content"]

    def test_user_message_contains_query(self) -> None:
        messages = build_messages("context", "What is the EBITDA margin?")
        assert "What is the EBITDA margin?" in messages[1]["content"]

    def test_user_message_wraps_context_in_tags(self) -> None:
        messages = build_messages("context body", "query")
        user_content = messages[1]["content"]
        assert "<CONTEXT>" in user_content
        assert "</CONTEXT>" in user_content

    def test_estimate_prompt_tokens_returns_integer(self) -> None:
        messages = build_messages("context here", "what is this?")
        estimate = estimate_prompt_tokens(messages)
        assert isinstance(estimate, int)
        assert estimate > 0

    def test_estimate_prompt_tokens_scales_with_content(self) -> None:
        short_msgs = build_messages("short", "q")
        long_msgs = build_messages("A" * 1000, "q")
        assert estimate_prompt_tokens(long_msgs) > estimate_prompt_tokens(short_msgs)


# ---------------------------------------------------------------------------
# LLM provider factory
# ---------------------------------------------------------------------------


class TestLLMProviderFactory:

    def test_local_provider_created(self) -> None:
        provider = create_llm_provider("local", "local", "", 30)
        assert isinstance(provider, LocalProvider)

    def test_openai_provider_created(self) -> None:
        provider = create_llm_provider("openai", "gpt-4o", "sk-test", 30)
        assert isinstance(provider, OpenAIProvider)

    def test_anthropic_provider_created(self) -> None:
        provider = create_llm_provider("anthropic", "claude-3-haiku-20240307", "sk-ant-test", 30)
        assert isinstance(provider, AnthropicProvider)

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(LLMProviderError, match="Unknown LLM provider"):
            create_llm_provider("gemini", "gemini-pro", "key", 30)

    def test_provider_stores_model_name(self) -> None:
        provider = create_llm_provider("local", "my-model", "", 30)
        assert provider.model_name == "my-model"

    def test_provider_stores_timeout(self) -> None:
        provider = create_llm_provider("local", "m", "", 60)
        assert provider.timeout_seconds == 60

    def test_all_providers_are_llm_provider_instances(self) -> None:
        for name in ("local", "openai", "anthropic"):
            provider = create_llm_provider(name, "model", "key", 30)
            assert isinstance(provider, LLMProvider)


# ---------------------------------------------------------------------------
# Local provider
# ---------------------------------------------------------------------------


class TestLocalProvider:

    def test_generate_returns_llm_response(self) -> None:
        provider = make_local_provider()
        messages = build_messages("context text", "what is revenue?")
        response = provider.generate(messages)
        assert isinstance(response, LLMResponse)

    def test_generate_content_is_non_empty_string(self) -> None:
        provider = make_local_provider()
        messages = build_messages("Some context here.", "query?")
        response = provider.generate(messages)
        assert isinstance(response.content, str)
        assert len(response.content) > 0

    def test_generate_reports_token_usage(self) -> None:
        provider = make_local_provider()
        messages = build_messages("context", "query")
        response = provider.generate(messages)
        assert response.token_usage.prompt_tokens > 0
        assert response.token_usage.completion_tokens > 0
        assert response.token_usage.total_tokens > 0

    def test_generate_reports_correct_provider(self) -> None:
        provider = make_local_provider()
        messages = build_messages("context", "query")
        response = provider.generate(messages)
        assert response.provider == "local"

    def test_generate_reports_latency(self) -> None:
        provider = make_local_provider()
        messages = build_messages("context", "query")
        response = provider.generate(messages)
        assert response.latency_ms >= 0

    def test_provider_name_is_local(self) -> None:
        provider = make_local_provider()
        assert provider.provider_name == "local"

    def test_generate_without_context_produces_response(self) -> None:
        provider = make_local_provider()
        # Message without context tags
        messages = [{"role": "user", "content": "plain query with no context"}]
        response = provider.generate(messages)
        assert isinstance(response.content, str)


# ---------------------------------------------------------------------------
# OpenAI provider (mocked SDK)
# ---------------------------------------------------------------------------


class TestOpenAIProvider:

    def _make_provider(self) -> OpenAIProvider:
        return OpenAIProvider(
            model_name="gpt-4o",
            api_key="sk-test",
            timeout_seconds=30,
        )

    def _make_openai_response(self, content: str = "The answer is here.") -> MagicMock:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = content
        mock_response.model = "gpt-4o"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 200
        mock_response.usage.completion_tokens = 50
        return mock_response

    def test_generate_returns_llm_response(self) -> None:
        provider = self._make_provider()
        messages = build_messages("context", "query")
        mock_resp = self._make_openai_response()
        mock_openai_mod = MagicMock()
        mock_openai_mod.OpenAI.return_value.chat.completions.create.return_value = mock_resp

        with patch.dict("sys.modules", {"openai": mock_openai_mod}):
            response = provider.generate(messages)

        assert isinstance(response, LLMResponse)

    def test_generate_returns_correct_content(self) -> None:
        provider = self._make_provider()
        messages = build_messages("context", "query")
        mock_resp = self._make_openai_response("Revenue was 2.4 billion.")

        mock_openai_mod = MagicMock()
        mock_openai_mod.OpenAI.return_value.chat.completions.create.return_value = mock_resp

        with patch.dict("sys.modules", {"openai": mock_openai_mod}):
            response = provider.generate(messages)

        assert response.content == "Revenue was 2.4 billion."

    def test_generate_reports_token_usage(self) -> None:
        provider = self._make_provider()
        messages = build_messages("context", "query")
        mock_resp = self._make_openai_response()

        mock_openai_mod = MagicMock()
        mock_openai_mod.OpenAI.return_value.chat.completions.create.return_value = mock_resp

        with patch.dict("sys.modules", {"openai": mock_openai_mod}):
            response = provider.generate(messages)

        assert response.token_usage.prompt_tokens == 200
        assert response.token_usage.completion_tokens == 50
        assert response.token_usage.total_tokens == 250

    def test_timeout_raises_llm_timeout_error(self) -> None:
        provider = self._make_provider()
        messages = build_messages("context", "query")
        mock_openai_mod = MagicMock()
        mock_openai_mod.APITimeoutError = Exception
        mock_openai_mod.APIError = ValueError
        mock_openai_mod.OpenAI.return_value.chat.completions.create.side_effect = (
            Exception("timeout")
        )
        with patch.dict("sys.modules", {"openai": mock_openai_mod}):
            with pytest.raises((LLMTimeoutError, LLMProviderError)):
                provider.generate(messages)

    def test_api_error_raises_llm_provider_error(self) -> None:
        provider = self._make_provider()
        messages = build_messages("context", "query")
        mock_openai_mod = MagicMock()
        mock_openai_mod.APITimeoutError = type("APITimeoutError", (Exception,), {})
        mock_openai_mod.APIError = type("APIError", (Exception,), {})

        class FakeAPIError(Exception):
            pass

        mock_openai_mod.APIError = FakeAPIError
        mock_openai_mod.APITimeoutError = type("APITimeoutError", (Exception,), {})
        mock_openai_mod.OpenAI.return_value.chat.completions.create.side_effect = (
            FakeAPIError("rate limit")
        )
        with patch.dict("sys.modules", {"openai": mock_openai_mod}):
            with pytest.raises(LLMProviderError):
                provider.generate(messages)

    def test_missing_package_raises_llm_provider_error(self) -> None:
        provider = self._make_provider()
        messages = build_messages("context", "query")

        # Remove openai from sys.modules to simulate it not being installed
        import sys
        saved = sys.modules.pop("openai", None)
        try:
            with patch.dict("sys.modules", {"openai": None}):  # type: ignore[dict-item]
                with pytest.raises((LLMProviderError, ImportError)):
                    provider.generate(messages)
        finally:
            if saved is not None:
                sys.modules["openai"] = saved

    def test_provider_name_is_openai(self) -> None:
        provider = self._make_provider()
        assert provider.provider_name == "openai"


# ---------------------------------------------------------------------------
# Anthropic provider (mocked SDK)
# ---------------------------------------------------------------------------


class TestAnthropicProvider:

    def _make_provider(self) -> AnthropicProvider:
        return AnthropicProvider(
            model_name="claude-3-haiku-20240307",
            api_key="sk-ant-test",
            timeout_seconds=30,
        )

    def _make_anthropic_response(self, content: str = "The answer is here.") -> MagicMock:
        mock_response = MagicMock()
        text_block = MagicMock()
        text_block.text = content
        mock_response.content = [text_block]
        mock_response.model = "claude-3-haiku-20240307"
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 300
        mock_response.usage.output_tokens = 80
        return mock_response

    def test_generate_returns_llm_response(self) -> None:
        provider = self._make_provider()
        messages = build_messages("context", "query")
        mock_resp = self._make_anthropic_response()
        mock_anth_mod = MagicMock()
        mock_anth_mod.Anthropic.return_value.messages.create.return_value = mock_resp

        with patch.dict("sys.modules", {"anthropic": mock_anth_mod}):
            response = provider.generate(messages)

        assert isinstance(response, LLMResponse)

    def test_generate_returns_correct_content(self) -> None:
        provider = self._make_provider()
        messages = build_messages("context", "query")
        mock_resp = self._make_anthropic_response("Cloud revenue grew 34 percent.")
        mock_anth_mod = MagicMock()
        mock_anth_mod.Anthropic.return_value.messages.create.return_value = mock_resp

        with patch.dict("sys.modules", {"anthropic": mock_anth_mod}):
            response = provider.generate(messages)

        assert response.content == "Cloud revenue grew 34 percent."

    def test_generate_reports_token_usage(self) -> None:
        provider = self._make_provider()
        messages = build_messages("context", "query")
        mock_resp = self._make_anthropic_response()
        mock_anth_mod = MagicMock()
        mock_anth_mod.Anthropic.return_value.messages.create.return_value = mock_resp

        with patch.dict("sys.modules", {"anthropic": mock_anth_mod}):
            response = provider.generate(messages)

        assert response.token_usage.prompt_tokens == 300
        assert response.token_usage.completion_tokens == 80
        assert response.token_usage.total_tokens == 380

    def test_system_message_separated_from_user_messages(self) -> None:
        provider = self._make_provider()
        messages = build_messages("context", "query")
        mock_resp = self._make_anthropic_response()
        mock_anth_mod = MagicMock()
        mock_anth_mod.Anthropic.return_value.messages.create.return_value = mock_resp

        with patch.dict("sys.modules", {"anthropic": mock_anth_mod}):
            provider.generate(messages)

        call_kwargs = mock_anth_mod.Anthropic.return_value.messages.create.call_args.kwargs
        # System message must be a string argument, not in the messages list
        assert "system" in call_kwargs
        assert call_kwargs["system"] == SYSTEM_PROMPT
        # User messages must not include the system role
        for msg in call_kwargs.get("messages", []):
            assert msg.get("role") != "system"

    def test_missing_package_raises_llm_provider_error(self) -> None:
        provider = self._make_provider()
        messages = build_messages("context", "query")

        with patch.dict("sys.modules", {"anthropic": None}):  # type: ignore[dict-item]
            with pytest.raises((LLMProviderError, ImportError)):
                provider.generate(messages)

    def test_provider_name_is_anthropic(self) -> None:
        provider = self._make_provider()
        assert provider.provider_name == "anthropic"


# ---------------------------------------------------------------------------
# QueryService -- full pipeline
# ---------------------------------------------------------------------------


class TestQueryService:

    def _make_service(
        self,
        chunks: list[ChunkResult] | None = None,
        provider: LLMProvider | None = None,
        threshold: float = 0.35,
        top_k: int = 5,
    ) -> QueryService:
        repo = make_mock_repository(chunks=chunks)
        llm = provider or make_local_provider()
        settings = make_mock_settings(top_k=top_k, threshold=threshold)
        return QueryService(repository=repo, llm_provider=llm, settings=settings)

    def _mock_embed(self, text: str) -> list[float]:
        return [0.1] * 384

    def test_query_returns_query_result(self) -> None:
        service = self._make_service()
        with patch("app.services.query_service.embed_single", side_effect=self._mock_embed):
            result = service.query("u1", "What was the revenue?")
        assert isinstance(result, QueryResult)

    def test_query_result_has_correct_user_id(self) -> None:
        service = self._make_service()
        with patch("app.services.query_service.embed_single", side_effect=self._mock_embed):
            result = service.query("u_abc", "query text")
        assert result.user_id == "u_abc"

    def test_query_result_preserves_query_text(self) -> None:
        service = self._make_service()
        with patch("app.services.query_service.embed_single", side_effect=self._mock_embed):
            result = service.query("u1", "What is the EBITDA margin?")
        assert result.query == "What is the EBITDA margin?"

    def test_query_result_route_is_retrieve(self) -> None:
        service = self._make_service()
        with patch("app.services.query_service.embed_single", side_effect=self._mock_embed):
            result = service.query("u1", "query")
        assert result.route == "RETRIEVE"

    def test_query_result_has_answer(self) -> None:
        service = self._make_service()
        with patch("app.services.query_service.embed_single", side_effect=self._mock_embed):
            result = service.query("u1", "query")
        assert result.has_answer

    def test_query_result_has_sources(self) -> None:
        service = self._make_service()
        with patch("app.services.query_service.embed_single", side_effect=self._mock_embed):
            result = service.query("u1", "query")
        assert len(result.sources) > 0

    def test_query_result_chunks_retrieved_correct(self) -> None:
        chunks = [make_chunk_result(score=0.9, chunk_index=i) for i in range(3)]
        service = self._make_service(chunks=chunks)
        with patch("app.services.query_service.embed_single", side_effect=self._mock_embed):
            result = service.query("u1", "query")
        assert result.chunks_retrieved == 3

    def test_query_uses_top_k_from_settings(self) -> None:
        repo = make_mock_repository()
        service = QueryService(
            repository=repo,
            llm_provider=make_local_provider(),
            settings=make_mock_settings(top_k=7),
        )
        with patch("app.services.query_service.embed_single", side_effect=self._mock_embed):
            service.query("u1", "query")
        call_kwargs = repo.search_chunks.call_args.kwargs
        assert call_kwargs["top_k"] == 7

    def test_query_top_k_override_used_when_provided(self) -> None:
        repo = make_mock_repository()
        service = QueryService(
            repository=repo,
            llm_provider=make_local_provider(),
            settings=make_mock_settings(top_k=5),
        )
        with patch("app.services.query_service.embed_single", side_effect=self._mock_embed):
            service.query("u1", "query", top_k=3)
        call_kwargs = repo.search_chunks.call_args.kwargs
        assert call_kwargs["top_k"] == 3

    def test_query_passes_user_id_to_repository(self) -> None:
        repo = make_mock_repository()
        service = QueryService(
            repository=repo,
            llm_provider=make_local_provider(),
            settings=make_mock_settings(),
        )
        with patch("app.services.query_service.embed_single", side_effect=self._mock_embed):
            service.query("u_tenant", "query")
        call_kwargs = repo.search_chunks.call_args.kwargs
        assert call_kwargs["user_id"] == "u_tenant"

    def test_query_result_latency_ms_is_non_negative(self) -> None:
        service = self._make_service()
        with patch("app.services.query_service.embed_single", side_effect=self._mock_embed):
            result = service.query("u1", "query")
        assert result.latency_ms >= 0

    def test_repository_error_propagates(self) -> None:
        repo = make_mock_repository()
        repo.search_chunks.side_effect = VectorStoreError("connection lost")
        service = QueryService(
            repository=repo,
            llm_provider=make_local_provider(),
            settings=make_mock_settings(),
        )
        with pytest.raises(VectorStoreError):
            with patch("app.services.query_service.embed_single", side_effect=self._mock_embed):
                service.query("u1", "query")

    def test_token_usage_present_in_result(self) -> None:
        service = self._make_service()
        with patch("app.services.query_service.embed_single", side_effect=self._mock_embed):
            result = service.query("u1", "query")
        assert isinstance(result.token_usage, TokenUsage)
        assert result.token_usage.total_tokens > 0


# ---------------------------------------------------------------------------
# QueryService -- no-result path
# ---------------------------------------------------------------------------


class TestQueryServiceNoResult:

    def _mock_embed(self, text: str) -> list[float]:
        return [0.1] * 384

    def test_no_result_when_all_chunks_below_threshold(self) -> None:
        # All chunks score below the threshold
        chunks = [make_chunk_result(score=0.1, chunk_index=i) for i in range(3)]
        repo = make_mock_repository(chunks=chunks)
        service = QueryService(
            repository=repo,
            llm_provider=make_local_provider(),
            settings=make_mock_settings(threshold=0.35),
        )
        with patch("app.services.query_service.embed_single", side_effect=self._mock_embed):
            result = service.query("u1", "query")
        assert result.answer is None
        assert result.is_no_result

    def test_no_result_when_empty_chunks_returned(self) -> None:
        repo = make_mock_repository(chunks=[])
        service = QueryService(
            repository=repo,
            llm_provider=make_local_provider(),
            settings=make_mock_settings(threshold=0.35),
        )
        with patch("app.services.query_service.embed_single", side_effect=self._mock_embed):
            result = service.query("u1", "query")
        assert result.answer is None
        assert result.is_no_result

    def test_no_result_sources_are_empty(self) -> None:
        repo = make_mock_repository(chunks=[])
        service = QueryService(
            repository=repo,
            llm_provider=make_local_provider(),
            settings=make_mock_settings(),
        )
        with patch("app.services.query_service.embed_single", side_effect=self._mock_embed):
            result = service.query("u1", "query")
        assert result.sources == []

    def test_no_result_token_usage_is_zero(self) -> None:
        repo = make_mock_repository(chunks=[])
        service = QueryService(
            repository=repo,
            llm_provider=make_local_provider(),
            settings=make_mock_settings(),
        )
        with patch("app.services.query_service.embed_single", side_effect=self._mock_embed):
            result = service.query("u1", "query")
        assert result.token_usage.total_tokens == 0

    def test_no_result_llm_not_called(self) -> None:
        repo = make_mock_repository(chunks=[])
        mock_provider = MagicMock(spec=LLMProvider)
        service = QueryService(
            repository=repo,
            llm_provider=mock_provider,
            settings=make_mock_settings(),
        )
        with patch("app.services.query_service.embed_single", side_effect=self._mock_embed):
            service.query("u1", "query")
        mock_provider.generate.assert_not_called()

    def test_no_result_reason_is_set(self) -> None:
        repo = make_mock_repository(chunks=[])
        service = QueryService(
            repository=repo,
            llm_provider=make_local_provider(),
            settings=make_mock_settings(),
        )
        with patch("app.services.query_service.embed_single", side_effect=self._mock_embed):
            result = service.query("u1", "query")
        assert result.no_result_reason is not None
        assert len(result.no_result_reason) > 0

    def test_no_result_route_still_retrieve(self) -> None:
        repo = make_mock_repository(chunks=[])
        service = QueryService(
            repository=repo,
            llm_provider=make_local_provider(),
            settings=make_mock_settings(),
        )
        with patch("app.services.query_service.embed_single", side_effect=self._mock_embed):
            result = service.query("u1", "query")
        assert result.route == "RETRIEVE"

    def test_no_result_chunks_used_is_zero(self) -> None:
        repo = make_mock_repository(chunks=[])
        service = QueryService(
            repository=repo,
            llm_provider=make_local_provider(),
            settings=make_mock_settings(),
        )
        with patch("app.services.query_service.embed_single", side_effect=self._mock_embed):
            result = service.query("u1", "query")
        assert result.chunks_used == 0
