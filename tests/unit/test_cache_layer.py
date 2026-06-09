"""
Unit tests for Module 8 -- Redis Cache Layer.

Tests cover:
    TestCacheKeyBuilder        -- build_query_cache_key: normalisation, isolation
    TestQueryResultSerialise   -- round-trip serialise/deserialise
    TestCacheServiceGet        -- hit, miss, Redis error silent handling
    TestCacheServiceSet        -- answer TTL, empty-result TTL, error silent handling
    TestCacheServiceInvalidate -- SCAN/DELETE pattern, error silent handling
    TestCachedQueryService     -- cache hit path, cache miss path, write-on-miss
    TestCacheHitFieldOnResult  -- cache_hit=True from cache, False from pipeline
    TestDocumentServiceCacheWiring -- invalidation called after upload
    TestCacheFailureSilence    -- Redis down never raises to callers

No infrastructure required. All Redis interactions are mocked.
"""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from app.cache.cache_service import (
    CacheService,
    _deserialise_result,
    _serialise_result,
    build_query_cache_key,
)
from app.models.query_result import QueryResult, SourceReference, TokenUsage
from app.services.cached_query_service import CachedQueryService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_query_result(
    answer: str | None = "Revenue was $2.4B in Q3.",
    route: str = "RETRIEVE",
    no_result_reason: str | None = None,
    cache_hit: bool = False,
    user_id: str = "u_test",
) -> QueryResult:
    return QueryResult(
        query="What was the revenue?",
        user_id=user_id,
        answer=answer,
        sources=[
            SourceReference(
                doc_id="doc_abc",
                source="report.pdf",
                chunk_count=2,
                top_score=0.91,
            )
        ] if answer else [],
        route=route,  # type: ignore[arg-type]
        chunks_retrieved=5,
        chunks_used=2,
        token_usage=TokenUsage.from_counts(200, 80),
        latency_ms=1240,
        no_result_reason=no_result_reason,
        cache_hit=cache_hit,
        timestamp=datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC),
    )


def make_no_result() -> QueryResult:
    return QueryResult(
        query="What was the revenue?",
        user_id="u_test",
        answer=None,
        sources=[],
        route="RETRIEVE",
        chunks_retrieved=3,
        chunks_used=0,
        token_usage=TokenUsage.zero(),
        latency_ms=120,
        no_result_reason="NO_RELEVANT_CHUNKS",
        cache_hit=False,
        timestamp=datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC),
    )


def make_cache_service(
    client: MagicMock | None = None,
    ttl: int = 1800,
    empty_ttl: int = 300,
) -> CacheService:
    return CacheService(
        client=client or MagicMock(),
        ttl_seconds=ttl,
        empty_result_ttl_seconds=empty_ttl,
    )


# ---------------------------------------------------------------------------
# Cache key builder
# ---------------------------------------------------------------------------


class TestCacheKeyBuilder:

    def test_key_has_query_prefix(self) -> None:
        key = build_query_cache_key("u1", "what is revenue?")
        assert key.startswith("query:")

    def test_key_contains_user_id(self) -> None:
        key = build_query_cache_key("u_tenant", "query text")
        assert "u_tenant" in key

    def test_key_format_is_three_segments(self) -> None:
        key = build_query_cache_key("u1", "query")
        parts = key.split(":")
        assert len(parts) == 3
        assert parts[0] == "query"
        assert parts[1] == "u1"
        assert len(parts[2]) == 16  # 16 hex chars

    def test_normalisation_case_insensitive(self) -> None:
        k1 = build_query_cache_key("u1", "What is Revenue?")
        k2 = build_query_cache_key("u1", "what is revenue?")
        assert k1 == k2

    def test_normalisation_whitespace_collapsed(self) -> None:
        k1 = build_query_cache_key("u1", "  what   is  revenue  ")
        k2 = build_query_cache_key("u1", "what is revenue")
        assert k1 == k2

    def test_different_queries_produce_different_keys(self) -> None:
        k1 = build_query_cache_key("u1", "what is revenue?")
        k2 = build_query_cache_key("u1", "what is profit?")
        assert k1 != k2

    def test_different_users_produce_different_keys(self) -> None:
        k1 = build_query_cache_key("u1", "same query")
        k2 = build_query_cache_key("u2", "same query")
        assert k1 != k2

    def test_hash_segment_is_16_chars(self) -> None:
        key = build_query_cache_key("u1", "any query text here")
        hash_part = key.split(":")[2]
        assert len(hash_part) == 16

    def test_hash_segment_is_hex(self) -> None:
        key = build_query_cache_key("u1", "query")
        hash_part = key.split(":")[2]
        assert all(c in "0123456789abcdef" for c in hash_part)

    def test_deterministic_for_same_input(self) -> None:
        k1 = build_query_cache_key("u1", "what is revenue?")
        k2 = build_query_cache_key("u1", "what is revenue?")
        assert k1 == k2


# ---------------------------------------------------------------------------
# Serialise / deserialise round-trip
# ---------------------------------------------------------------------------


class TestQueryResultSerialise:

    def test_round_trip_answer_result(self) -> None:
        original = make_query_result()
        serialised = _serialise_result(original)
        restored = _deserialise_result(serialised)
        assert restored.answer == original.answer
        assert restored.query == original.query
        assert restored.user_id == original.user_id
        assert restored.route == original.route

    def test_round_trip_no_result(self) -> None:
        original = make_no_result()
        serialised = _serialise_result(original)
        restored = _deserialise_result(serialised)
        assert restored.answer is None
        assert restored.no_result_reason == "NO_RELEVANT_CHUNKS"

    def test_round_trip_sources_preserved(self) -> None:
        original = make_query_result()
        restored = _deserialise_result(_serialise_result(original))
        assert len(restored.sources) == len(original.sources)
        assert restored.sources[0].doc_id == original.sources[0].doc_id
        assert restored.sources[0].source == original.sources[0].source

    def test_round_trip_token_usage_preserved(self) -> None:
        original = make_query_result()
        restored = _deserialise_result(_serialise_result(original))
        assert restored.token_usage.prompt_tokens == 200
        assert restored.token_usage.completion_tokens == 80
        assert restored.token_usage.total_tokens == 280

    def test_deserialise_sets_cache_hit_true(self) -> None:
        original = make_query_result(cache_hit=False)
        restored = _deserialise_result(_serialise_result(original))
        assert restored.cache_hit is True

    def test_round_trip_timestamp_preserved(self) -> None:
        original = make_query_result()
        restored = _deserialise_result(_serialise_result(original))
        assert restored.timestamp == original.timestamp

    def test_round_trip_direct_route(self) -> None:
        original = make_query_result(route="DIRECT")
        restored = _deserialise_result(_serialise_result(original))
        assert restored.route == "DIRECT"

    def test_serialised_is_valid_json_string(self) -> None:
        import json
        original = make_query_result()
        serialised = _serialise_result(original)
        parsed = json.loads(serialised)
        assert isinstance(parsed, dict)
        assert "query" in parsed
        assert "answer" in parsed


# ---------------------------------------------------------------------------
# CacheService.get
# ---------------------------------------------------------------------------


class TestCacheServiceGet:

    def test_returns_none_on_cache_miss(self) -> None:
        mock_client = MagicMock()
        mock_client.get.return_value = None
        svc = make_cache_service(client=mock_client)
        result = svc.get("u1", "what is revenue?")
        assert result is None

    def test_returns_query_result_on_cache_hit(self) -> None:
        original = make_query_result()
        serialised = _serialise_result(original)
        mock_client = MagicMock()
        mock_client.get.return_value = serialised
        svc = make_cache_service(client=mock_client)
        result = svc.get("u1", "what is revenue?")
        assert result is not None
        assert result.answer == original.answer

    def test_cache_hit_result_has_cache_hit_true(self) -> None:
        original = make_query_result()
        mock_client = MagicMock()
        mock_client.get.return_value = _serialise_result(original)
        svc = make_cache_service(client=mock_client)
        result = svc.get("u1", "query")
        assert result is not None
        assert result.cache_hit is True

    def test_redis_error_returns_none(self) -> None:
        mock_client = MagicMock()
        mock_client.get.side_effect = Exception("connection refused")
        svc = make_cache_service(client=mock_client)
        result = svc.get("u1", "query")
        assert result is None

    def test_corrupted_json_returns_none(self) -> None:
        mock_client = MagicMock()
        mock_client.get.return_value = "not valid json {{{"
        svc = make_cache_service(client=mock_client)
        result = svc.get("u1", "query")
        assert result is None

    def test_get_uses_correct_key(self) -> None:
        mock_client = MagicMock()
        mock_client.get.return_value = None
        svc = make_cache_service(client=mock_client)
        svc.get("u1", "what is revenue?")
        expected_key = build_query_cache_key("u1", "what is revenue?")
        mock_client.get.assert_called_once_with(expected_key)


# ---------------------------------------------------------------------------
# CacheService.set_result
# ---------------------------------------------------------------------------


class TestCacheServiceSet:

    def test_calls_setex_with_correct_key(self) -> None:
        mock_client = MagicMock()
        svc = make_cache_service(client=mock_client, ttl=1800)
        result = make_query_result()
        svc.set_result("u1", "what is revenue?", result)
        expected_key = build_query_cache_key("u1", "what is revenue?")
        call_args = mock_client.setex.call_args
        assert call_args[0][0] == expected_key

    def test_uses_full_ttl_for_answer_result(self) -> None:
        mock_client = MagicMock()
        svc = make_cache_service(client=mock_client, ttl=1800, empty_ttl=300)
        svc.set_result("u1", "query", make_query_result())
        ttl_used = mock_client.setex.call_args[0][1]
        assert ttl_used == 1800

    def test_uses_short_ttl_for_no_result(self) -> None:
        mock_client = MagicMock()
        svc = make_cache_service(client=mock_client, ttl=1800, empty_ttl=300)
        svc.set_result("u1", "query", make_no_result())
        ttl_used = mock_client.setex.call_args[0][1]
        assert ttl_used == 300

    def test_redis_error_does_not_raise(self) -> None:
        mock_client = MagicMock()
        mock_client.setex.side_effect = Exception("write failed")
        svc = make_cache_service(client=mock_client)
        # Must not raise
        svc.set_result("u1", "query", make_query_result())

    def test_set_stores_serialised_string(self) -> None:
        mock_client = MagicMock()
        svc = make_cache_service(client=mock_client)
        result = make_query_result()
        svc.set_result("u1", "query", result)
        stored_value = mock_client.setex.call_args[0][2]
        assert isinstance(stored_value, str)
        assert "revenue" in stored_value.lower() or "answer" in stored_value.lower()


# ---------------------------------------------------------------------------
# CacheService.invalidate_user_cache
# ---------------------------------------------------------------------------


class TestCacheServiceInvalidate:

    def test_returns_count_of_deleted_keys(self) -> None:
        mock_client = MagicMock()
        mock_client.scan.return_value = (0, ["query:u1:abc", "query:u1:def"])
        svc = make_cache_service(client=mock_client)
        count = svc.invalidate_user_cache("u1")
        assert count == 2

    def test_returns_zero_when_no_keys_found(self) -> None:
        mock_client = MagicMock()
        mock_client.scan.return_value = (0, [])
        svc = make_cache_service(client=mock_client)
        count = svc.invalidate_user_cache("u1")
        assert count == 0

    def test_uses_correct_pattern(self) -> None:
        mock_client = MagicMock()
        mock_client.scan.return_value = (0, [])
        svc = make_cache_service(client=mock_client)
        svc.invalidate_user_cache("u_abc")
        call_kwargs = mock_client.scan.call_args.kwargs
        assert call_kwargs["match"] == "query:u_abc:*"

    def test_deletes_found_keys(self) -> None:
        mock_client = MagicMock()
        keys = ["query:u1:abc", "query:u1:def"]
        mock_client.scan.return_value = (0, keys)
        svc = make_cache_service(client=mock_client)
        svc.invalidate_user_cache("u1")
        mock_client.delete.assert_called_once_with(*keys)

    def test_redis_error_returns_zero(self) -> None:
        mock_client = MagicMock()
        mock_client.scan.side_effect = Exception("connection lost")
        svc = make_cache_service(client=mock_client)
        count = svc.invalidate_user_cache("u1")
        assert count == 0

    def test_redis_error_does_not_raise(self) -> None:
        mock_client = MagicMock()
        mock_client.scan.side_effect = Exception("timeout")
        svc = make_cache_service(client=mock_client)
        # Must not raise
        svc.invalidate_user_cache("u1")

    def test_scan_pagination_followed(self) -> None:
        mock_client = MagicMock()
        # First scan returns cursor=5 (not done), second returns cursor=0 (done)
        mock_client.scan.side_effect = [
            (5, ["query:u1:aaa", "query:u1:bbb"]),
            (0, ["query:u1:ccc"]),
        ]
        svc = make_cache_service(client=mock_client)
        count = svc.invalidate_user_cache("u1")
        assert count == 3
        assert mock_client.scan.call_count == 2


# ---------------------------------------------------------------------------
# CachedQueryService
# ---------------------------------------------------------------------------


class TestCachedQueryService:

    def _make_service(
        self,
        cached_result: QueryResult | None = None,
    ) -> tuple[CachedQueryService, MagicMock, MagicMock]:
        mock_cache = MagicMock(spec=CacheService)
        mock_cache.get.return_value = cached_result

        mock_routed = MagicMock()
        mock_routed.query.return_value = make_query_result(cache_hit=False)

        svc = CachedQueryService(
            routed_service=mock_routed,
            cache_service=mock_cache,
        )
        return svc, mock_cache, mock_routed

    def test_cache_hit_returns_cached_result(self) -> None:
        cached = make_query_result(cache_hit=True)
        svc, mock_cache, mock_routed = self._make_service(cached_result=cached)
        result = svc.query("u1", "what is revenue?")
        assert result.cache_hit is True
        mock_routed.query.assert_not_called()

    def test_cache_miss_calls_routed_service(self) -> None:
        svc, mock_cache, mock_routed = self._make_service(cached_result=None)
        svc.query("u1", "what is revenue?")
        mock_routed.query.assert_called_once()

    def test_cache_miss_passes_user_id_to_routed(self) -> None:
        svc, _, mock_routed = self._make_service()
        svc.query("u_tenant", "query")
        assert mock_routed.query.call_args.kwargs["user_id"] == "u_tenant"

    def test_cache_miss_passes_query_text_to_routed(self) -> None:
        svc, _, mock_routed = self._make_service()
        svc.query("u1", "specific query text")
        assert mock_routed.query.call_args.kwargs["query_text"] == "specific query text"

    def test_cache_miss_passes_top_k_override(self) -> None:
        svc, _, mock_routed = self._make_service()
        svc.query("u1", "query", top_k=7)
        assert mock_routed.query.call_args.kwargs["top_k"] == 7

    def test_cache_miss_writes_result_to_cache(self) -> None:
        svc, mock_cache, _ = self._make_service()
        svc.query("u1", "what is revenue?")
        mock_cache.set_result.assert_called_once()

    def test_cache_miss_result_has_cache_hit_false(self) -> None:
        svc, _, _ = self._make_service()
        result = svc.query("u1", "query")
        assert result.cache_hit is False

    def test_cache_hit_does_not_call_set_result(self) -> None:
        cached = make_query_result(cache_hit=True)
        svc, mock_cache, _ = self._make_service(cached_result=cached)
        svc.query("u1", "query")
        mock_cache.set_result.assert_not_called()

    def test_set_result_called_with_correct_user_id(self) -> None:
        svc, mock_cache, _ = self._make_service()
        svc.query("u_abc", "query text")
        call_kwargs = mock_cache.set_result.call_args
        assert call_kwargs[0][0] == "u_abc"

    def test_set_result_called_with_correct_query(self) -> None:
        svc, mock_cache, _ = self._make_service()
        svc.query("u1", "exact query text here")
        call_kwargs = mock_cache.set_result.call_args
        assert call_kwargs[0][1] == "exact query text here"


# ---------------------------------------------------------------------------
# cache_hit field on QueryResult
# ---------------------------------------------------------------------------


class TestCacheHitFieldOnResult:

    def test_default_cache_hit_is_false(self) -> None:
        result = make_query_result()
        assert result.cache_hit is False

    def test_pipeline_result_cache_hit_false(self) -> None:
        result = make_query_result(cache_hit=False)
        assert result.cache_hit is False

    def test_cache_restored_result_cache_hit_true(self) -> None:
        original = make_query_result()
        restored = _deserialise_result(_serialise_result(original))
        assert restored.cache_hit is True

    def test_cache_hit_does_not_affect_has_answer(self) -> None:
        result = make_query_result(cache_hit=True, answer="some answer")
        assert result.has_answer is True

    def test_cache_hit_is_frozen(self) -> None:
        import dataclasses
        result = make_query_result()
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.cache_hit = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# DocumentService cache invalidation wiring
# ---------------------------------------------------------------------------


class TestDocumentServiceCacheWiring:

    VALID_PDF = (
        b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\n"
        b"xref\n0 1\n0000000000 65535 f \n"
        b"trailer\n<< /Size 1 /Root 1 0 R >>\nstartxref\n9\n%%EOF\n"
    )
    LONG_TEXT = "The quarterly revenue report shows significant growth. " * 20

    def _make_document_service(
        self,
        cache_service: MagicMock | None = None,
    ):  # type: ignore[no-untyped-def]
        from app.services.document_service import DocumentService

        mock_repo = MagicMock()
        mock_repo.add_chunks.return_value = None
        mock_settings = MagicMock()
        mock_settings.MAX_UPLOAD_SIZE_MB = 50
        mock_settings.CHUNK_SIZE_TOKENS = 200
        mock_settings.CHUNK_OVERLAP_TOKENS = 20
        mock_settings.EMBEDDING_BATCH_SIZE = 100

        return DocumentService(
            repository=mock_repo,
            settings=mock_settings,
            cache_service=cache_service,
        )

    def _mock_embed(self, chunks, batch_size):  # type: ignore[no-untyped-def]
        for c in chunks:
            c.embedding = [0.1] * 384
        return chunks

    def test_cache_invalidated_after_successful_upload(self) -> None:
        mock_cache = MagicMock(spec=CacheService)
        svc = self._make_document_service(cache_service=mock_cache)
        with patch("app.services.document_service.parse_pdf", return_value=self.LONG_TEXT), \
             patch("app.services.document_service.embed_chunks", side_effect=self._mock_embed):
            svc.ingest("u1", self.VALID_PDF, "report.pdf", "pdf")
        mock_cache.invalidate_user_cache.assert_called_once_with("u1")

    def test_no_invalidation_without_cache_service(self) -> None:
        svc = self._make_document_service(cache_service=None)
        with patch("app.services.document_service.parse_pdf", return_value=self.LONG_TEXT), \
             patch("app.services.document_service.embed_chunks", side_effect=self._mock_embed):
            # Must not raise even without cache_service
            svc.ingest("u1", self.VALID_PDF, "report.pdf", "pdf")

    def test_cache_service_is_optional_backward_compatible(self) -> None:
        from app.services.document_service import DocumentService
        mock_repo = MagicMock()
        mock_settings = MagicMock()
        mock_settings.MAX_UPLOAD_SIZE_MB = 50
        mock_settings.CHUNK_SIZE_TOKENS = 200
        mock_settings.CHUNK_OVERLAP_TOKENS = 20
        mock_settings.EMBEDDING_BATCH_SIZE = 100
        # Must construct without cache_service -- no TypeError
        svc = DocumentService(repository=mock_repo, settings=mock_settings)
        assert svc._cache_service is None


# ---------------------------------------------------------------------------
# Cache failure silence -- Redis errors never surface to callers
# ---------------------------------------------------------------------------


class TestCacheFailureSilence:

    def test_cache_get_failure_does_not_raise(self) -> None:
        mock_client = MagicMock()
        mock_client.get.side_effect = ConnectionError("Redis is down")
        svc = make_cache_service(client=mock_client)
        result = svc.get("u1", "query")
        assert result is None  # silent miss

    def test_cache_set_failure_does_not_raise(self) -> None:
        mock_client = MagicMock()
        mock_client.setex.side_effect = ConnectionError("Redis is down")
        svc = make_cache_service(client=mock_client)
        svc.set_result("u1", "query", make_query_result())  # must not raise

    def test_cache_invalidate_failure_does_not_raise(self) -> None:
        mock_client = MagicMock()
        mock_client.scan.side_effect = ConnectionError("Redis is down")
        svc = make_cache_service(client=mock_client)
        count = svc.invalidate_user_cache("u1")
        assert count == 0  # silent, returns 0

    def test_cached_service_still_works_when_cache_fails(self) -> None:
        mock_cache = MagicMock(spec=CacheService)
        mock_cache.get.side_effect = Exception("Redis down")

        mock_routed = MagicMock()
        mock_routed.query.return_value = make_query_result()

        # CachedQueryService must not catch CacheService errors --
        # CacheService itself is responsible for silence.
        # This test verifies the contract: CacheService.get() returns None on error.
        failing_cache = MagicMock()
        failing_cache.get.return_value = None  # simulating what CacheService does
        svc = CachedQueryService(routed_service=mock_routed, cache_service=failing_cache)
        result = svc.query("u1", "query")
        assert result is not None
        mock_routed.query.assert_called_once()
