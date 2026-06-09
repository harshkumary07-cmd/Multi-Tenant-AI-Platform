"""
Unit tests for Module 7 -- Router Agent and Routed Query Service.

Tests cover:
    TestRouteDecision          -- RouteDecision dataclass properties
    TestRouterAgentRule1       -- No documents -> DIRECT
    TestRouterAgentRule2       -- Filename signal -> RETRIEVE
    TestRouterAgentRule3       -- Strong RETRIEVE keywords -> RETRIEVE
    TestRouterAgentRule4       -- Strong DIRECT keywords -> DIRECT
    TestRouterAgentRule5       -- Ambiguous default -> RETRIEVE
    TestRouterAgentEdgeCases   -- Empty query, mixed signals, long queries
    TestRouterAgentFixtures    -- 50-case parametrized fixture suite (ADR-003)
    TestRoutedQueryService     -- Coordinator dispatch (RETRIEVE and DIRECT paths)
    TestBuildDirectMessages    -- build_direct_messages() from prompt_builder

No infrastructure required. All tests use mocked repository.
"""

from unittest.mock import MagicMock

import pytest

from app.agents.router_agent import (
    REASON_AMBIGUOUS_DEFAULT,
    REASON_DIRECT_KEYWORD,
    REASON_FILENAME_SIGNAL,
    REASON_NO_DOCUMENTS,
    REASON_RETRIEVE_KEYWORD,
    RouteDecision,
    RouterAgent,
)
from app.models.exceptions import VectorStoreError
from app.models.query_result import QueryResult, SourceReference, TokenUsage
from app.rag.prompt_builder import (
    DIRECT_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_direct_messages,
)
from app.services.routed_query_service import RoutedQueryService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_repo(doc_count: int = 1) -> MagicMock:
    repo = MagicMock()
    repo.count_documents.return_value = doc_count
    return repo


def make_router(doc_count: int = 1) -> RouterAgent:
    return RouterAgent(repository=make_repo(doc_count))


def make_query_result(route: str = "RETRIEVE") -> QueryResult:
    return QueryResult(
        query="test query",
        user_id="u_test",
        answer="The answer is 42.",
        sources=[SourceReference("doc_a", "report.pdf", 2, 0.9)],
        route=route,  # type: ignore[arg-type]
        chunks_retrieved=3,
        chunks_used=2,
        token_usage=TokenUsage.from_counts(200, 50),
        latency_ms=1200,
    )


def make_mock_settings() -> MagicMock:
    s = MagicMock()
    s.RETRIEVAL_TOP_K = 5
    s.RETRIEVAL_CONFIDENCE_THRESHOLD = 0.35
    return s


# ---------------------------------------------------------------------------
# TestRouteDecision
# ---------------------------------------------------------------------------


class TestRouteDecision:

    def test_is_direct_true_when_direct(self) -> None:
        d = RouteDecision(route="DIRECT", reason=REASON_NO_DOCUMENTS)
        assert d.is_direct is True
        assert d.is_retrieve is False

    def test_is_retrieve_true_when_retrieve(self) -> None:
        d = RouteDecision(route="RETRIEVE", reason=REASON_AMBIGUOUS_DEFAULT)
        assert d.is_retrieve is True
        assert d.is_direct is False

    def test_route_field_accessible(self) -> None:
        d = RouteDecision(route="DIRECT", reason=REASON_NO_DOCUMENTS)
        assert d.route == "DIRECT"

    def test_reason_field_accessible(self) -> None:
        d = RouteDecision(route="RETRIEVE", reason=REASON_FILENAME_SIGNAL)
        assert d.reason == REASON_FILENAME_SIGNAL

    def test_is_frozen(self) -> None:
        import dataclasses
        d = RouteDecision(route="DIRECT", reason=REASON_NO_DOCUMENTS)
        with pytest.raises(dataclasses.FrozenInstanceError):
            d.route = "RETRIEVE"  # type: ignore[misc]

    def test_equality(self) -> None:
        d1 = RouteDecision(route="DIRECT", reason=REASON_NO_DOCUMENTS)
        d2 = RouteDecision(route="DIRECT", reason=REASON_NO_DOCUMENTS)
        assert d1 == d2

    def test_inequality_different_route(self) -> None:
        d1 = RouteDecision(route="DIRECT", reason=REASON_NO_DOCUMENTS)
        d2 = RouteDecision(route="RETRIEVE", reason=REASON_NO_DOCUMENTS)
        assert d1 != d2


# ---------------------------------------------------------------------------
# TestRouterAgentRule1 -- No documents -> DIRECT
# ---------------------------------------------------------------------------


class TestRouterAgentRule1:

    def test_zero_documents_returns_direct(self) -> None:
        router = make_router(doc_count=0)
        d = router.decide("u1", "what is the revenue?")
        assert d.route == "DIRECT"
        assert d.reason == REASON_NO_DOCUMENTS

    def test_zero_documents_any_query_returns_direct(self) -> None:
        router = make_router(doc_count=0)
        queries = [
            "summarise my documents",
            "what does the report say",
            "annual_report.pdf",
            "hello",
            "",
        ]
        for q in queries:
            d = router.decide("u1", q)
            assert d.route == "DIRECT", f"Expected DIRECT for query: {q!r}"
            assert d.reason == REASON_NO_DOCUMENTS

    def test_one_document_does_not_trigger_rule1(self) -> None:
        router = make_router(doc_count=1)
        # With one document, Rule 1 does NOT fire -- other rules apply
        d = router.decide("u1", "hello")
        assert d.reason != REASON_NO_DOCUMENTS

    def test_count_documents_called_with_correct_user_id(self) -> None:
        repo = make_repo(doc_count=0)
        router = RouterAgent(repository=repo)
        router.decide("tenant_xyz", "any query")
        repo.count_documents.assert_called_once_with("tenant_xyz")

    def test_vector_store_error_propagates(self) -> None:
        repo = make_repo()
        repo.count_documents.side_effect = VectorStoreError("db down")
        router = RouterAgent(repository=repo)
        with pytest.raises(VectorStoreError):
            router.decide("u1", "query")


# ---------------------------------------------------------------------------
# TestRouterAgentRule2 -- Filename signal -> RETRIEVE
# ---------------------------------------------------------------------------


class TestRouterAgentRule2:

    def test_pdf_extension_triggers_retrieve(self) -> None:
        router = make_router(doc_count=1)
        d = router.decide("u1", "what does annual_report.pdf say about revenue?")
        assert d.route == "RETRIEVE"
        assert d.reason == REASON_FILENAME_SIGNAL

    def test_csv_extension_triggers_retrieve(self) -> None:
        router = make_router(doc_count=1)
        d = router.decide("u1", "analyse the data in sales_data.csv")
        assert d.route == "RETRIEVE"
        assert d.reason == REASON_FILENAME_SIGNAL

    def test_pdf_uppercase_triggers_retrieve(self) -> None:
        router = make_router(doc_count=1)
        d = router.decide("u1", "open REPORT.PDF and summarise it")
        assert d.route == "RETRIEVE"
        assert d.reason == REASON_FILENAME_SIGNAL

    def test_pdf_mixed_case_triggers_retrieve(self) -> None:
        router = make_router(doc_count=1)
        d = router.decide("u1", "look at Report.Pdf please")
        assert d.route == "RETRIEVE"
        assert d.reason == REASON_FILENAME_SIGNAL

    def test_filename_with_spaces_triggers_retrieve(self) -> None:
        router = make_router(doc_count=1)
        d = router.decide("u1", "what is in q3 report.pdf?")
        assert d.route == "RETRIEVE"
        assert d.reason == REASON_FILENAME_SIGNAL

    def test_bare_pdf_word_does_not_trigger(self) -> None:
        # "pdf" alone without a filename prefix should NOT trigger Rule 2
        router = make_router(doc_count=1)
        d = router.decide("u1", "what is a pdf format?")
        # Should fall through to Rule 4 (direct keyword "what is")
        assert d.reason != REASON_FILENAME_SIGNAL

    def test_filename_overrides_direct_keyword(self) -> None:
        # Rule 2 (filename) is checked before Rule 4 (direct keywords)
        router = make_router(doc_count=1)
        d = router.decide("u1", "what is in my_data.csv")
        assert d.route == "RETRIEVE"
        assert d.reason == REASON_FILENAME_SIGNAL


# ---------------------------------------------------------------------------
# TestRouterAgentRule3 -- Strong RETRIEVE keywords -> RETRIEVE
# ---------------------------------------------------------------------------


class TestRouterAgentRule3:

    def _check_retrieve(self, query: str) -> None:
        router = make_router(doc_count=1)
        d = router.decide("u1", query)
        assert d.route == "RETRIEVE", f"Expected RETRIEVE for: {query!r}"
        assert d.reason == REASON_RETRIEVE_KEYWORD

    def test_summarise(self) -> None:
        self._check_retrieve("summarise the main points")

    def test_summarize_american(self) -> None:
        self._check_retrieve("summarize the document for me")

    def test_from_my_file(self) -> None:
        self._check_retrieve("what can you tell me from my file?")

    def test_from_my_document(self) -> None:
        self._check_retrieve("extract the key data from my document")

    def test_in_my_document(self) -> None:
        self._check_retrieve("is the revenue figure in my document?")

    def test_in_the_document(self) -> None:
        self._check_retrieve("is there a table in the document?")

    def test_according_to(self) -> None:
        self._check_retrieve("according to the report, what was Q3 revenue?")

    def test_what_does_the_document_say(self) -> None:
        self._check_retrieve("what does the document say about margins?")

    def test_from_the_file(self) -> None:
        self._check_retrieve("get the figures from the file")

    def test_based_on_the_document(self) -> None:
        self._check_retrieve("based on the document, what should we do?")

    def test_as_per(self) -> None:
        self._check_retrieve("as per the report, revenue grew 34%")

    def test_what_do_the_documents(self) -> None:
        self._check_retrieve("what do the documents indicate about growth?")

    def test_retrieve_keyword_case_insensitive(self) -> None:
        router = make_router(doc_count=1)
        d = router.decide("u1", "SUMMARISE the quarterly results")
        assert d.route == "RETRIEVE"
        assert d.reason == REASON_RETRIEVE_KEYWORD


# ---------------------------------------------------------------------------
# TestRouterAgentRule4 -- Strong DIRECT keywords -> DIRECT
# ---------------------------------------------------------------------------


class TestRouterAgentRule4:

    def _check_direct(self, query: str) -> None:
        router = make_router(doc_count=1)
        d = router.decide("u1", query)
        assert d.route == "DIRECT", f"Expected DIRECT for: {query!r}"
        assert d.reason == REASON_DIRECT_KEYWORD

    def test_what_is(self) -> None:
        self._check_direct("what is compound interest?")

    def test_what_are(self) -> None:
        self._check_direct("what are the main causes of inflation?")

    def test_who_is(self) -> None:
        self._check_direct("who is the CEO of Apple?")

    def test_who_was(self) -> None:
        self._check_direct("who was the first president of the USA?")

    def test_define(self) -> None:
        self._check_direct("define amortization")

    def test_definition_of(self) -> None:
        self._check_direct("definition of EBITDA")

    def test_explain(self) -> None:
        self._check_direct("explain how interest rates work")

    def test_how_does(self) -> None:
        self._check_direct("how does compound interest work?")

    def test_how_do(self) -> None:
        self._check_direct("how do stock options work?")

    def test_tell_me_about(self) -> None:
        self._check_direct("tell me about the history of finance")

    def test_describe(self) -> None:
        self._check_direct("describe the difference between debt and equity")

    def test_when_was(self) -> None:
        self._check_direct("when was the Federal Reserve founded?")

    def test_where_is(self) -> None:
        self._check_direct("where is the NYSE located?")

    def test_why_is(self) -> None:
        self._check_direct("why is inflation bad for bond prices?")

    def test_direct_keyword_case_insensitive(self) -> None:
        router = make_router(doc_count=1)
        d = router.decide("u1", "WHAT IS the capital of France?")
        assert d.route == "DIRECT"
        assert d.reason == REASON_DIRECT_KEYWORD


# ---------------------------------------------------------------------------
# TestRouterAgentRule5 -- Ambiguous default -> RETRIEVE
# ---------------------------------------------------------------------------


class TestRouterAgentRule5:

    def test_no_signal_with_docs_returns_retrieve(self) -> None:
        router = make_router(doc_count=1)
        d = router.decide("u1", "revenue Q3")
        assert d.route == "RETRIEVE"
        assert d.reason == REASON_AMBIGUOUS_DEFAULT

    def test_short_query_no_signal_returns_retrieve(self) -> None:
        router = make_router(doc_count=1)
        d = router.decide("u1", "margins")
        assert d.route == "RETRIEVE"
        assert d.reason == REASON_AMBIGUOUS_DEFAULT

    def test_number_only_query_returns_retrieve(self) -> None:
        router = make_router(doc_count=1)
        d = router.decide("u1", "2024")
        assert d.route == "RETRIEVE"
        assert d.reason == REASON_AMBIGUOUS_DEFAULT

    def test_proper_noun_query_returns_retrieve(self) -> None:
        router = make_router(doc_count=1)
        d = router.decide("u1", "APAC revenue")
        assert d.route == "RETRIEVE"
        assert d.reason == REASON_AMBIGUOUS_DEFAULT

    def test_question_without_signal_returns_retrieve(self) -> None:
        router = make_router(doc_count=1)
        d = router.decide("u1", "did revenue grow last quarter?")
        assert d.route == "RETRIEVE"
        assert d.reason == REASON_AMBIGUOUS_DEFAULT

    def test_multiple_docs_ambiguous_still_retrieve(self) -> None:
        router = make_router(doc_count=5)
        d = router.decide("u1", "operating margin")
        assert d.route == "RETRIEVE"
        assert d.reason == REASON_AMBIGUOUS_DEFAULT


# ---------------------------------------------------------------------------
# TestRouterAgentEdgeCases
# ---------------------------------------------------------------------------


class TestRouterAgentEdgeCases:

    def test_empty_query_with_no_docs_returns_direct(self) -> None:
        router = make_router(doc_count=0)
        d = router.decide("u1", "")
        assert d.route == "DIRECT"
        assert d.reason == REASON_NO_DOCUMENTS

    def test_empty_query_with_docs_returns_retrieve(self) -> None:
        router = make_router(doc_count=1)
        d = router.decide("u1", "")
        assert d.route == "RETRIEVE"
        assert d.reason == REASON_AMBIGUOUS_DEFAULT

    def test_very_long_query_does_not_raise(self) -> None:
        router = make_router(doc_count=1)
        long_query = "what is the revenue? " * 200
        d = router.decide("u1", long_query)
        assert d.route in ("DIRECT", "RETRIEVE")

    def test_retrieve_keyword_beats_direct_keyword(self) -> None:
        # "summarise" (Rule 3) appears before "what is" (Rule 4)
        router = make_router(doc_count=1)
        d = router.decide("u1", "what is the summarise function?")
        # "summarise" is in the query -- Rule 3 fires
        assert d.reason == REASON_RETRIEVE_KEYWORD

    def test_filename_beats_retrieve_keyword(self) -> None:
        # Rule 2 is checked before Rule 3
        router = make_router(doc_count=1)
        d = router.decide("u1", "summarise report.pdf for me")
        assert d.reason == REASON_FILENAME_SIGNAL

    def test_filename_beats_direct_keyword(self) -> None:
        # Rule 2 is checked before Rule 4
        router = make_router(doc_count=1)
        d = router.decide("u1", "what is in data.csv?")
        assert d.reason == REASON_FILENAME_SIGNAL

    def test_decide_always_returns_route_decision(self) -> None:
        router = make_router(doc_count=1)
        for query in ["", "hello", "define PDF", "revenue", "summarise report.pdf"]:
            result = router.decide("u1", query)
            assert isinstance(result, RouteDecision)
            assert result.route in ("DIRECT", "RETRIEVE")


# ---------------------------------------------------------------------------
# TestRouterAgentFixtures -- 50-case parametrized suite (ADR-003)
# ---------------------------------------------------------------------------


ROUTING_FIXTURES: list[tuple[str, int, str, str]] = [
    # (query, doc_count, expected_route, expected_reason)

    # --- Rule 1: no documents ---
    ("hello", 0, "DIRECT", REASON_NO_DOCUMENTS),
    ("what is revenue?", 0, "DIRECT", REASON_NO_DOCUMENTS),
    ("summarise my data", 0, "DIRECT", REASON_NO_DOCUMENTS),
    ("report.pdf", 0, "DIRECT", REASON_NO_DOCUMENTS),
    ("", 0, "DIRECT", REASON_NO_DOCUMENTS),

    # --- Rule 2: filename signals ---
    ("open my_report.pdf", 1, "RETRIEVE", REASON_FILENAME_SIGNAL),
    ("analyse sales.csv", 1, "RETRIEVE", REASON_FILENAME_SIGNAL),
    ("what is in Q3.PDF?", 1, "RETRIEVE", REASON_FILENAME_SIGNAL),
    ("revenue in annual report.pdf", 1, "RETRIEVE", REASON_FILENAME_SIGNAL),
    ("check data.CSV", 1, "RETRIEVE", REASON_FILENAME_SIGNAL),

    # --- Rule 3: strong RETRIEVE keywords ---
    ("summarise the findings", 1, "RETRIEVE", REASON_RETRIEVE_KEYWORD),
    ("summarize the key points", 1, "RETRIEVE", REASON_RETRIEVE_KEYWORD),
    ("from my file, extract revenue", 1, "RETRIEVE", REASON_RETRIEVE_KEYWORD),
    ("from my document, what is the target?", 1, "RETRIEVE", REASON_RETRIEVE_KEYWORD),
    ("in my document, does it mention growth?", 1, "RETRIEVE", REASON_RETRIEVE_KEYWORD),
    ("what does the document say about Q4?", 1, "RETRIEVE", REASON_RETRIEVE_KEYWORD),
    ("according to the report, margins improved", 1, "RETRIEVE", REASON_RETRIEVE_KEYWORD),
    ("from the file, get the numbers", 1, "RETRIEVE", REASON_RETRIEVE_KEYWORD),
    ("in the document, find the table", 1, "RETRIEVE", REASON_RETRIEVE_KEYWORD),
    ("based on the document, what next?", 1, "RETRIEVE", REASON_RETRIEVE_KEYWORD),
    ("as per the report, revenue is up", 1, "RETRIEVE", REASON_RETRIEVE_KEYWORD),
    ("what do the documents say about risk?", 1, "RETRIEVE", REASON_RETRIEVE_KEYWORD),
    ("from the report, extract the figures", 1, "RETRIEVE", REASON_RETRIEVE_KEYWORD),
    ("in the report, is there a chart?", 1, "RETRIEVE", REASON_RETRIEVE_KEYWORD),
    ("per the document, growth was 20%", 1, "RETRIEVE", REASON_RETRIEVE_KEYWORD),

    # --- Rule 4: strong DIRECT keywords ---
    ("what is EBITDA?", 1, "DIRECT", REASON_DIRECT_KEYWORD),
    ("what are hedge funds?", 1, "DIRECT", REASON_DIRECT_KEYWORD),
    ("who is Jerome Powell?", 1, "DIRECT", REASON_DIRECT_KEYWORD),
    ("who was John Maynard Keynes?", 1, "DIRECT", REASON_DIRECT_KEYWORD),
    ("define amortization", 1, "DIRECT", REASON_DIRECT_KEYWORD),
    ("definition of net present value", 1, "DIRECT", REASON_DIRECT_KEYWORD),
    ("explain quantitative easing", 1, "DIRECT", REASON_DIRECT_KEYWORD),
    ("how does a bond work?", 1, "DIRECT", REASON_DIRECT_KEYWORD),
    ("how do interest rates affect stocks?", 1, "DIRECT", REASON_DIRECT_KEYWORD),
    ("tell me about the stock market", 1, "DIRECT", REASON_DIRECT_KEYWORD),
    ("describe the difference between stocks and bonds", 1, "DIRECT", REASON_DIRECT_KEYWORD),
    ("when was the SEC founded?", 1, "DIRECT", REASON_DIRECT_KEYWORD),
    ("where is the Federal Reserve headquartered?", 1, "DIRECT", REASON_DIRECT_KEYWORD),
    ("why is inflation a problem?", 1, "DIRECT", REASON_DIRECT_KEYWORD),
    ("why does money lose value over time?", 1, "DIRECT", REASON_DIRECT_KEYWORD),

    # --- Rule 5: ambiguous default -> RETRIEVE ---
    ("revenue", 1, "RETRIEVE", REASON_AMBIGUOUS_DEFAULT),
    ("Q3 growth", 1, "RETRIEVE", REASON_AMBIGUOUS_DEFAULT),
    ("operating margin 2024", 1, "RETRIEVE", REASON_AMBIGUOUS_DEFAULT),
    ("APAC performance", 1, "RETRIEVE", REASON_AMBIGUOUS_DEFAULT),
    ("total costs", 1, "RETRIEVE", REASON_AMBIGUOUS_DEFAULT),
    ("did we grow?", 1, "RETRIEVE", REASON_AMBIGUOUS_DEFAULT),
    ("cloud revenue", 1, "RETRIEVE", REASON_AMBIGUOUS_DEFAULT),
    ("forecast", 1, "RETRIEVE", REASON_AMBIGUOUS_DEFAULT),
    ("guidance Q4", 1, "RETRIEVE", REASON_AMBIGUOUS_DEFAULT),
    ("cash position", 1, "RETRIEVE", REASON_AMBIGUOUS_DEFAULT),
]


@pytest.mark.parametrize(
    "query,doc_count,expected_route,expected_reason",
    ROUTING_FIXTURES,
    ids=[f"{i:02d}_{r}_{reason[:8]}" for i, (_, __, r, reason) in enumerate(ROUTING_FIXTURES)],
)
def test_routing_fixtures(
    query: str,
    doc_count: int,
    expected_route: str,
    expected_reason: str,
) -> None:
    """50-case fixture suite verifying all routing rules per ADR-003."""
    router = make_router(doc_count=doc_count)
    decision = router.decide("u_fixture", query)
    assert decision.route == expected_route, (
        f"Query: {query!r}\n"
        f"Expected route: {expected_route}, got: {decision.route}\n"
        f"Reason: {decision.reason}"
    )
    assert decision.reason == expected_reason, (
        f"Query: {query!r}\n"
        f"Expected reason: {expected_reason}, got: {decision.reason}"
    )


# ---------------------------------------------------------------------------
# TestRoutedQueryService
# ---------------------------------------------------------------------------


class TestRoutedQueryService:

    def _make_service(
        self,
        route: str = "RETRIEVE",
        doc_count: int = 1,
    ) -> tuple[RoutedQueryService, MagicMock, MagicMock, MagicMock]:
        mock_router = MagicMock(spec=RouterAgent)
        mock_router.decide.return_value = RouteDecision(
            route=route,  # type: ignore[arg-type]
            reason=REASON_AMBIGUOUS_DEFAULT if route == "RETRIEVE" else REASON_NO_DOCUMENTS,
        )

        mock_query_service = MagicMock()
        mock_query_service.query.return_value = make_query_result(route="RETRIEVE")

        mock_llm = MagicMock()
        mock_llm.generate.return_value = MagicMock(
            content="General knowledge answer.",
            token_usage=TokenUsage.from_counts(100, 40),
            model="local",
            provider="local",
            latency_ms=5,
        )

        settings = make_mock_settings()

        service = RoutedQueryService(
            router=mock_router,
            query_service=mock_query_service,
            llm_provider=mock_llm,
            settings=settings,
        )
        return service, mock_router, mock_query_service, mock_llm

    def test_retrieve_path_calls_query_service(self) -> None:
        service, _, mock_qs, mock_llm = self._make_service(route="RETRIEVE")
        service.query("u1", "what is the revenue?")
        mock_qs.query.assert_called_once()
        mock_llm.generate.assert_not_called()

    def test_retrieve_path_passes_user_id(self) -> None:
        service, _, mock_qs, _ = self._make_service(route="RETRIEVE")
        service.query("u_tenant", "query text")
        call_kwargs = mock_qs.query.call_args.kwargs
        assert call_kwargs["user_id"] == "u_tenant"

    def test_retrieve_path_passes_query_text(self) -> None:
        service, _, mock_qs, _ = self._make_service(route="RETRIEVE")
        service.query("u1", "what is the operating margin?")
        call_kwargs = mock_qs.query.call_args.kwargs
        assert call_kwargs["query_text"] == "what is the operating margin?"

    def test_retrieve_path_passes_top_k_override(self) -> None:
        service, _, mock_qs, _ = self._make_service(route="RETRIEVE")
        service.query("u1", "query", top_k=3)
        call_kwargs = mock_qs.query.call_args.kwargs
        assert call_kwargs["top_k"] == 3

    def test_retrieve_path_returns_query_result(self) -> None:
        service, _, _, _ = self._make_service(route="RETRIEVE")
        result = service.query("u1", "query")
        assert isinstance(result, QueryResult)

    def test_direct_path_calls_llm_not_query_service(self) -> None:
        service, _, mock_qs, mock_llm = self._make_service(route="DIRECT")
        service.query("u1", "what is compound interest?")
        mock_qs.query.assert_not_called()
        mock_llm.generate.assert_called_once()

    def test_direct_path_returns_query_result(self) -> None:
        service, _, _, _ = self._make_service(route="DIRECT")
        result = service.query("u1", "what is compound interest?")
        assert isinstance(result, QueryResult)

    def test_direct_path_route_is_direct(self) -> None:
        service, _, _, _ = self._make_service(route="DIRECT")
        result = service.query("u1", "what is compound interest?")
        assert result.route == "DIRECT"

    def test_direct_path_sources_are_empty(self) -> None:
        service, _, _, _ = self._make_service(route="DIRECT")
        result = service.query("u1", "what is compound interest?")
        assert result.sources == []

    def test_direct_path_chunks_are_zero(self) -> None:
        service, _, _, _ = self._make_service(route="DIRECT")
        result = service.query("u1", "what is compound interest?")
        assert result.chunks_retrieved == 0
        assert result.chunks_used == 0

    def test_direct_path_has_answer(self) -> None:
        service, _, _, _ = self._make_service(route="DIRECT")
        result = service.query("u1", "what is compound interest?")
        assert result.answer == "General knowledge answer."

    def test_direct_path_has_token_usage(self) -> None:
        service, _, _, _ = self._make_service(route="DIRECT")
        result = service.query("u1", "what is compound interest?")
        assert result.token_usage.total_tokens > 0

    def test_direct_path_uses_direct_prompt(self) -> None:
        service, _, _, mock_llm = self._make_service(route="DIRECT")
        service.query("u1", "what is GDP?")
        call_args = mock_llm.generate.call_args
        messages = call_args.args[0] if call_args.args else call_args.kwargs.get("messages")
        system_content = next(m["content"] for m in messages if m["role"] == "system")
        # Must use DIRECT_SYSTEM_PROMPT, not SYSTEM_PROMPT
        assert system_content == DIRECT_SYSTEM_PROMPT
        assert "<CONTEXT>" not in messages[1]["content"]

    def test_router_called_with_correct_args(self) -> None:
        service, mock_router, _, _ = self._make_service(route="RETRIEVE")
        service.query("u_xyz", "some query text")
        mock_router.decide.assert_called_once_with(
            user_id="u_xyz",
            query_text="some query text",
        )

    def test_vector_store_error_propagates(self) -> None:
        service, mock_router, _, _ = self._make_service(route="RETRIEVE")
        mock_router.decide.side_effect = VectorStoreError("db down")
        with pytest.raises(VectorStoreError):
            service.query("u1", "query")

    def test_direct_path_user_id_in_result(self) -> None:
        service, _, _, _ = self._make_service(route="DIRECT")
        result = service.query("u_direct", "what is GDP?")
        assert result.user_id == "u_direct"

    def test_direct_path_query_preserved_in_result(self) -> None:
        service, _, _, _ = self._make_service(route="DIRECT")
        result = service.query("u1", "what is the capital of France?")
        assert result.query == "what is the capital of France?"


# ---------------------------------------------------------------------------
# TestBuildDirectMessages
# ---------------------------------------------------------------------------


class TestBuildDirectMessages:

    def test_returns_two_messages(self) -> None:
        messages = build_direct_messages("what is GDP?")
        assert len(messages) == 2

    def test_first_message_is_system(self) -> None:
        messages = build_direct_messages("what is GDP?")
        assert messages[0]["role"] == "system"

    def test_second_message_is_user(self) -> None:
        messages = build_direct_messages("what is GDP?")
        assert messages[1]["role"] == "user"

    def test_system_prompt_is_direct_prompt(self) -> None:
        messages = build_direct_messages("what is GDP?")
        assert messages[0]["content"] == DIRECT_SYSTEM_PROMPT

    def test_system_prompt_is_not_retrieve_prompt(self) -> None:
        messages = build_direct_messages("what is GDP?")
        assert messages[0]["content"] != SYSTEM_PROMPT

    def test_user_message_contains_query(self) -> None:
        messages = build_direct_messages("what is the capital of France?")
        assert "what is the capital of France?" in messages[1]["content"]

    def test_no_context_tags_in_user_message(self) -> None:
        messages = build_direct_messages("what is GDP?")
        assert "<CONTEXT>" not in messages[1]["content"]
        assert "</CONTEXT>" not in messages[1]["content"]

    def test_user_message_is_just_the_query(self) -> None:
        query = "explain compound interest"
        messages = build_direct_messages(query)
        assert messages[1]["content"] == query

    def test_direct_prompt_permits_general_knowledge(self) -> None:
        # DIRECT_SYSTEM_PROMPT must NOT contain the "ONLY" restriction
        assert "ONLY the information provided" not in DIRECT_SYSTEM_PROMPT

    def test_direct_prompt_is_non_empty(self) -> None:
        assert len(DIRECT_SYSTEM_PROMPT) > 50
