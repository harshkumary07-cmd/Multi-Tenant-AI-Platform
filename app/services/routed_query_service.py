"""
Routed query service -- orchestrates RouterAgent and QueryService.

This is the external interface that Module 9 (API layer) will wire to the
POST /query route handler. It replaces direct use of QueryService.

Dispatch logic:
    1. RouterAgent.decide(user_id, query_text) -> RouteDecision
    2a. RETRIEVE: delegate to QueryService.query() -- full RAG pipeline
    2b. DIRECT:   call LLM directly with a general-knowledge prompt,
                  no embedding, no ChromaDB retrieval

DIRECT path detail:
    When the router decides DIRECT, this service:
        - Builds a direct-answer prompt (no context block)
        - Calls llm_provider.generate() directly
        - Returns QueryResult with route="DIRECT", sources=[], chunks=0
    The LLM answers from general knowledge. Token usage is recorded.

QueryService is NOT modified by Module 7:
    QueryService.query() is called unchanged for RETRIEVE decisions.
    RoutedQueryService wraps it -- it does not extend or inherit from it.

Module 8 (Redis Cache) extension point:
    The caching layer will wrap RoutedQueryService, not QueryService.
    Cache lookup happens before decide() is called.
    Cache writes happen after RoutedQueryService.query() returns.

Dependency injection:
    RoutedQueryService accepts all dependencies as constructor arguments.
    Module 9 constructs it with real dependencies.
    Tests inject mocks for RouterAgent, QueryService, and LLMProvider.
"""

from app.agents.router_agent import RouterAgent
from app.config.settings import Settings
from app.logging.logger import get_logger
from app.logging.timing import LatencyTracker
from app.models.query_result import QueryResult
from app.rag.prompt_builder import build_direct_messages
from app.services.llm_service import LLMProvider
from app.services.query_service import QueryService

logger = get_logger(__name__)


class RoutedQueryService:
    """
    Coordinates routing and query execution.

    Holds RouterAgent, QueryService, and LLMProvider.
    Dispatches to the correct pipeline based on the routing decision.

    Args:
        router:        RouterAgent for DIRECT vs RETRIEVE decision.
        query_service: QueryService for the RETRIEVE pipeline.
        llm_provider:  LLMProvider for DIRECT answers (no retrieval).
        settings:      Application settings.
    """

    def __init__(
        self,
        router: RouterAgent,
        query_service: QueryService,
        llm_provider: LLMProvider,
        settings: Settings,
    ) -> None:
        self._router = router
        self._query_service = query_service
        self._llm = llm_provider
        self._settings = settings

    def query(
        self,
        user_id: str,
        query_text: str,
        top_k: int | None = None,
    ) -> QueryResult:
        """
        Route and execute a query.

        Asks the RouterAgent for a decision, then delegates to the
        appropriate pipeline. Always returns a QueryResult.

        Args:
            user_id:    Tenant identifier.
            query_text: The user's natural language query.
            top_k:      Optional top-k override for RETRIEVE path.
                        Ignored for DIRECT path.

        Returns:
            QueryResult with route="DIRECT" or route="RETRIEVE".

        Raises:
            VectorStoreError:    If count_documents() fails during routing,
                                 or if search_chunks() fails during retrieval.
            EmbeddingFailedError: If embed_single() fails during retrieval.
            LLMTimeoutError:     If the LLM times out on either path.
            LLMProviderError:    If the LLM returns an error on either path.
        """
        tracker = LatencyTracker()

        # ------------------------------------------------------------------
        # Step 1: Route the query
        # ------------------------------------------------------------------
        decision = self._router.decide(user_id=user_id, query_text=query_text)
        tracker.checkpoint("route")

        # ------------------------------------------------------------------
        # Step 2a: RETRIEVE path -- full RAG pipeline
        # ------------------------------------------------------------------
        if decision.is_retrieve:
            result = self._query_service.query(
                user_id=user_id,
                query_text=query_text,
                top_k=top_k,
            )
            tracker.checkpoint("retrieve_pipeline")

            logger.info(
                "routed query complete",
                extra={
                    "event": "ROUTED_QUERY_COMPLETE",
                    "user_id": user_id,
                    "route": "RETRIEVE",
                    "reason": decision.reason,
                    "chunks_retrieved": result.chunks_retrieved,
                    "chunks_used": result.chunks_used,
                    "has_answer": result.has_answer,
                    **tracker.to_log_fields(),
                },
            )
            return result

        # ------------------------------------------------------------------
        # Step 2b: DIRECT path -- LLM without retrieval
        # ------------------------------------------------------------------
        messages = build_direct_messages(query_text)
        tracker.checkpoint("build_direct_prompt")

        llm_response = self._llm.generate(messages)
        tracker.checkpoint("llm_generate")

        total_ms = tracker.total_ms()

        logger.info(
            "routed query complete",
            extra={
                "event": "ROUTED_QUERY_COMPLETE",
                "user_id": user_id,
                "route": "DIRECT",
                "reason": decision.reason,
                "prompt_tokens": llm_response.token_usage.prompt_tokens,
                "completion_tokens": llm_response.token_usage.completion_tokens,
                **tracker.to_log_fields(),
            },
        )

        return QueryResult(
            query=query_text,
            user_id=user_id,
            answer=llm_response.content,
            sources=[],
            route="DIRECT",
            chunks_retrieved=0,
            chunks_used=0,
            token_usage=llm_response.token_usage,
            latency_ms=total_ms,
        )
