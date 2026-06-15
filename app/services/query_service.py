"""
Query service -- RAG query pipeline orchestrator.

Coordinates the complete query pipeline:
    1. Embed the query text using the same model used during ingestion
    2. Retrieve the top-k most similar chunks from ChromaDB (tenant-isolated)
    3. Filter and assemble context (threshold, budget, deduplication)
    4. Build the LLM prompt
    5. Generate a response via the configured LLM provider
    6. Return a structured QueryResult

No-result handling:
    If zero chunks pass the confidence threshold, NoRelevantChunksError is
    raised by the context assembler. QueryService catches it and returns a
    QueryResult with answer=None and no_result_reason set. The LLM is NOT
    called -- it would otherwise produce a plausible-sounding but ungrounded
    answer from general knowledge.

Module 7 (Router Agent) extension point:
    QueryService is designed to be extended by the Router Agent. Module 7
    will wrap QueryService and decide -- before calling query() -- whether
    to retrieve documents or answer directly from LLM general knowledge.
    The route="RETRIEVE" field in QueryResult is already set for Module 7
    to inspect and potentially override.

Module 8 (Redis Cache) extension point:
    Module 8 will add cache lookup before embed_single() and cache write
    after QueryResult is produced. The cache key is
    query:{user_id}:{sha256(normalised_query)}. QueryService does not know
    about the cache -- the caching layer wraps it.

Dependency injection:
    QueryService receives ChromaRepository and LLMProvider as constructor
    arguments. Tests inject mocks. The route handler (Module 9) constructs
    QueryService with real dependencies.
"""

from app.config.settings import Settings
from app.logging.logger import get_logger
from app.logging.timing import LatencyTracker
from app.models.exceptions import NoRelevantChunksError
from app.models.query_result import QueryResult, TokenUsage
from app.rag.context_assembler import assemble_context
from app.rag.prompt_builder import build_messages
from app.repositories.chroma_repository import ChromaRepository
from app.services.embedding_service import embed_single
from app.services.llm_service import LLMProvider

logger = get_logger(__name__)


class QueryService:
    """
    Orchestrates the RAG query pipeline.

    Args:
        repository:   ChromaRepository for vector similarity search.
        llm_provider: LLMProvider implementation for answer generation.
        settings:     Application settings (threshold, top_k, etc.).
    """

    def __init__(
        self,
        repository: ChromaRepository,
        llm_provider: LLMProvider,
        settings: Settings,
    ) -> None:
        self._repository = repository
        self._llm = llm_provider
        self._settings = settings

    def query(
        self,
        user_id: str,
        query_text: str,
        top_k: int | None = None,
    ) -> QueryResult:
        """
        Execute the full RAG query pipeline for a user query.

        Args:
            user_id:    Tenant identifier. All retrieval is scoped to this user.
            query_text: The natural language query from the user.
            top_k:      Optional override for number of chunks to retrieve.
                        Falls back to settings.RETRIEVAL_TOP_K if None.

        Returns:
            QueryResult: Structured result with answer, sources, and metrics.
                         Always returns a QueryResult -- never raises on
                         no-result (returns QueryResult with answer=None).

        Raises:
            EmbeddingFailedError: If the embedding model raises an error.
            VectorStoreError:     If ChromaDB raises an error during search.
            LLMTimeoutError:      If the LLM provider times out.
            LLMProviderError:     If the LLM provider returns an error.
        """
        effective_top_k = (
            top_k
            if top_k is not None
            else self._settings.RETRIEVAL_TOP_K
        )

        query_lower = query_text.lower()

        summary_keywords = [
        "summary",
        "summarize",
        "summarise",
        "overview",
        "candidate profile",
        "candidate summary",
        "candidate overview",
        "resume summary",
        "complete profile",
        "full profile",
        "profile of the candidate",
        "tell me about",
        "describe the candidate",
        "complete details",
        "all details",
        "complete resume",
        "entire resume",
        "all information",
        "full information",
        ]

        is_summary_query = any(
            keyword in query_lower
            for keyword in summary_keywords
        )

        if is_summary_query:
            effective_top_k = max(effective_top_k, 7)
        tracker = LatencyTracker()

        logger.info(
            "query started",
            extra={
                "event": "QUERY_START",
                "user_id": user_id,
                "query_length": len(query_text),
                "top_k": effective_top_k,
            },
        )

        # ------------------------------------------------------------------
        # Stage 1: Embed the query
        # ------------------------------------------------------------------
        query_embedding = embed_single(query_text)
        tracker.checkpoint("embed_query")

        # ------------------------------------------------------------------
        # Stage 2: Retrieve similar chunks (tenant-isolated)
        # ------------------------------------------------------------------
        raw_chunks = self._repository.search_chunks(
            user_id=user_id,
            query_embedding=query_embedding,
            top_k=effective_top_k,
        )
        tracker.checkpoint("retrieve")
        chunks_retrieved = len(raw_chunks)

        logger.debug(
            "chunks retrieved",
            extra={
                "user_id": user_id,
                "chunks_retrieved": chunks_retrieved,
                "top_k": effective_top_k,
                "top_score": round(raw_chunks[0].score, 4) if raw_chunks else None,
            },
        )

        # ------------------------------------------------------------------
        # Stage 3: Filter, assemble context (handles NoRelevantChunksError)
        # ------------------------------------------------------------------
        try:
            assembled = assemble_context(
                chunks=raw_chunks,
                threshold=self._settings.RETRIEVAL_CONFIDENCE_THRESHOLD,
            )
            tracker.checkpoint("assemble_context")
        except NoRelevantChunksError as exc:
            tracker.checkpoint("assemble_context")
            total_ms = tracker.total_ms()

            logger.info(
                "query no result",
                extra={
                    "event": "QUERY_NO_RESULT",
                    "user_id": user_id,
                    "chunks_retrieved": chunks_retrieved,
                    "reason": "NO_RELEVANT_CHUNKS",
                    "total_latency_ms": total_ms,
                },
            )

            return QueryResult(
                query=query_text,
                user_id=user_id,
                answer=None,
                sources=[],
                route="RETRIEVE",
                chunks_retrieved=chunks_retrieved,
                chunks_used=0,
                token_usage=TokenUsage.zero(),
                latency_ms=total_ms,
                no_result_reason=str(exc.message),
            )

        # ------------------------------------------------------------------
        # Stage 4: Build prompt
        # ------------------------------------------------------------------
        messages = build_messages(
            context_text=assembled.context_text,
            query=query_text,
        )
        tracker.checkpoint("build_prompt")

        # ------------------------------------------------------------------
        # Stage 5: Generate LLM response
        # ------------------------------------------------------------------
        llm_response = self._llm.generate(messages)
        tracker.checkpoint("llm_generate")

        total_ms = tracker.total_ms()

        logger.info(
            "query complete",
            extra={
                "event": "QUERY_COMPLETE",
                "user_id": user_id,
                "route": "RETRIEVE",
                "chunks_retrieved": chunks_retrieved,
                "chunks_used": assembled.chunk_count,
                "sources_count": len(assembled.sources),
                "prompt_tokens": llm_response.token_usage.prompt_tokens,
                "completion_tokens": llm_response.token_usage.completion_tokens,
                "total_tokens": llm_response.token_usage.total_tokens,
                "llm_provider": llm_response.provider,
                "llm_model": llm_response.model,
                **tracker.to_log_fields(),
            },
        )

        return QueryResult(
            query=query_text,
            user_id=user_id,
            answer=llm_response.content,
            sources=assembled.sources,
            route="RETRIEVE",
            chunks_retrieved=chunks_retrieved,
            chunks_used=assembled.chunk_count,
            token_usage=llm_response.token_usage,
            latency_ms=total_ms,
        )
