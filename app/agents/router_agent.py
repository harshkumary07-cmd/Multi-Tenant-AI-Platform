"""
Router Agent -- deterministic DIRECT vs RETRIEVE routing.

Implements ADR-003: rule-based routing with no LLM call.
Every routing decision is deterministic, logged, and traceable.

Decision rules (evaluated in priority order):

    Rule 1 -- No documents uploaded (DIRECT)
        count_documents(user_id) == 0
        The user has no stored content to retrieve from.
        Reason: "no_documents"

    Rule 2 -- Filename signal in query (RETRIEVE)
        Query contains a pattern matching *.pdf or *.csv (case-insensitive).
        The user is explicitly referencing an uploaded file.
        Reason: "filename_signal"

    Rule 3 -- Strong RETRIEVE keyword (RETRIEVE)
        Query contains a term that implies the user wants content
        from their uploaded documents.
        Signals: summarise, summarize, from my, in my document, according to,
                 what does the document, in the report, from the file,
                 from the document, in my file, as per, per the
        Reason: "retrieve_keyword"

    Rule 4 -- Strong DIRECT keyword (DIRECT)
        Query contains a term that implies a general knowledge question
        with no connection to uploaded documents.
        Signals: what is, what are, who is, who was, define, definition of,
                 explain, how does, how do, tell me about, describe,
                 when was, when did, where is, where was, why is, why does
        Reason: "direct_keyword"

    Rule 5 -- Ambiguous with documents present (RETRIEVE, default)
        No strong signal detected. Documents are present.
        Default to RETRIEVE per ADR-003: correctness beats speed.
        A false RETRIEVE costs ~1s latency.
        A false DIRECT may contradict the user's own documents.
        Reason: "ambiguous_default"

Latency budget:
    Rule 1 involves one ChromaDB count query (~5-15ms).
    Rules 2-5 are pure string matching on the query text (<1ms).
    Total routing overhead: <20ms in the common case.

Tenant isolation:
    count_documents() is called with user_id -- the count is scoped
    to the requesting user's documents only. Another user's document
    count cannot influence routing decisions.
"""

import re
from dataclasses import dataclass
from typing import Literal

from app.logging.logger import get_logger
from app.logging.timing import elapsed_ms, start_timer
from app.repositories.chroma_repository import ChromaRepository

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Route decision type
# ---------------------------------------------------------------------------

RouteType = Literal["DIRECT", "RETRIEVE"]

REASON_NO_DOCUMENTS = "no_documents"
REASON_FILENAME_SIGNAL = "filename_signal"
REASON_RETRIEVE_KEYWORD = "retrieve_keyword"
REASON_DIRECT_KEYWORD = "direct_keyword"
REASON_AMBIGUOUS_DEFAULT = "ambiguous_default"


@dataclass(frozen=True)
class RouteDecision:
    """
    Result of a single routing decision.

    Immutable once produced. Logged on every query for full traceability.

    Attributes:
        route:  "DIRECT" or "RETRIEVE".
        reason: Machine-readable signal name that triggered the decision.
                One of: no_documents, filename_signal, retrieve_keyword,
                        direct_keyword, ambiguous_default.
    """

    route: RouteType
    reason: str

    @property
    def is_direct(self) -> bool:
        return self.route == "DIRECT"

    @property
    def is_retrieve(self) -> bool:
        return self.route == "RETRIEVE"


# ---------------------------------------------------------------------------
# Signal patterns (compiled once at module load)
# ---------------------------------------------------------------------------

# Rule 2: filename-like patterns (.pdf or .csv, optionally preceded by name chars)
_FILENAME_PATTERN = re.compile(
    r"\b[\w\-\. ]+\.(pdf|csv)\b",
    re.IGNORECASE,
)

# Rule 3: strong RETRIEVE signals
_RETRIEVE_KEYWORDS: list[str] = [
    "summarise",
    "summarize",
    "from my file",
    "from my document",
    "in my file",
    "in my document",
    "in the document",
    "in the report",
    "in the file",
    "from the file",
    "from the document",
    "from the report",
    "according to",
    "what does the document",
    "what does my document",
    "what does the file",
    "what does the report",
    "as per",
    "per the",
    "based on the document",
    "based on my document",
    "based on the file",
    "based on the report",
    "from my data",
    "in my data",
    "from the data",
    "what does it say",
    "what do the documents",
    "what do my documents",
]

# Rule 4: strong DIRECT signals
# These are prefixes/phrases strongly associated with general knowledge queries.
# NOTE: "tell me about" and "describe" are intentionally excluded here because
# they are also used for resume/document summary queries (handled by QueryService
# summary detection). When documents are present, Rule 5 defaults to RETRIEVE.
_DIRECT_KEYWORDS: list[str] = [
    "what is ",
    "what is\n",
    "what are ",
    "what are\n",
    "who is ",
    "who is\n",
    "who was ",
    "who was\n",
    "where is ",
    "where is\n",
    "where was ",
    "where was\n",
    "when was ",
    "when was\n",
    "when did ",
    "when did\n",
    "why is ",
    "why is\n",
    "why does ",
    "why does\n",
    "why do ",
    "how does ",
    "how does\n",
    "how do ",
    "how do\n",
    "how is ",
    "how is\n",
    "define ",
    "define\n",
    "definition of ",
    "explain ",
    "explain\n",
    "give me a definition",
    "what does it mean",
    "what does that mean",
    "in general,",
    "generally speaking",
    "historically,",
    "historically speaking",
]


def _normalise(query: str) -> str:
    """
    Return a lowercased, whitespace-normalised copy of the query.

    Used for keyword matching so that capitalisation and whitespace
    variations do not affect routing decisions.
    """
    return " ".join(query.lower().split())


# ---------------------------------------------------------------------------
# RouterAgent
# ---------------------------------------------------------------------------


class RouterAgent:
    """
    Deterministic rule-based Router Agent.

    Decides DIRECT or RETRIEVE for each query using five ordered rules.
    No LLM call is made. The decision is fully deterministic and traceable.

    Args:
        repository: ChromaRepository used for Rule 1 (count_documents).
    """

    def __init__(self, repository: ChromaRepository) -> None:
        self._repository = repository

    def decide(self, user_id: str, query_text: str) -> RouteDecision:
        """
        Apply the five routing rules and return a RouteDecision.

        Rules are evaluated in priority order. The first matching rule wins.

        Args:
            user_id:    Tenant identifier. Scopes the document count query.
            query_text: The user's natural language query.

        Returns:
            RouteDecision: The route ("DIRECT" or "RETRIEVE") and the
                           signal name that triggered it.

        Raises:
            VectorStoreError: If count_documents() fails (ChromaDB unavailable).
                              This propagates -- do not catch here.
                              The caller (RoutedQueryService) lets it propagate
                              to the route handler which returns 503.
        """
        start = start_timer()
        normalised = _normalise(query_text)

        # ------------------------------------------------------------------
        # Rule 1: no documents uploaded -> DIRECT
        # ------------------------------------------------------------------
        doc_count = self._repository.count_documents(user_id)
        if doc_count == 0:
            decision = RouteDecision(route="DIRECT", reason=REASON_NO_DOCUMENTS)
            self._log(decision, user_id, elapsed_ms(start), doc_count)
            return decision

        # ------------------------------------------------------------------
        # Rule 2: filename signal in query -> RETRIEVE
        # ------------------------------------------------------------------
        if _FILENAME_PATTERN.search(query_text):
            decision = RouteDecision(route="RETRIEVE", reason=REASON_FILENAME_SIGNAL)
            self._log(decision, user_id, elapsed_ms(start), doc_count)
            return decision

        # ------------------------------------------------------------------
        # Rule 3: strong RETRIEVE keyword -> RETRIEVE
        # ------------------------------------------------------------------
        for keyword in _RETRIEVE_KEYWORDS:
            if keyword in normalised:
                decision = RouteDecision(route="RETRIEVE", reason=REASON_RETRIEVE_KEYWORD)
                self._log(decision, user_id, elapsed_ms(start), doc_count)
                return decision

        # ------------------------------------------------------------------
        # Rule 4: strong DIRECT keyword -> DIRECT
        # ------------------------------------------------------------------
        for keyword in _DIRECT_KEYWORDS:
            if normalised.startswith(keyword) or f" {keyword}" in f" {normalised}":
                decision = RouteDecision(route="DIRECT", reason=REASON_DIRECT_KEYWORD)
                self._log(decision, user_id, elapsed_ms(start), doc_count)
                return decision

        # ------------------------------------------------------------------
        # Rule 5: ambiguous with documents present -> RETRIEVE (default)
        # ------------------------------------------------------------------
        decision = RouteDecision(route="RETRIEVE", reason=REASON_AMBIGUOUS_DEFAULT)
        self._log(decision, user_id, elapsed_ms(start), doc_count)
        return decision

    @staticmethod
    def _log(
        decision: RouteDecision,
        user_id: str,
        latency_ms: int,
        doc_count: int,
    ) -> None:
        logger.info(
            "route decision",
            extra={
                "event": "ROUTE_DECISION",
                "user_id": user_id,
                "route": decision.route,
                "reason": decision.reason,
                "doc_count": doc_count,
                "routing_latency_ms": latency_ms,
            },
        )
