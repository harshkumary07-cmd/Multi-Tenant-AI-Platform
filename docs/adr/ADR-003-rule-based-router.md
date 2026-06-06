# ADR-003: Rule-based Router Agent (not LLM-based)

**Date:** 2024-01-15
**Status:** Accepted

---

## Context

For each query, the system must decide DIRECT (general knowledge) or RETRIEVE
(search user's documents). This can be done by keyword/intent heuristics or
by asking an LLM to classify the query.

## Decision

Use a **deterministic, rule-based Router Agent** with regex patterns and
keyword signals. No LLM call is made for the routing decision.

## Alternatives considered

| Option | Latency | Cost | Determinism | Decision |
|---|---|---|---|---|
| LLM classification | +300-800ms | ~$0.001/query | Non-deterministic | Rejected |
| Fine-tuned classifier | +50-100ms | model hosting | Deterministic | Rejected (over-engineering) |
| Rule-based heuristics | +5ms | $0.000 | Fully deterministic | **Accepted** |

## Decision tree (summary)

1. No documents uploaded -> DIRECT
2. Filename mentioned in query -> RETRIEVE
3. Strong RETRIEVE signals (summarise / from my file) -> RETRIEVE
4. Strong DIRECT signals (what is / define / who is) -> DIRECT
5. Ambiguous with documents present -> RETRIEVE (default)

**Ambiguous default rationale:** A false RETRIEVE costs ~1s of extra latency.
A false DIRECT gives a general-knowledge answer that may contradict the user's
own documents. Correctness beats speed.

## Consequences

**Becomes easier:**
- Full traceability: every routing decision has a logged reason
- Zero additional cost per query
- 50-case fixture test suite validates all branches

**Phase 2 upgrade path:**
If misclassification rate is measurably impacting users, replace heuristics
with a lightweight classifier. The `decide(query, user_id) -> RouteDecision`
interface does not change.

## References

- Architecture Handbook, Section 8: Router Agent
- Implementation Blueprint, Module 7: Router Agent
