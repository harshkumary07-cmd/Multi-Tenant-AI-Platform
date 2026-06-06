# ADR-001: Single ChromaDB collection with metadata filtering

**Date:** 2024-01-15
**Status:** Accepted

---

## Context

The platform stores document embeddings for multiple users (tenants) in ChromaDB.
Design question: should each user have a separate collection, or should all users
share one collection with per-user metadata filtering?

## Decision

Use a **single collection named `documents`** for all tenants.
Tenant isolation is enforced via a mandatory `where={"user_id": user_id}` filter
on every ChromaDB query operation.

## Alternatives considered

| Option | Pro | Con | Decision |
|---|---|---|---|
| Per-tenant collection (`u1_documents`) | No shared data structure | N users = N collections to manage, migrate, back up | Rejected |
| Separate ChromaDB instances | Strongest isolation | Operationally prohibitive at scale | Rejected |
| Single collection + metadata filter | Scales to any user count | Isolation depends on filter correctness | **Accepted** |

## Consequences

**Becomes easier:**
- One collection to manage, back up, and migrate regardless of user count
- Scaling: adding users has no impact on collection count

**Becomes harder:**
- Isolation correctness depends on the `where` filter being applied on every query
- Mitigation: filter enforced at method signature level (mandatory `user_id` param)
- Mitigation: tenant isolation test suite as a hard CI merge gate

## References

- Architecture Handbook, Section 3: Multi-Tenancy
- Implementation Blueprint, Module 4: Vector Database Layer
