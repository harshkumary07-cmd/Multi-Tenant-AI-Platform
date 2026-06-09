"""
Routing intelligence layer -- implemented in Module 7.

Modules:
    router_agent.py -- RouterAgent class with decide() method;
                       RouteDecision dataclass (route + reason);
                       five ordered deterministic routing rules.

The Router Agent decides DIRECT vs RETRIEVE for each query.
Decision is rule-based -- no LLM call required.
Agents import from repositories and services only;
never from app.api.
"""
