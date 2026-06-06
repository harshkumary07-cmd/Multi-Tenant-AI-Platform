"""
Routing intelligence layer (implemented in M7).

The Router Agent (router_agent.py) decides DIRECT vs RETRIEVE for each query.
Decision is deterministic and rule-based -- no LLM call required.
Agents import from services only; never from api/ or repositories/ directly.
"""
