"""
Data access layer -- all I/O with external systems.

One class per external system: ChromaDB, Redis, log store.
Repositories contain no business logic.
Every ChromaDB method requires user_id as a mandatory parameter.
"""
