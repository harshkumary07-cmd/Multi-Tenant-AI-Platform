"""
ChromaDB connection management and collection utilities (implemented in M4).

    client.py  -- singleton HttpClient, get_or_create_collection()
    tenant.py  -- collection name constants, tenant isolation helpers

Single 'documents' collection. Tenant isolation via metadata filter.
"""
