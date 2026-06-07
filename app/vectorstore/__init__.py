"""
ChromaDB connection management and collection utilities -- implemented in Module 4.

Modules:
    client.py  -- singleton HttpClient, initialise_chroma(), close_chroma_client()
    tenant.py  -- metadata field name constants, ChunkMetadata TypedDict,
                  build_chunk_id(), build_chunk_metadata()

Single 'documents' collection. Tenant isolation via metadata filter:
    where={"user_id": {"$eq": user_id}} on every read operation.
"""
