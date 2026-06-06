"""
Business logic orchestration layer.

Services coordinate repositories, embedding models, and external APIs.
They work in domain models (app/models/), not HTTP schemas (app/schemas/).
Services never import from app.api.
"""
