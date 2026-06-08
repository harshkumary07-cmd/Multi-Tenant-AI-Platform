"""
RAG (Retrieval-Augmented Generation) pipeline components.

Modules (M5 adds parsers; M6 adds pipeline, context assembler, prompt builder):
    parsers/             -- file-type-specific text extraction (M5)
    pipeline.py          -- end-to-end retrieval orchestration (M6)
    context_assembler.py -- deduplicate, rank, and window retrieved chunks (M6)
    prompt_builder.py    -- construct system + context + user message list (M6)
"""
