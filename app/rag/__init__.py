"""
RAG (Retrieval-Augmented Generation) pipeline components.

Modules:
    parsers/             -- file-type-specific text extraction (M5)
    context_assembler.py -- filter chunks, enforce budget, deduplicate sources (M6)
    prompt_builder.py    -- construct system + context + user message list (M6)
    pipeline.py          -- end-to-end retrieval orchestration (future)
"""
