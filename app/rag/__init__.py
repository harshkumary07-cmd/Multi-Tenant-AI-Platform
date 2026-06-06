"""
RAG (Retrieval-Augmented Generation) pipeline components.

Modules (added in M5 and M6):
    pipeline.py          -- end-to-end retrieval orchestration
    context_assembler.py -- deduplicate, rank, and window retrieved chunks
    prompt_builder.py    -- construct system + context + user message list
    parsers/             -- file-type-specific text extraction
"""
