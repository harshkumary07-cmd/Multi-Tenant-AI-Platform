"""
File-type-specific text extraction modules (added in M5).

    pdf_parser.py -- pdfplumber extraction with header/footer removal
    csv_parser.py -- pandas parsing with row serialisation

Each parser accepts raw bytes and returns clean text.
Parsers raise domain exceptions on failure.
"""
