"""
File-type-specific text extraction modules -- implemented in Module 5.

Modules:
    pdf_parser.py -- pdfplumber extraction with header/footer removal and cleaning
    csv_parser.py -- pandas parsing with column:value row serialisation

Each parser accepts raw bytes and returns a clean text string.
Parsers raise domain exceptions on failure:
    CorruptFileError    -- PDF cannot be opened or parsed
    CSVParseError       -- CSV encoding or delimiter error
    EmptyDocumentError  -- no usable text after parsing
"""
