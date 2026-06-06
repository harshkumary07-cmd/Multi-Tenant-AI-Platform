# Test Fixtures

Binary test files committed to the repository for deterministic test inputs.

## Files

### sample.pdf
A real PDF document with known text content used in Module 5+ for:
- Integration tests asserting specific chunk counts
- RAG retrieval tests asserting specific content is returned
- Upload pipeline smoke tests

**Replace this placeholder with a real PDF before Module 5.**
Requirements: minimum 5 pages, known text content, under 500KB.

### sample.csv
A real CSV document with known content used in Module 5+ for:
- CSV ingestion pipeline tests
- Row serialisation format assertions

**Replace this placeholder with a real CSV before Module 5.**
Requirements: minimum 20 rows, 3-5 columns, under 50KB.

### corrupt.pdf
A binary file with an invalid PDF header used in Module 5+ for:
- Testing CORRUPT_FILE error handling
- Verifying partial write cleanup on ingestion failure

**Replace this placeholder with a file whose first bytes are not `%PDF-`.**
