"""
Pydantic transport contracts for HTTP request and response bodies.

Schemas validate input and shape output at the API boundary.
Route handlers receive schemas; they pass domain models to services.

Schemas added per module:
    M5: upload_request.py   -- POST /upload-doc form fields
    M5: upload_response.py  -- POST /upload-doc 201 response
    M6: query_request.py    -- POST /query request body
    M6: query_response.py   -- POST /query 200 response
"""
