"""
Tests for POST /upload-doc.

Covers:
    - Authentication (401 on missing header)
    - Successful upload (201 with UploadResponse schema)
    - Invalid file type (400)
    - File too large (413)
    - Corrupt PDF (400)
    - Empty document (400)
    - Missing file field (422)
    - Response schema validation
    - VectorStore error during ingestion (503)

All service calls are mocked.
"""

import io
from datetime import UTC, datetime
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.config.dependencies import get_document_service
from app.models.document import DocumentRecord
from app.models.exceptions import (
    CorruptFileError,
    EmptyDocumentError,
    FileTooLargeError,
    InvalidFileTypeError,
    VectorStoreError,
)
from main import app


def _make_document_record(
    user_id: str = "u_test",
    filename: str = "report.pdf",
    chunks_stored: int = 42,
) -> DocumentRecord:
    return DocumentRecord(
        doc_id="doc_abc123",
        user_id=user_id,
        filename=filename,
        file_type="pdf",
        chunks_stored=chunks_stored,
        uploaded_at=datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC),
    )


def _mock_service(record: DocumentRecord) -> MagicMock:
    mock = MagicMock()
    mock.ingest.return_value = record
    return mock


def _pdf_payload(content: bytes = b"%PDF-1.4 minimal content here for testing purposes") -> dict:
    return {
        "file": ("report.pdf", io.BytesIO(content), "application/pdf"),
    }


class TestUploadRouteAuthentication:

    def test_missing_header_returns_401(self, client: TestClient) -> None:
        response = client.post(
            "/upload-doc",
            data={"file_type": "pdf"},
            files={"file": ("r.pdf", b"%PDF-test", "application/pdf")},
        )
        assert response.status_code == 401

    def test_missing_header_error_code(self, client: TestClient) -> None:
        response = client.post(
            "/upload-doc",
            data={"file_type": "pdf"},
            files={"file": ("r.pdf", b"%PDF-test", "application/pdf")},
        )
        assert response.json()["error_code"] == "UNAUTHORIZED"


class TestUploadRouteHappyPath:

    def test_successful_upload_returns_201(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        mock_svc = _mock_service(_make_document_record())
        app.dependency_overrides[get_document_service] = lambda: mock_svc
        try:
            response = client.post(
                "/upload-doc",
                data={"file_type": "pdf"},
                files={"file": ("report.pdf", b"%PDF-valid", "application/pdf")},
                headers=auth_headers,
            )
            assert response.status_code == 201
        finally:
            app.dependency_overrides.clear()

    def test_response_contains_document_id(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        mock_svc = _mock_service(_make_document_record())
        app.dependency_overrides[get_document_service] = lambda: mock_svc
        try:
            response = client.post(
                "/upload-doc",
                data={"file_type": "pdf"},
                files={"file": ("r.pdf", b"%PDF-1", "application/pdf")},
                headers=auth_headers,
            )
            assert "document_id" in response.json()
        finally:
            app.dependency_overrides.clear()

    def test_response_contains_chunks_stored(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        mock_svc = _mock_service(_make_document_record(chunks_stored=42))
        app.dependency_overrides[get_document_service] = lambda: mock_svc
        try:
            response = client.post(
                "/upload-doc",
                data={"file_type": "pdf"},
                files={"file": ("r.pdf", b"%PDF-1", "application/pdf")},
                headers=auth_headers,
            )
            assert response.json()["chunks_stored"] == 42
        finally:
            app.dependency_overrides.clear()

    def test_response_status_is_complete(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        mock_svc = _mock_service(_make_document_record())
        app.dependency_overrides[get_document_service] = lambda: mock_svc
        try:
            response = client.post(
                "/upload-doc",
                data={"file_type": "pdf"},
                files={"file": ("r.pdf", b"%PDF-1", "application/pdf")},
                headers=auth_headers,
            )
            assert response.json()["status"] == "complete"
        finally:
            app.dependency_overrides.clear()

    def test_csv_upload_accepted(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        record = _make_document_record(filename="data.csv")
        record_csv = DocumentRecord(
            doc_id=record.doc_id,
            user_id=record.user_id,
            filename="data.csv",
            file_type="csv",
            chunks_stored=record.chunks_stored,
        )
        mock_svc = MagicMock()
        mock_svc.ingest.return_value = record_csv
        app.dependency_overrides[get_document_service] = lambda: mock_svc
        try:
            response = client.post(
                "/upload-doc",
                data={"file_type": "csv"},
                files={"file": ("data.csv", b"col1,col2\nval1,val2", "text/csv")},
                headers=auth_headers,
            )
            assert response.status_code == 201
        finally:
            app.dependency_overrides.clear()


class TestUploadRouteErrors:

    def test_invalid_file_type_returns_400(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        mock_svc = MagicMock()
        mock_svc.ingest.side_effect = InvalidFileTypeError("unsupported type")
        app.dependency_overrides[get_document_service] = lambda: mock_svc
        try:
            response = client.post(
                "/upload-doc",
                data={"file_type": "pdf"},
                files={"file": ("r.pdf", b"not a pdf", "application/pdf")},
                headers=auth_headers,
            )
            assert response.status_code == 400
        finally:
            app.dependency_overrides.clear()

    def test_file_too_large_returns_413(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        mock_svc = MagicMock()
        mock_svc.ingest.side_effect = FileTooLargeError("file too large")
        app.dependency_overrides[get_document_service] = lambda: mock_svc
        try:
            response = client.post(
                "/upload-doc",
                data={"file_type": "pdf"},
                files={"file": ("big.pdf", b"%PDF-big", "application/pdf")},
                headers=auth_headers,
            )
            assert response.status_code == 413
        finally:
            app.dependency_overrides.clear()

    def test_corrupt_file_returns_400(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        mock_svc = MagicMock()
        mock_svc.ingest.side_effect = CorruptFileError("corrupt")
        app.dependency_overrides[get_document_service] = lambda: mock_svc
        try:
            response = client.post(
                "/upload-doc",
                data={"file_type": "pdf"},
                files={"file": ("bad.pdf", b"%PDF-corrupt", "application/pdf")},
                headers=auth_headers,
            )
            assert response.status_code == 400
        finally:
            app.dependency_overrides.clear()

    def test_vector_store_error_returns_503(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        mock_svc = MagicMock()
        mock_svc.ingest.side_effect = VectorStoreError("db down")
        app.dependency_overrides[get_document_service] = lambda: mock_svc
        try:
            response = client.post(
                "/upload-doc",
                data={"file_type": "pdf"},
                files={"file": ("r.pdf", b"%PDF-1", "application/pdf")},
                headers=auth_headers,
            )
            assert response.status_code == 503
        finally:
            app.dependency_overrides.clear()

    def test_empty_document_returns_400(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        mock_svc = MagicMock()
        mock_svc.ingest.side_effect = EmptyDocumentError("no text")
        app.dependency_overrides[get_document_service] = lambda: mock_svc
        try:
            response = client.post(
                "/upload-doc",
                data={"file_type": "pdf"},
                files={"file": ("empty.pdf", b"%PDF-empty", "application/pdf")},
                headers=auth_headers,
            )
            assert response.status_code == 400
        finally:
            app.dependency_overrides.clear()

    def test_missing_file_field_returns_422(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        from app.config.dependencies import get_document_service
        mock_svc = MagicMock()
        app.dependency_overrides[get_document_service] = lambda: mock_svc
        try:
            response = client.post(
                "/upload-doc",
                data={"file_type": "pdf"},
                headers=auth_headers,
            )
            assert response.status_code == 422
        finally:
            app.dependency_overrides.clear()

    def test_error_response_has_error_code(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        mock_svc = MagicMock()
        mock_svc.ingest.side_effect = InvalidFileTypeError("bad type")
        app.dependency_overrides[get_document_service] = lambda: mock_svc
        try:
            response = client.post(
                "/upload-doc",
                data={"file_type": "pdf"},
                files={"file": ("r.pdf", b"bad", "application/pdf")},
                headers=auth_headers,
            )
            body = response.json()
            assert "error_code" in body
            assert "message" in body
        finally:
            app.dependency_overrides.clear()
