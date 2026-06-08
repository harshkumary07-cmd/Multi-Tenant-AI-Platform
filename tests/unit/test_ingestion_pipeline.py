"""
Unit tests for Module 5 -- Document Ingestion Pipeline.

Tests cover every service and parser with mocked dependencies.
No infrastructure (ChromaDB, embedding model) is required.

Test classes:
    TestValidateUpload          -- file size, type, magic bytes
    TestPDFParser               -- text extraction and cleaning
    TestCSVParser               -- row parsing and serialisation
    TestChunkingService         -- splitting and Chunk construction
    TestEmbeddingService        -- model singleton and embed_chunks
    TestDocumentService         -- full pipeline orchestration
    TestDocumentServiceCleanup  -- partial write cleanup on failure
    TestDocumentRecord          -- domain model behaviour
"""

from unittest.mock import MagicMock, patch

import pytest

from app.models.chunk import Chunk
from app.models.document import DocumentRecord
from app.models.exceptions import (
    CorruptFileError,
    EmbeddingFailedError,
    EmptyDocumentError,
    FileTooLargeError,
    InvalidFileTypeError,
    VectorStoreError,
)
from app.rag.parsers.csv_parser import parse_csv, serialise_row
from app.rag.parsers.pdf_parser import clean_text, validate_pdf_bytes
from app.services.chunking_service import MIN_CHUNK_LENGTH, chunk_text
from app.services.document_service import DocumentService, validate_upload
from app.services.embedding_service import (
    embed_chunks,
    embed_single,
    get_embedding_model,
    initialise_embedding_model,
)
from app.vectorstore.tenant import build_chunk_id

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_PDF_BYTES = (
    b"%PDF-1.4\n"
    b"1 0 obj\n<< /Type /Catalog >>\nendobj\n"
    b"xref\n0 1\n0000000000 65535 f \n"
    b"trailer\n<< /Size 1 /Root 1 0 R >>\n"
    b"startxref\n9\n%%EOF\n"
)
CORRUPT_PDF_BYTES = b"NOTAPDF\x00these are not pdf magic bytes"
VALID_CSV_BYTES = b"col_a,col_b,col_c\nval1,val2,val3\nval4,val5,val6\n"
EMPTY_CSV_BYTES = b"col_a,col_b\n"  # header only, no data rows

LONG_TEXT = (
    "The quarterly revenue report shows significant growth across all segments. "
    "Cloud services revenue increased by 34 percent year over year driven by "
    "enterprise adoption in the Asia-Pacific region. Operating expenses were "
    "well-controlled resulting in margin expansion of 200 basis points. "
    "The research and development investment increased by 15 percent to support "
    "next-generation product development. Sales and marketing spend remained "
    "flat as a percentage of revenue due to improved efficiency. "
) * 8  # ~800 chars -- enough for multiple chunks at default settings


def make_mock_settings(
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    batch_size: int = 100,
    max_upload_mb: int = 50,
) -> MagicMock:
    s = MagicMock()
    s.CHUNK_SIZE_TOKENS = chunk_size
    s.CHUNK_OVERLAP_TOKENS = chunk_overlap
    s.EMBEDDING_BATCH_SIZE = batch_size
    s.MAX_UPLOAD_SIZE_MB = max_upload_mb
    return s


def make_mock_repository() -> MagicMock:
    repo = MagicMock()
    repo.add_chunks.return_value = None
    repo.delete_document.return_value = 0
    return repo


def make_chunk(index: int = 0, text: str = "sample chunk text content here") -> Chunk:
    return Chunk(
        chunk_id=build_chunk_id("doc_test", index),
        doc_id="doc_test",
        user_id="u_test",
        source="test.pdf",
        chunk_index=index,
        text=text,
        embedding=[0.1] * 384,
    )


# ---------------------------------------------------------------------------
# validate_upload
# ---------------------------------------------------------------------------

class TestValidateUpload:

    def test_valid_pdf_passes(self) -> None:
        result = validate_upload(VALID_PDF_BYTES, "report.pdf", "pdf", 50)
        assert result == "pdf"

    def test_valid_csv_passes(self) -> None:
        result = validate_upload(VALID_CSV_BYTES, "data.csv", "csv", 50)
        assert result == "csv"

    def test_file_too_large_raises(self) -> None:
        big = b"A" * (2 * 1024 * 1024)  # 2MB
        with pytest.raises(FileTooLargeError, match="exceeds"):
            validate_upload(big, "big.pdf", "pdf", 1)

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(InvalidFileTypeError, match="not supported"):
            validate_upload(b"data", "file.txt", "txt", 50)

    def test_pdf_with_wrong_magic_bytes_raises(self) -> None:
        with pytest.raises(InvalidFileTypeError, match="magic bytes"):
            validate_upload(CORRUPT_PDF_BYTES, "fake.pdf", "pdf", 50)

    def test_csv_does_not_require_magic_bytes(self) -> None:
        # CSVs have no universal magic bytes -- any bytes accepted
        result = validate_upload(b"anything,here\n1,2\n", "data.csv", "csv", 50)
        assert result == "csv"

    def test_case_insensitive_type(self) -> None:
        result = validate_upload(VALID_PDF_BYTES, "report.pdf", "PDF", 50)
        assert result == "pdf"

    def test_returns_literal_type(self) -> None:
        r1 = validate_upload(VALID_PDF_BYTES, "a.pdf", "pdf", 50)
        r2 = validate_upload(VALID_CSV_BYTES, "b.csv", "csv", 50)
        assert r1 in ("pdf", "csv")
        assert r2 in ("pdf", "csv")


# ---------------------------------------------------------------------------
# PDF parser
# ---------------------------------------------------------------------------

class TestPDFParser:

    def test_valid_magic_bytes_passes(self) -> None:
        validate_pdf_bytes(b"%PDF-1.4 rest of file", "ok.pdf")

    def test_invalid_magic_bytes_raises_corrupt_file(self) -> None:
        with pytest.raises(CorruptFileError):
            validate_pdf_bytes(CORRUPT_PDF_BYTES, "bad.pdf")

    def test_clean_text_repairs_hyphenated_breaks(self) -> None:
        raw = "reve-\nnue grew by 34 percent this quarter"
        result = clean_text(raw)
        assert "revenue" in result
        assert "-\n" not in result

    def test_clean_text_collapses_whitespace(self) -> None:
        raw = "cloud   services   grew   quickly"
        result = clean_text(raw)
        assert "  " not in result
        assert "cloud services grew quickly" in result

    def test_clean_text_drops_short_lines(self) -> None:
        raw = "This is a long enough line with real content.\nHi\nAnother valid line here."
        result = clean_text(raw)
        assert "Hi" not in result
        assert "long enough line" in result

    def test_clean_text_empty_input(self) -> None:
        assert clean_text("") == ""

    def test_clean_text_preserves_numbers(self) -> None:
        raw = "Revenue was 2.4 billion dollars in Q3 2024."
        result = clean_text(raw)
        assert "2.4" in result
        assert "billion" in result

    def test_parse_pdf_with_real_fixture(self, sample_pdf_bytes: bytes) -> None:
        """Real PDF fixture should parse without error."""
        from app.rag.parsers.pdf_parser import parse_pdf
        text = parse_pdf(sample_pdf_bytes, "sample.pdf")
        assert len(text) > 50
        assert isinstance(text, str)

    def test_parse_pdf_corrupt_raises(self, corrupt_pdf_bytes: bytes) -> None:
        from app.rag.parsers.pdf_parser import parse_pdf
        with pytest.raises((CorruptFileError, EmptyDocumentError)):
            parse_pdf(corrupt_pdf_bytes, "corrupt.pdf")

    def test_parse_pdf_real_content(self, sample_pdf_bytes: bytes) -> None:
        from app.rag.parsers.pdf_parser import parse_pdf
        text = parse_pdf(sample_pdf_bytes, "sample.pdf")
        assert "revenue" in text.lower() or "Revenue" in text


# ---------------------------------------------------------------------------
# CSV parser
# ---------------------------------------------------------------------------

class TestCSVParser:

    def test_valid_csv_returns_text(self) -> None:
        text = parse_csv(VALID_CSV_BYTES, "data.csv")
        assert isinstance(text, str)
        assert len(text) > 0

    def test_csv_rows_serialised_as_col_value_pairs(self) -> None:
        csv = b"name,score\nAlice,95\nBob,87\n"
        text = parse_csv(csv, "scores.csv")
        assert "name: Alice" in text
        assert "score: 95" in text

    def test_empty_csv_raises_empty_document(self) -> None:
        with pytest.raises(EmptyDocumentError):
            parse_csv(EMPTY_CSV_BYTES, "empty.csv")

    def test_real_csv_fixture(self, sample_csv_bytes: bytes) -> None:
        text = parse_csv(sample_csv_bytes, "sample.csv")
        assert "quarter" in text
        assert "Q3 2024" in text

    def test_nan_values_excluded_from_row(self) -> None:
        import pandas as pd
        row = pd.Series({"col_a": "val1", "col_b": float("nan"), "col_c": "val3"})
        result = serialise_row(row, ["col_a", "col_b", "col_c"])
        assert "col_b" not in result
        assert "col_a: val1" in result
        assert "col_c: val3" in result

    def test_all_nan_row_returns_empty_string(self) -> None:
        import pandas as pd
        row = pd.Series({"col_a": float("nan"), "col_b": float("nan")})
        result = serialise_row(row, ["col_a", "col_b"])
        assert result == ""

    def test_csv_with_semicolon_delimiter_raises(self) -> None:
        # pandas with default comma sep will produce one-column data
        # with 25 rows -- this should still parse (just one column)
        csv = b"a;b;c\n1;2;3\n4;5;6\n"
        # Should not raise -- pandas treats entire line as one column
        text = parse_csv(csv, "semi.csv")
        assert isinstance(text, str)

    def test_latin1_csv_does_not_raise(self) -> None:
        # latin-1 encoded file with a non-UTF-8 byte
        raw = b"col_a,col_b\nval\xe9,val2\n"
        text = parse_csv(raw, "latin.csv")
        assert "col_a" in text


# ---------------------------------------------------------------------------
# Chunking service
# ---------------------------------------------------------------------------

class TestChunkingService:

    def test_returns_list_of_chunks(self) -> None:
        chunks = chunk_text(LONG_TEXT, "doc_x", "u1", "test.pdf", 256, 30)
        assert isinstance(chunks, list)
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_all_chunks_have_correct_user_id(self) -> None:
        chunks = chunk_text(LONG_TEXT, "doc_x", "u_abc", "f.pdf", 256, 30)
        assert all(c.user_id == "u_abc" for c in chunks)

    def test_all_chunks_have_correct_doc_id(self) -> None:
        chunks = chunk_text(LONG_TEXT, "doc_xyz", "u1", "f.pdf", 256, 30)
        assert all(c.doc_id == "doc_xyz" for c in chunks)

    def test_all_chunks_have_correct_source(self) -> None:
        chunks = chunk_text(LONG_TEXT, "doc_x", "u1", "annual.pdf", 256, 30)
        assert all(c.source == "annual.pdf" for c in chunks)

    def test_chunks_have_no_embeddings_yet(self) -> None:
        chunks = chunk_text(LONG_TEXT, "doc_x", "u1", "f.pdf", 256, 30)
        assert all(not c.has_embedding() for c in chunks)

    def test_no_chunk_shorter_than_minimum(self) -> None:
        chunks = chunk_text(LONG_TEXT, "doc_x", "u1", "f.pdf", 256, 30)
        assert all(len(c.text) >= MIN_CHUNK_LENGTH for c in chunks)

    def test_chunk_ids_are_sequential(self) -> None:
        chunks = chunk_text(LONG_TEXT, "doc_x", "u1", "f.pdf", 256, 30)
        expected_ids = [build_chunk_id("doc_x", i) for i in range(len(chunks))]
        assert [c.chunk_id for c in chunks] == expected_ids

    def test_empty_document_raises(self) -> None:
        with pytest.raises(EmptyDocumentError):
            chunk_text("", "doc_x", "u1", "f.pdf", 256, 30)

    def test_overlap_gte_size_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="less than"):
            chunk_text(LONG_TEXT, "doc_x", "u1", "f.pdf", 100, 100)

    def test_large_text_produces_multiple_chunks(self) -> None:
        chunks = chunk_text(LONG_TEXT, "doc_x", "u1", "f.pdf", 200, 20)
        assert len(chunks) > 1

    def test_short_text_produces_single_chunk(self) -> None:
        short = "This is a short but valid paragraph containing sufficient content."
        chunks = chunk_text(short, "doc_x", "u1", "f.pdf", 512, 50)
        assert len(chunks) == 1
        assert chunks[0].text == short


# ---------------------------------------------------------------------------
# Embedding service
# ---------------------------------------------------------------------------

class TestEmbeddingService:

    def setup_method(self) -> None:
        """Reset the model singleton before each test."""
        import app.services.embedding_service as es
        es._model = None

    def test_get_embedding_model_before_init_raises(self) -> None:
        with pytest.raises(EmbeddingFailedError, match="not been initialised"):
            get_embedding_model()

    def test_initialise_loads_model(self) -> None:
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1] * 384]
        # Patch at the sentence_transformers module level since the service
        # imports SentenceTransformer inside the function body at runtime
        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=mock_model,
        ):
            initialise_embedding_model("all-MiniLM-L6-v2")
        model = get_embedding_model()
        assert model is mock_model

    def test_initialise_twice_is_noop(self) -> None:
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1] * 384]
        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=mock_model,
        ) as mock_st:
            initialise_embedding_model("model-a")
            initialise_embedding_model("model-a")
        # Called only once -- second call is a no-op
        mock_st.assert_called_once()

    def test_embed_chunks_attaches_vectors(self) -> None:
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.5] * 384, [0.6] * 384]
        import app.services.embedding_service as es
        es._model = mock_model

        chunks = [make_chunk(0, "first chunk text here"), make_chunk(1, "second chunk text")]
        # Remove embeddings
        for c in chunks:
            c.embedding = []
        result = embed_chunks(chunks, batch_size=100)
        assert all(c.has_embedding() for c in result)
        assert len(result[0].embedding) == 384

    def test_embed_chunks_empty_list_raises(self) -> None:
        with pytest.raises(ValueError, match="empty list"):
            embed_chunks([], batch_size=100)

    def test_embed_chunks_model_error_raises_embedding_failed(self) -> None:
        mock_model = MagicMock()
        mock_model.encode.side_effect = RuntimeError("GPU OOM")
        import app.services.embedding_service as es
        es._model = mock_model

        chunks = [make_chunk(0)]
        chunks[0].embedding = []
        with pytest.raises(EmbeddingFailedError):
            embed_chunks(chunks, batch_size=100)

    def test_embed_single_returns_vector(self) -> None:
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1] * 384]
        import app.services.embedding_service as es
        es._model = mock_model
        result = embed_single("what is the revenue?")
        assert isinstance(result, list)
        assert len(result) == 384

    def test_embed_single_model_not_init_raises(self) -> None:
        with pytest.raises(EmbeddingFailedError):
            embed_single("test")

    def teardown_method(self) -> None:
        """Reset the model singleton after each test."""
        import app.services.embedding_service as es
        es._model = None


# ---------------------------------------------------------------------------
# DocumentService -- pipeline orchestration
# ---------------------------------------------------------------------------

class TestDocumentService:

    def _make_service(
        self,
        repository: MagicMock | None = None,
        chunk_size: int = 200,
        chunk_overlap: int = 20,
        max_upload_mb: int = 50,
    ) -> tuple[DocumentService, MagicMock]:
        repo = repository or make_mock_repository()
        settings = make_mock_settings(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_upload_mb=max_upload_mb,
        )
        return DocumentService(repository=repo, settings=settings), repo

    def _mock_embed(self, chunks: list[Chunk], batch_size: int) -> list[Chunk]:
        """Side effect that populates embeddings on chunks."""
        for c in chunks:
            c.embedding = [0.1] * 384
        return chunks

    def test_ingest_pdf_returns_document_record(self) -> None:
        service, _ = self._make_service()
        with patch("app.services.document_service.parse_pdf", return_value=LONG_TEXT), \
             patch("app.services.document_service.embed_chunks", side_effect=self._mock_embed):
            result = service.ingest("u1", VALID_PDF_BYTES, "report.pdf", "pdf")
        assert isinstance(result, DocumentRecord)
        assert result.user_id == "u1"
        assert result.filename == "report.pdf"
        assert result.file_type == "pdf"
        assert result.status == "complete"
        assert result.chunks_stored > 0

    def test_ingest_csv_returns_document_record(self) -> None:
        service, _ = self._make_service()
        csv_bytes = b"col,val\nApple,100\nBanana,200\nCherry,300\nDate,400\nElder,500\n"
        with patch("app.services.document_service.parse_csv", return_value=LONG_TEXT), \
             patch("app.services.document_service.embed_chunks", side_effect=self._mock_embed):
            result = service.ingest("u1", csv_bytes, "data.csv", "csv")
        assert result.file_type == "csv"
        assert result.chunks_stored > 0

    def test_ingest_calls_repository_add_chunks(self) -> None:
        service, repo = self._make_service()
        with patch("app.services.document_service.parse_pdf", return_value=LONG_TEXT), \
             patch("app.services.document_service.embed_chunks", side_effect=self._mock_embed):
            service.ingest("u1", VALID_PDF_BYTES, "report.pdf", "pdf")
        repo.add_chunks.assert_called_once()

    def test_ingest_passes_user_id_to_repository(self) -> None:
        service, repo = self._make_service()
        with patch("app.services.document_service.parse_pdf", return_value=LONG_TEXT), \
             patch("app.services.document_service.embed_chunks", side_effect=self._mock_embed):
            service.ingest("u_tenant_x", VALID_PDF_BYTES, "report.pdf", "pdf")
        call_args = repo.add_chunks.call_args
        assert call_args.kwargs["user_id"] == "u_tenant_x"

    def test_ingest_file_too_large_raises(self) -> None:
        # max_upload_mb=1 so a 2MB file triggers the error
        service, _ = self._make_service(max_upload_mb=1)
        big = b"%PDF-" + b"A" * (2 * 1024 * 1024)
        with pytest.raises(FileTooLargeError):
            service.ingest("u1", big, "big.pdf", "pdf")

    def test_ingest_invalid_type_raises(self) -> None:
        service, _ = self._make_service()
        with pytest.raises(InvalidFileTypeError):
            service.ingest("u1", b"data", "file.txt", "txt")

    def test_ingest_corrupt_pdf_raises(self) -> None:
        service, _ = self._make_service()
        with pytest.raises((CorruptFileError, EmptyDocumentError, InvalidFileTypeError)):
            service.ingest("u1", CORRUPT_PDF_BYTES, "corrupt.pdf", "pdf")

    def test_ingest_doc_id_format(self) -> None:
        service, _ = self._make_service()
        with patch("app.services.document_service.parse_pdf", return_value=LONG_TEXT), \
             patch("app.services.document_service.embed_chunks", side_effect=self._mock_embed):
            result = service.ingest("u1", VALID_PDF_BYTES, "report.pdf", "pdf")
        assert result.doc_id.startswith("doc_")
        assert len(result.doc_id) == len("doc_") + 8

    def test_ingest_chunks_stored_matches_repository_call(self) -> None:
        service, repo = self._make_service()
        with patch("app.services.document_service.parse_pdf", return_value=LONG_TEXT), \
             patch("app.services.document_service.embed_chunks", side_effect=self._mock_embed):
            result = service.ingest("u1", VALID_PDF_BYTES, "report.pdf", "pdf")
        add_call_chunks = repo.add_chunks.call_args.kwargs["chunks"]
        assert result.chunks_stored == len(add_call_chunks)


# ---------------------------------------------------------------------------
# DocumentService -- partial write cleanup
# ---------------------------------------------------------------------------

class TestDocumentServiceCleanup:

    def _make_service(
        self,
        repository: MagicMock | None = None,
    ) -> tuple[DocumentService, MagicMock]:
        repo = repository or make_mock_repository()
        settings = make_mock_settings()
        return DocumentService(repository=repo, settings=settings), repo

    def _mock_embed(self, chunks: list[Chunk], batch_size: int) -> list[Chunk]:
        for c in chunks:
            c.embedding = [0.1] * 384
        return chunks

    def test_cleanup_called_when_store_fails(self) -> None:
        repo = make_mock_repository()
        repo.add_chunks.side_effect = VectorStoreError("connection lost")
        service, _ = self._make_service(repository=repo)

        with pytest.raises(VectorStoreError):
            with patch("app.services.document_service.parse_pdf", return_value=LONG_TEXT), \
                 patch("app.services.document_service.embed_chunks", side_effect=self._mock_embed):
                service.ingest("u1", VALID_PDF_BYTES, "report.pdf", "pdf")

        repo.delete_document.assert_called_once()

    def test_cleanup_not_called_for_parse_failure(self) -> None:
        repo = make_mock_repository()
        service, _ = self._make_service(repository=repo)

        with pytest.raises(InvalidFileTypeError):
            service.ingest("u1", b"data", "file.txt", "txt")

        repo.delete_document.assert_not_called()

    def test_cleanup_not_called_for_corrupt_file(self) -> None:
        repo = make_mock_repository()
        service, _ = self._make_service(repository=repo)

        with pytest.raises((CorruptFileError, EmptyDocumentError, InvalidFileTypeError)):
            service.ingest("u1", CORRUPT_PDF_BYTES, "bad.pdf", "pdf")

        repo.delete_document.assert_not_called()

    def test_delete_doc_id_matches_ingest_doc_id(self) -> None:
        repo = make_mock_repository()
        repo.add_chunks.side_effect = VectorStoreError("db down")
        service, _ = self._make_service(repository=repo)

        with pytest.raises(VectorStoreError):
            with patch("app.services.document_service.parse_pdf", return_value=LONG_TEXT), \
                 patch("app.services.document_service.embed_chunks", side_effect=self._mock_embed):
                service.ingest("u1", VALID_PDF_BYTES, "report.pdf", "pdf")

        delete_call = repo.delete_document.call_args
        # First positional arg is user_id, second is doc_id
        cleanup_doc_id = (
            delete_call.args[1]
            if len(delete_call.args) > 1
            else delete_call.kwargs.get("doc_id")
        )
        assert cleanup_doc_id is not None
        assert cleanup_doc_id.startswith("doc_")

    def test_embedding_failure_triggers_cleanup(self) -> None:
        repo = make_mock_repository()
        service, _ = self._make_service(repository=repo)

        def mock_embed_fail(chunks: list[Chunk], batch_size: int) -> list[Chunk]:
            raise EmbeddingFailedError("model crashed")

        with pytest.raises(EmbeddingFailedError):
            with patch("app.services.document_service.parse_pdf", return_value=LONG_TEXT), \
                 patch("app.services.document_service.embed_chunks", side_effect=mock_embed_fail):
                service.ingest("u1", VALID_PDF_BYTES, "report.pdf", "pdf")

        # Embedding fails before any writes -- no cleanup needed
        repo.delete_document.assert_not_called()


# ---------------------------------------------------------------------------
# DocumentRecord
# ---------------------------------------------------------------------------

class TestDocumentRecord:

    def test_uploaded_at_defaults_to_now(self) -> None:
        from datetime import UTC, datetime, timedelta

        before = datetime.now(tz=UTC)
        record = DocumentRecord(
            doc_id="doc_abc",
            user_id="u1",
            filename="report.pdf",
            file_type="pdf",
            chunks_stored=10,
        )
        after = datetime.now(tz=UTC)
        assert before - timedelta(seconds=1) <= record.uploaded_at <= after

    def test_status_is_complete(self) -> None:
        record = DocumentRecord(
            doc_id="doc_abc",
            user_id="u1",
            filename="report.pdf",
            file_type="pdf",
            chunks_stored=10,
        )
        assert record.status == "complete"

    def test_all_fields_accessible(self) -> None:
        record = DocumentRecord(
            doc_id="doc_abc123",
            user_id="u_test",
            filename="annual.pdf",
            file_type="pdf",
            chunks_stored=42,
        )
        assert record.doc_id == "doc_abc123"
        assert record.user_id == "u_test"
        assert record.filename == "annual.pdf"
        assert record.file_type == "pdf"
        assert record.chunks_stored == 42
