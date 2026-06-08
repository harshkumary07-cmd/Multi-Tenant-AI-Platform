"""
PDF text extraction using pdfplumber.

Extracts and cleans text from PDF files page by page. Returns a single
clean text string suitable for the chunking service.

Why pdfplumber over pypdf/pypdf2:
    pdfplumber preserves layout coordinates (x, y position of each
    character on the page). This enables position-based header and footer
    detection: text near the top or bottom of a page (within 50pt of the
    edge) is treated as a running header/footer and removed.
    pypdf2/pypdf extract text without layout information, producing
    noisier output that includes page numbers, section headers, and
    decorative separators as part of the main text.

Cleaning operations applied to every page:
    1. Remove text above PAGE_HEADER_THRESHOLD (top 50pt) -- headers
    2. Remove text below PAGE_FOOTER_THRESHOLD (bottom 50pt) -- footers
    3. Repair hyphenated line breaks: "reve-\nnue" -> "revenue"
    4. Collapse multiple consecutive whitespace to single space
    5. Strip leading/trailing whitespace per line
    6. Drop lines shorter than MIN_LINE_LENGTH (10 chars) after stripping

Pages with fewer than MIN_PAGE_CHARS (50 chars) after cleaning are
skipped entirely. This handles scanned pages that produce near-empty
text (pdfplumber cannot OCR -- it can only extract embedded text).
"""

import io
import re

import pdfplumber

from app.logging.logger import get_logger
from app.models.exceptions import CorruptFileError, EmptyDocumentError

logger = get_logger(__name__)

# Characters from the top/bottom of a page that are treated as headers/footers.
# 50pt = ~18mm, which captures most standard running headers and footers.
PAGE_HEADER_THRESHOLD = 50.0
PAGE_FOOTER_THRESHOLD = 50.0

# Minimum characters on a page after cleaning for it to be included.
MIN_PAGE_CHARS = 50

# Minimum characters on a single line for it to be included.
MIN_LINE_LENGTH = 10

# PDF magic bytes -- first 5 bytes of any valid PDF file.
PDF_MAGIC = b"%PDF-"


def validate_pdf_bytes(file_bytes: bytes, filename: str) -> None:
    """
    Validate that the file bytes begin with the PDF magic bytes.

    Args:
        file_bytes: Raw file bytes.
        filename:   Original filename for error messages.

    Raises:
        CorruptFileError: If the file does not begin with "%PDF-".
    """
    if not file_bytes.startswith(PDF_MAGIC):
        raise CorruptFileError(
            f"'{filename}' does not appear to be a valid PDF file. "
            f"Expected file to start with '%PDF-' but got "
            f"'{file_bytes[:8]!r}'. "
            "Re-export the file from its source application and try again."
        )


def extract_page_text(page: pdfplumber.page.Page) -> str:
    """
    Extract text from a single PDF page, filtering header/footer regions.

    Uses pdfplumber's word-level extraction with bounding boxes to
    identify and exclude text in the header and footer zones.

    Falls back to full-page text extraction if word-level extraction
    returns no words (some PDFs use non-standard encoding).

    Args:
        page: A pdfplumber Page object.

    Returns:
        str: Cleaned text from the page body, or empty string if the
             page contains no extractable text.
    """
    page_height = float(page.height or 792.0)
    footer_y = page_height - PAGE_FOOTER_THRESHOLD

    # Try word-level extraction with position filtering first
    try:
        words = page.extract_words(
            x_tolerance=3,
            y_tolerance=3,
            keep_blank_chars=False,
        )
        if words:
            body_words = [
                w["text"]
                for w in words
                if float(w.get("top", 0)) > PAGE_HEADER_THRESHOLD
                and float(w.get("top", 0)) < footer_y
            ]
            if body_words:
                return " ".join(body_words)
    except Exception:
        pass

    # Fallback: full page text extraction without position filtering
    return page.extract_text() or ""


def clean_text(raw_text: str) -> str:
    """
    Apply text cleaning operations to extracted PDF text.

    Operations (in order):
        1. Repair soft hyphens at line boundaries: "reve-\nnu" -> "revenue"
        2. Collapse multiple whitespace characters to a single space
        3. Strip each line, drop lines shorter than MIN_LINE_LENGTH
        4. Join surviving lines with a single space

    Args:
        raw_text: Raw text extracted from a PDF page or the full document.

    Returns:
        str: Cleaned text with noise removed.
    """
    if not raw_text:
        return ""

    # Repair hyphenated line breaks
    # Pattern: word-ending hyphen followed by newline and word continuation
    text = re.sub(r"-\n([a-zA-Z])", r"\1", raw_text)

    # Collapse multiple whitespace to single space (preserves \n between lines)
    text = re.sub(r"[ \t]+", " ", text)

    # Process line by line: strip and filter short lines
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if len(stripped) >= MIN_LINE_LENGTH:
            lines.append(stripped)

    return " ".join(lines)


def parse_pdf(file_bytes: bytes, filename: str) -> str:
    """
    Extract and clean all text from a PDF file.

    Entry point for the ingestion pipeline. Validates the file magic
    bytes, opens the PDF with pdfplumber, extracts and cleans text from
    each page, and returns the combined document text.

    Args:
        file_bytes: Raw bytes of the uploaded PDF file.
        filename:   Original filename, used in error messages and logging.

    Returns:
        str: All extracted text, pages joined with double newlines.
             The returned string is suitable for the chunking service.

    Raises:
        CorruptFileError:  If the file does not begin with PDF magic bytes,
                           or if pdfplumber raises any exception during parsing.
        EmptyDocumentError: If no usable text is found after cleaning all pages.
    """
    validate_pdf_bytes(file_bytes, filename)

    page_texts: list[str] = []
    skipped_pages = 0

    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            total_pages = len(pdf.pages)
            logger.debug(
                "pdf opened",
                extra={"source_file": filename, "total_pages": total_pages},
            )

            for page_num, page in enumerate(pdf.pages, start=1):
                raw = extract_page_text(page)
                cleaned = clean_text(raw)

                if len(cleaned) < MIN_PAGE_CHARS:
                    skipped_pages += 1
                    logger.debug(
                        "page skipped (insufficient text)",
                        extra={
                            "source_file": filename,
                            "page": page_num,
                            "chars_after_clean": len(cleaned),
                        },
                    )
                    continue

                page_texts.append(cleaned)

    except (CorruptFileError, EmptyDocumentError):
        raise
    except Exception as exc:
        raise CorruptFileError(
            f"Failed to parse '{filename}': {exc}. "
            "The file may be password-protected, corrupt, or in an "
            "unsupported PDF variant. Re-export and try again."
        ) from exc

    if not page_texts:
        raise EmptyDocumentError(
            f"'{filename}' contains no extractable text after cleaning "
            f"({skipped_pages} pages skipped). "
            "The PDF may consist entirely of scanned images. "
            "pdfplumber cannot OCR scanned content -- "
            "use a PDF with embedded text."
        )

    full_text = "\n\n".join(page_texts)

    logger.info(
        "pdf parsed",
        extra={
            "source_file": filename,
            "pages_extracted": len(page_texts),
            "pages_skipped": skipped_pages,
            "total_chars": len(full_text),
        },
    )

    return full_text
