"""
CSV text extraction using pandas.

Reads a CSV file and serialises each row into a natural-language-like
string suitable for embedding. Returns a single text string where each
row becomes one line.

Why row serialisation instead of raw CSV text:
    Embedding models are trained on natural language, not spreadsheet
    syntax. A row like {"quarter": "Q3", "revenue": 500000} is better
    represented as "quarter: Q3 | revenue: 500000" than as the raw
    CSV line "Q3,500000". The serialised form gives the embedding model
    enough context to understand what each value means.

Row serialisation format:
    "col1: val1 | col2: val2 | col3: val3"

    NaN values are excluded from the serialised string.
    All values are cast to string before serialisation.
    Rows where every column is NaN are skipped entirely.

Encoding detection:
    Tries UTF-8 first, then falls back to latin-1.
    latin-1 is a strict superset of ASCII and will decode any byte
    sequence without raising a UnicodeDecodeError (values may be
    garbled but parsing will succeed).
"""

import io

import pandas as pd

from app.logging.logger import get_logger
from app.models.exceptions import CSVParseError, EmptyDocumentError

logger = get_logger(__name__)

# Separates column:value pairs within a single serialised row.
ROW_SEPARATOR = " | "

# Minimum number of non-empty rows required for the document to be accepted.
MIN_ROW_COUNT = 1


def _try_parse(file_bytes: bytes, encoding: str) -> pd.DataFrame:
    """
    Attempt to parse CSV bytes with the given encoding.

    Args:
        file_bytes: Raw CSV file bytes.
        encoding:   Encoding name to try (e.g. "utf-8", "latin-1").

    Returns:
        pd.DataFrame: Parsed dataframe.

    Raises:
        Exception: Any pandas parsing exception.
    """
    return pd.read_csv(
        io.BytesIO(file_bytes),
        encoding=encoding,
        dtype=str,          # read all columns as strings to avoid type errors
        keep_default_na=True,
        na_values=["", "NA", "N/A", "null", "NULL", "none", "None"],
    )


def serialise_row(row: pd.Series, columns: list[str]) -> str:
    """
    Serialise a single DataFrame row to a natural-language-like string.

    Only includes columns where the value is not NaN.

    Args:
        row:     A pandas Series representing one CSV row.
        columns: List of column names in order.

    Returns:
        str: "col1: val1 | col2: val2 | ..." with NaN columns excluded.
             Empty string if all values are NaN.
    """
    parts: list[str] = []
    for col in columns:
        value = row.get(col)
        if pd.isna(value):
            continue
        parts.append(f"{col}: {str(value).strip()}")
    return ROW_SEPARATOR.join(parts)


def parse_csv(file_bytes: bytes, filename: str) -> str:
    """
    Parse a CSV file and serialise all rows to text.

    Attempts UTF-8 encoding first, falls back to latin-1. Raises
    CSVParseError if both encodings fail or if the file has no columns.

    Args:
        file_bytes: Raw bytes of the uploaded CSV file.
        filename:   Original filename, used in error messages and logging.

    Returns:
        str: All serialised rows joined by newlines. Each line corresponds
             to one non-empty CSV row. Suitable for the chunking service.

    Raises:
        CSVParseError:     If pandas cannot parse the file with either encoding,
                           or if the CSV has zero columns.
        EmptyDocumentError: If all rows are empty after serialisation.
    """
    df: pd.DataFrame | None = None
    last_error: Exception | None = None

    for encoding in ("utf-8", "latin-1"):
        try:
            df = _try_parse(file_bytes, encoding)
            logger.debug(
                "csv parsed",
                extra={
                    "source_file": filename,
                    "encoding": encoding,
                    "rows": len(df),
                    "columns": len(df.columns),
                },
            )
            break
        except Exception as exc:
            last_error = exc
            continue

    if df is None:
        raise CSVParseError(
            f"Failed to parse '{filename}' as CSV with UTF-8 or latin-1 encoding. "
            f"Last error: {last_error}. "
            "Check that the file is a valid comma-separated text file."
        )

    if len(df.columns) == 0:
        raise CSVParseError(
            f"'{filename}' has zero columns. "
            "The file may be empty or not a valid CSV."
        )

    columns = list(df.columns)
    serialised_rows: list[str] = []

    for _, row in df.iterrows():
        line = serialise_row(row, columns)
        if line:  # skip rows where all values were NaN
            serialised_rows.append(line)

    if len(serialised_rows) < MIN_ROW_COUNT:
        raise EmptyDocumentError(
            f"'{filename}' contains no usable rows after parsing. "
            f"Parsed {len(df)} rows but all were empty or contained only NaN values."
        )

    full_text = "\n".join(serialised_rows)

    logger.info(
        "csv parsed",
        extra={
            "source_file": filename,
            "total_rows": len(df),
            "serialised_rows": len(serialised_rows),
            "columns": columns,
            "total_chars": len(full_text),
        },
    )

    return full_text
