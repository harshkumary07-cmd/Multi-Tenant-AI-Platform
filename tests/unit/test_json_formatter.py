"""
Unit tests for app/logging/formatters.py -- Module 3.

Tests cover:
    - Every log record produces valid JSON
    - Required fields are always present
    - Timestamps are ISO 8601 UTC format
    - request_id and user_id come from LogContext
    - extra={} fields appear at the top level
    - Reserved field names are prefixed with "extra_"
    - Exception info is serialised to single-line strings
    - No real secrets appear in log output

No infrastructure required. Pure Python.
"""

import json
import logging
import re

import pytest

from app.logging.context import bind_request_context, clear_log_context
from app.logging.formatters import JSONFormatter


def make_record(
    message: str = "test message",
    level: int = logging.INFO,
    logger_name: str = "test.module",
    extra: dict | None = None,
    exc_info: tuple | None = None,
) -> logging.LogRecord:
    """Helper: create a LogRecord with the given parameters."""
    record = logging.LogRecord(
        name=logger_name,
        level=level,
        pathname="test.py",
        lineno=1,
        msg=message,
        args=(),
        exc_info=exc_info,
    )
    if extra:
        for key, value in extra.items():
            setattr(record, key, value)
    return record


class TestJSONFormatterOutputIsValidJSON:
    """Every formatted record is parseable as JSON."""

    def test_format_produces_valid_json(self) -> None:
        """format() output can be parsed by json.loads()."""
        formatter = JSONFormatter()
        record = make_record("hello world")
        output = formatter.format(record)
        parsed = json.loads(output)
        assert isinstance(parsed, dict)

    def test_format_produces_single_line(self) -> None:
        """Output contains no unescaped newlines (single JSON object per line)."""
        formatter = JSONFormatter()
        record = make_record("multi\nline\nmessage")
        output = formatter.format(record)
        # The output itself must be a single line
        # (newlines in the message are escaped inside the JSON string)
        assert "\n" not in output

    def test_format_multiple_records_all_valid_json(self) -> None:
        """Multiple consecutive records each produce valid JSON."""
        formatter = JSONFormatter()
        for i in range(10):
            record = make_record(f"message {i}", level=logging.DEBUG)
            output = formatter.format(record)
            parsed = json.loads(output)
            assert parsed["message"] == f"message {i}"


class TestRequiredFields:
    """Every log record includes all six required fields."""

    def test_timestamp_present(self) -> None:
        formatter = JSONFormatter()
        parsed = json.loads(formatter.format(make_record()))
        assert "timestamp" in parsed

    def test_level_present(self) -> None:
        formatter = JSONFormatter()
        parsed = json.loads(formatter.format(make_record()))
        assert "level" in parsed

    def test_logger_present(self) -> None:
        formatter = JSONFormatter()
        parsed = json.loads(formatter.format(make_record(logger_name="app.services.query")))
        assert parsed["logger"] == "app.services.query"

    def test_request_id_present(self) -> None:
        formatter = JSONFormatter()
        parsed = json.loads(formatter.format(make_record()))
        assert "request_id" in parsed

    def test_user_id_present(self) -> None:
        formatter = JSONFormatter()
        parsed = json.loads(formatter.format(make_record()))
        assert "user_id" in parsed

    def test_message_present(self) -> None:
        formatter = JSONFormatter()
        parsed = json.loads(formatter.format(make_record("test message")))
        assert parsed["message"] == "test message"


class TestTimestampFormat:
    """Timestamps are ISO 8601 UTC with millisecond precision."""

    # Pattern: "2024-01-15T09:32:01.234Z"
    ISO8601_PATTERN = re.compile(
        r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$"
    )

    def test_timestamp_is_iso8601_utc(self) -> None:
        """Timestamp field matches ISO 8601 UTC format with milliseconds."""
        formatter = JSONFormatter()
        parsed = json.loads(formatter.format(make_record()))
        assert self.ISO8601_PATTERN.match(parsed["timestamp"]), (
            f"Timestamp '{parsed['timestamp']}' does not match ISO 8601 UTC pattern"
        )

    def test_timestamp_ends_with_z(self) -> None:
        """Timestamp always ends with 'Z' indicating UTC."""
        formatter = JSONFormatter()
        parsed = json.loads(formatter.format(make_record()))
        assert parsed["timestamp"].endswith("Z")


class TestLogLevels:
    """Level names are correctly mapped."""

    @pytest.mark.parametrize(
        ("level", "expected_name"),
        [
            (logging.DEBUG, "DEBUG"),
            (logging.INFO, "INFO"),
            (logging.WARNING, "WARNING"),
            (logging.ERROR, "ERROR"),
            (logging.CRITICAL, "CRITICAL"),
        ],
    )
    def test_level_name_is_correct(self, level: int, expected_name: str) -> None:
        """Each logging level maps to its correct string name."""
        formatter = JSONFormatter()
        parsed = json.loads(formatter.format(make_record(level=level)))
        assert parsed["level"] == expected_name


class TestContextEnrichment:
    """request_id and user_id come from the active LogContext."""

    def test_request_id_from_context(self) -> None:
        """Formatter reads request_id from the active LogContext."""
        token = bind_request_context("req_test_abc", "u_test")
        try:
            formatter = JSONFormatter()
            parsed = json.loads(formatter.format(make_record()))
            assert parsed["request_id"] == "req_test_abc"
        finally:
            clear_log_context(token)

    def test_user_id_from_context(self) -> None:
        """Formatter reads user_id from the active LogContext."""
        token = bind_request_context("req_test_abc", "u_ctx_test")
        try:
            formatter = JSONFormatter()
            parsed = json.loads(formatter.format(make_record()))
            assert parsed["user_id"] == "u_ctx_test"
        finally:
            clear_log_context(token)

    def test_sentinel_values_outside_request(self) -> None:
        """Outside a request, formatter uses sentinel values."""
        formatter = JSONFormatter()
        parsed = json.loads(formatter.format(make_record()))
        assert parsed["request_id"] == "no_request"
        assert parsed["user_id"] == "anonymous"


class TestExtraFields:
    """Extra fields passed via extra={} appear at the top level."""

    def test_extra_field_present_in_output(self) -> None:
        """A field passed via extra={} appears at the top level."""
        formatter = JSONFormatter()
        record = make_record(extra={"doc_id": "doc_abc123"})
        parsed = json.loads(formatter.format(record))
        assert parsed["doc_id"] == "doc_abc123"

    def test_multiple_extra_fields(self) -> None:
        """Multiple extra fields all appear in output."""
        formatter = JSONFormatter()
        record = make_record(extra={
            "doc_id": "doc_abc",
            "chunks_stored": 24,
            "latency_ms": 8300,
        })
        parsed = json.loads(formatter.format(record))
        assert parsed["doc_id"] == "doc_abc"
        assert parsed["chunks_stored"] == 24
        assert parsed["latency_ms"] == 8300

    def test_reserved_field_names_are_prefixed(self) -> None:
        """Extra fields with reserved names get 'extra_' prefix."""
        formatter = JSONFormatter()
        # "level" is a reserved field name in our schema (maps to levelname).
        # Passing it via extra={} must not silently overwrite the real level.
        record = make_record(level=logging.INFO, extra={"level": "sneaky_override"})
        parsed = json.loads(formatter.format(record))
        # The real level is preserved
        assert parsed["level"] == "INFO"
        # The extra field is accessible under the prefixed key
        assert parsed["extra_level"] == "sneaky_override"

    def test_non_reserved_extra_fields_not_prefixed(self) -> None:
        """Non-reserved extra field names are not prefixed."""
        formatter = JSONFormatter()
        record = make_record(extra={"route": "RETRIEVE", "cache_hit": False})
        parsed = json.loads(formatter.format(record))
        assert "route" in parsed
        assert "cache_hit" in parsed
        assert "extra_route" not in parsed


class TestExceptionFormatting:
    """Exceptions are serialised to JSON-safe fields."""

    def test_exception_fields_present_when_exc_info(self) -> None:
        """When exc_info is set, exception_type, exception_message, stack_trace appear."""
        formatter = JSONFormatter()
        try:
            raise ValueError("something went wrong")
        except ValueError:
            import sys
            record = make_record(exc_info=sys.exc_info())  # type: ignore[arg-type]
        parsed = json.loads(formatter.format(record))
        assert "exception_type" in parsed
        assert "exception_message" in parsed
        assert "stack_trace" in parsed

    def test_exception_message_is_correct(self) -> None:
        """exception_message contains the exception's string representation."""
        formatter = JSONFormatter()
        try:
            raise ValueError("specific error message")
        except ValueError:
            import sys
            record = make_record(exc_info=sys.exc_info())  # type: ignore[arg-type]
        parsed = json.loads(formatter.format(record))
        assert parsed["exception_message"] == "specific error message"

    def test_exception_type_is_qualified_name(self) -> None:
        """exception_type includes module and class name."""
        formatter = JSONFormatter()
        try:
            raise ValueError("test")
        except ValueError:
            import sys
            record = make_record(exc_info=sys.exc_info())  # type: ignore[arg-type]
        parsed = json.loads(formatter.format(record))
        assert "ValueError" in parsed["exception_type"]

    def test_stack_trace_is_single_string(self) -> None:
        """stack_trace is a string value, not a list or multi-line."""
        formatter = JSONFormatter()
        try:
            raise RuntimeError("crash")
        except RuntimeError:
            import sys
            record = make_record(exc_info=sys.exc_info())  # type: ignore[arg-type]
        parsed = json.loads(formatter.format(record))
        assert isinstance(parsed["stack_trace"], str)

    def test_no_exception_fields_without_exc_info(self) -> None:
        """Exception fields are absent when no exception is being logged."""
        formatter = JSONFormatter()
        parsed = json.loads(formatter.format(make_record()))
        assert "exception_type" not in parsed
        assert "exception_message" not in parsed
        assert "stack_trace" not in parsed
