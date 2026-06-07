"""
JSON log formatter for structured, machine-parseable log output.

Every log line emitted by this formatter is a single JSON object
on a single line, terminated by a newline. This format is:
    - Directly ingestible by CloudWatch, Loki, Datadog, Splunk
    - Queryable by field name in any log aggregation tool
    - Parseable by json.loads() without pre-processing
    - Safe for log lines that contain special characters or newlines
      (newlines in messages are escaped as \\n within the JSON string)

Log schema (every line contains these fields):
    timestamp   ISO 8601 UTC timestamp with millisecond precision
    level       Log level name: DEBUG, INFO, WARNING, ERROR, CRITICAL
    logger      Logger name (module path, e.g. "app.services.query_service")
    request_id  From LogContext -- "no_request" outside request lifecycle
    user_id     From LogContext -- "anonymous" outside request lifecycle
    message     The log message string
    + any extra fields passed via extra={} on the log call

Exception fields (present only when exc_info is provided):
    exception_type    Fully qualified exception class name
    exception_message str(exception)
    stack_trace       Full traceback as a single escaped string

Design decisions:
    - UTC timestamps only. Local time zones in logs cause confusion when
      correlating events across services or after DST changes.
    - Millisecond precision. Microseconds are not useful for latency
      analysis and inflate log line length.
    - Stack traces as escaped single-line strings. Multi-line output
      breaks JSON parsing in log aggregators. The full trace is still
      available -- just on one line.
    - No colour codes. ANSI escape codes break log parsing. Use a
      terminal log viewer that can parse and colour JSON if needed.
"""

import json
import logging
import traceback
from datetime import UTC, datetime
from typing import Any

from app.logging.context import get_log_context


class JSONFormatter(logging.Formatter):
    """
    Custom log formatter that emits one JSON object per log record.

    Installed on the root logger handler at application startup.
    All loggers obtained via get_logger(__name__) inherit this formatter.

    Fields always present:
        timestamp, level, logger, request_id, user_id, message

    Fields present only on exception records:
        exception_type, exception_message, stack_trace

    Extra fields:
        Any key-value pairs passed via extra={} on the log call are
        merged into the top-level JSON object. Keys that collide with
        reserved field names (timestamp, level, etc.) are prefixed
        with "extra_" to prevent overwriting core fields.

    Reserved field names (cannot be overridden via extra={}):
        timestamp, level, logger, request_id, user_id, message,
        exception_type, exception_message, stack_trace
    """

    # Fields whose names are reserved in the output schema.
    # Extra fields with these names are prefixed with "extra_".
    _RESERVED_FIELDS = frozenset({
        "timestamp",
        "level",
        "logger",
        "request_id",
        "user_id",
        "message",
        "exception_type",
        "exception_message",
        "stack_trace",
    })

    # Standard logging.LogRecord attributes that are not extra fields.
    # These are excluded when extracting extra={} key-value pairs.
    _LOGRECORD_STANDARD_ATTRS = frozenset({
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "message", "module",
        "msecs", "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "thread", "threadName",
        "taskName",  # Python 3.12+
    })

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record as a single-line JSON string.

        Args:
            record: The log record to format.

        Returns:
            str: A single-line JSON object. Always ends without a trailing
                 newline (the StreamHandler appends the newline).
        """
        # Retrieve request-scoped context (never None -- returns sentinel)
        context = get_log_context()

        # Build the core log entry
        entry: dict[str, Any] = {
            "timestamp": self._utc_timestamp(record),
            "level": record.levelname,
            "logger": record.name,
            "request_id": context.request_id,
            "user_id": context.user_id,
            "message": record.getMessage(),
        }

        # Merge extra={} fields, protecting reserved names
        extra_fields = self._extract_extra_fields(record)
        for key, value in extra_fields.items():
            safe_key = f"extra_{key}" if key in self._RESERVED_FIELDS else key
            entry[safe_key] = value

        # Add exception information if present
        if record.exc_info:
            exc_type, exc_value, exc_tb = record.exc_info
            if exc_type is not None and exc_value is not None:
                entry["exception_type"] = (
                    f"{exc_type.__module__}.{exc_type.__qualname__}"
                )
                entry["exception_message"] = str(exc_value)
                entry["stack_trace"] = self._format_traceback(
                    exc_type, exc_value, exc_tb
                )

        return json.dumps(entry, default=str, ensure_ascii=False)

    @staticmethod
    def _utc_timestamp(record: logging.LogRecord) -> str:
        """
        Return the record's creation time as an ISO 8601 UTC string.

        Format: "2024-01-15T09:32:01.234Z"

        Uses record.created (a float Unix timestamp) for accuracy.
        Always UTC -- never local time. The trailing "Z" indicates UTC.
        """
        dt = datetime.fromtimestamp(record.created, tz=UTC)
        return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"

    def _extract_extra_fields(
        self, record: logging.LogRecord
    ) -> dict[str, Any]:
        """
        Extract the extra={} fields added by the calling log statement.

        Python's logging module attaches extra={} key-value pairs directly
        as attributes on the LogRecord. This method identifies them by
        excluding all standard LogRecord attributes.

        Args:
            record: The log record to inspect.

        Returns:
            dict: Key-value pairs from extra={} that are not standard
                  LogRecord attributes.
        """
        extra: dict[str, Any] = {}
        for key, value in record.__dict__.items():
            if key not in self._LOGRECORD_STANDARD_ATTRS and not key.startswith("_"):
                extra[key] = value
        return extra

    @staticmethod
    def _format_traceback(
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_tb: Any,
    ) -> str:
        """
        Format a traceback as a single escaped string suitable for JSON.

        Newlines within the traceback are preserved as \\n in the string.
        When the JSON is parsed, \\n becomes a real newline again, making
        the traceback readable in any tool that pretty-prints the JSON.

        Args:
            exc_type:  Exception class.
            exc_value: Exception instance.
            exc_tb:    Traceback object.

        Returns:
            str: The full formatted traceback as a single string value.
        """
        lines = traceback.format_exception(exc_type, exc_value, exc_tb)
        return "".join(lines).rstrip()
