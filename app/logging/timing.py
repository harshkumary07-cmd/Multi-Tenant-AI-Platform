"""
Latency measurement utilities for structured logging.

Provides lightweight timing helpers used in the request lifecycle and
in service methods that need to log their own execution time.

Design:
    All timers use time.monotonic() -- not time.time().
    time.monotonic() is guaranteed to be non-decreasing and is not
    affected by system clock adjustments (NTP, DST, manual changes).
    It measures elapsed wall-clock time including I/O wait, which is
    the correct metric for user-facing latency.

    Timers are not context managers by default to avoid hiding the
    start time in a with-block. The RequestLoggerMiddleware needs to
    record the start time before the request enters the route handler
    and the end time after the response is fully prepared -- a pattern
    that does not fit naturally into a single with-block.
"""

import time
from dataclasses import dataclass, field


def start_timer() -> float:
    """
    Record the current monotonic time as a timer start point.

    Returns:
        float: Monotonic timestamp in seconds. Pass this to elapsed_ms().

    Usage:
        start = start_timer()
        # ... do work ...
        latency = elapsed_ms(start)
    """
    return time.monotonic()


def elapsed_ms(start: float) -> int:
    """
    Compute elapsed milliseconds since start.

    Args:
        start: The value returned by start_timer().

    Returns:
        int: Elapsed time in whole milliseconds.
             Integer (not float) because sub-millisecond precision is
             not meaningful for HTTP request latency logging.

    Usage:
        start = start_timer()
        result = await some_async_operation()
        latency_ms = elapsed_ms(start)
        logger.info("operation complete", extra={"latency_ms": latency_ms})
    """
    return int((time.monotonic() - start) * 1000)


@dataclass
class LatencyTracker:
    """
    Multi-stage latency tracker for pipeline operations.

    Records named checkpoints throughout a multi-step pipeline (e.g.
    the document ingestion pipeline: parse, chunk, embed, store).
    At the end, emits a single log line with all stage durations.

    Usage:
        tracker = LatencyTracker()
        tracker.checkpoint("parse")
        # ... parse document ...
        tracker.checkpoint("chunk")
        # ... chunk text ...
        tracker.checkpoint("embed")
        # ... embed chunks ...
        tracker.checkpoint("store")

        logger.info("ingestion complete", extra={
            **tracker.to_log_fields(),
            "doc_id": doc_id,
            "chunks_stored": len(chunks),
        })

    The to_log_fields() output:
        {
            "total_latency_ms": 8300,
            "stage_parse_ms": 120,
            "stage_chunk_ms": 40,
            "stage_embed_ms": 5800,
            "stage_store_ms": 2340,
        }
    """

    _start: float = field(default_factory=time.monotonic, init=False)
    _last: float = field(default_factory=time.monotonic, init=False)
    _stages: dict[str, int] = field(default_factory=dict, init=False)

    def checkpoint(self, stage_name: str) -> int:
        """
        Record the time elapsed since the last checkpoint.

        Args:
            stage_name: A short label for this pipeline stage (e.g. "parse").
                        Used as the key in to_log_fields() output.

        Returns:
            int: Milliseconds elapsed since the previous checkpoint
                 (or since construction if this is the first checkpoint).
        """
        now = time.monotonic()
        stage_ms = int((now - self._last) * 1000)
        self._stages[stage_name] = stage_ms
        self._last = now
        return stage_ms

    def total_ms(self) -> int:
        """
        Return total elapsed milliseconds since construction.

        Returns:
            int: Total wall-clock time in milliseconds.
        """
        return int((time.monotonic() - self._start) * 1000)

    def to_log_fields(self) -> dict[str, int]:
        """
        Return a flat dict of all stage durations plus the total.

        Keys follow the pattern "stage_{name}_ms" for each checkpoint,
        plus "total_latency_ms" for the overall duration.

        Returns:
            dict[str, int]: Flat key-value pairs for use in extra={}.
        """
        fields: dict[str, int] = {"total_latency_ms": self.total_ms()}
        for stage_name, stage_ms in self._stages.items():
            fields[f"stage_{stage_name}_ms"] = stage_ms
        return fields
