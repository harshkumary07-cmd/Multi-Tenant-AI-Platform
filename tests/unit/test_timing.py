"""
Unit tests for app/logging/timing.py -- Module 3.

Tests cover:
    - start_timer returns a float
    - elapsed_ms returns a non-negative integer
    - elapsed_ms measures real elapsed time
    - LatencyTracker records checkpoints correctly
    - LatencyTracker.to_log_fields() returns the correct structure
    - total_ms() increases over time
"""

import time

from app.logging.timing import LatencyTracker, elapsed_ms, start_timer


class TestStartTimer:
    """start_timer() returns a float monotonic timestamp."""

    def test_returns_float(self) -> None:
        """start_timer() returns a float."""
        result = start_timer()
        assert isinstance(result, float)

    def test_second_call_is_greater_or_equal(self) -> None:
        """Monotonic clock never goes backwards."""
        t1 = start_timer()
        t2 = start_timer()
        assert t2 >= t1


class TestElapsedMs:
    """elapsed_ms() returns non-negative integer milliseconds."""

    def test_returns_integer(self) -> None:
        """elapsed_ms() returns an int."""
        start = start_timer()
        result = elapsed_ms(start)
        assert isinstance(result, int)

    def test_returns_non_negative(self) -> None:
        """elapsed_ms() is always >= 0."""
        start = start_timer()
        result = elapsed_ms(start)
        assert result >= 0

    def test_measures_real_elapsed_time(self) -> None:
        """elapsed_ms() reflects real time passing."""
        start = start_timer()
        time.sleep(0.05)  # 50ms sleep
        result = elapsed_ms(start)
        # Allow wide margin for CI environment jitter
        assert result >= 30, f"Expected >= 30ms, got {result}ms"
        assert result <= 500, f"Expected <= 500ms, got {result}ms"

    def test_immediate_call_is_near_zero(self) -> None:
        """Without sleep, elapsed_ms is very small (but may not be exactly 0)."""
        start = start_timer()
        result = elapsed_ms(start)
        assert result < 100, f"Immediate call should be < 100ms, got {result}ms"


class TestLatencyTracker:
    """LatencyTracker records named checkpoints and computes totals."""

    def test_total_ms_increases_over_time(self) -> None:
        """total_ms() grows as time passes."""
        tracker = LatencyTracker()
        first = tracker.total_ms()
        time.sleep(0.02)
        second = tracker.total_ms()
        assert second >= first

    def test_checkpoint_returns_stage_ms(self) -> None:
        """checkpoint() returns the ms elapsed since the last checkpoint."""
        tracker = LatencyTracker()
        time.sleep(0.02)
        ms = tracker.checkpoint("stage_a")
        assert isinstance(ms, int)
        assert ms >= 0

    def test_multiple_checkpoints_recorded(self) -> None:
        """Multiple checkpoints all appear in to_log_fields()."""
        tracker = LatencyTracker()
        tracker.checkpoint("parse")
        tracker.checkpoint("chunk")
        tracker.checkpoint("embed")
        fields = tracker.to_log_fields()
        assert "stage_parse_ms" in fields
        assert "stage_chunk_ms" in fields
        assert "stage_embed_ms" in fields

    def test_total_latency_ms_in_fields(self) -> None:
        """to_log_fields() includes 'total_latency_ms' key."""
        tracker = LatencyTracker()
        tracker.checkpoint("step")
        fields = tracker.to_log_fields()
        assert "total_latency_ms" in fields

    def test_all_field_values_are_integers(self) -> None:
        """All values in to_log_fields() are integers."""
        tracker = LatencyTracker()
        tracker.checkpoint("stage_a")
        tracker.checkpoint("stage_b")
        for key, value in tracker.to_log_fields().items():
            assert isinstance(value, int), f"Field '{key}' has non-int value: {value!r}"

    def test_all_field_values_are_non_negative(self) -> None:
        """Stage durations and total are all >= 0."""
        tracker = LatencyTracker()
        tracker.checkpoint("parse")
        tracker.checkpoint("store")
        for key, value in tracker.to_log_fields().items():
            assert value >= 0, f"Field '{key}' is negative: {value}"

    def test_stage_key_format(self) -> None:
        """Stage keys follow the 'stage_{name}_ms' pattern."""
        tracker = LatencyTracker()
        tracker.checkpoint("embedding")
        fields = tracker.to_log_fields()
        assert "stage_embedding_ms" in fields
        assert "embedding" not in fields  # raw name not present

    def test_no_checkpoints_gives_only_total(self) -> None:
        """A tracker with no checkpoints returns only 'total_latency_ms'."""
        tracker = LatencyTracker()
        fields = tracker.to_log_fields()
        assert list(fields.keys()) == ["total_latency_ms"]

    def test_to_log_fields_is_serialisable(self) -> None:
        """to_log_fields() output can be used in extra={} for JSON logging."""
        import json
        tracker = LatencyTracker()
        tracker.checkpoint("step_a")
        fields = tracker.to_log_fields()
        # Should not raise
        serialised = json.dumps(fields)
        parsed = json.loads(serialised)
        assert "total_latency_ms" in parsed
