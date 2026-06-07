"""
Unit tests for app/logging/context.py -- Module 3.

Tests cover:
    - Default context sentinel values outside a request
    - bind_request_context sets request_id and user_id
    - get_log_context returns the current context correctly
    - clear_log_context restores the previous state
    - update_user_id preserves request_id while replacing user_id
    - LogContext is immutable (frozen dataclass)
    - Isolation between independent bindings (simulates concurrent requests)

No infrastructure required. Pure Python -- runs in under 1 second.
"""

import pytest

from app.logging.context import (
    LogContext,
    bind_request_context,
    clear_log_context,
    get_log_context,
    update_user_id,
)


class TestDefaultContext:
    """Outside a request lifecycle, the sentinel is returned."""

    def test_get_log_context_returns_value_outside_request(self) -> None:
        """get_log_context() never raises -- returns sentinel outside request."""
        context = get_log_context()
        assert isinstance(context, LogContext)

    def test_default_request_id_is_no_request(self) -> None:
        """Outside a request, request_id is the 'no_request' sentinel."""
        context = get_log_context()
        assert context.request_id == "no_request"

    def test_default_user_id_is_anonymous(self) -> None:
        """Outside a request, user_id is the 'anonymous' sentinel."""
        context = get_log_context()
        assert context.user_id == "anonymous"


class TestBindRequestContext:
    """bind_request_context sets context for the current async task."""

    def test_bind_sets_request_id(self) -> None:
        """After binding, get_log_context() returns the bound request_id."""
        token = bind_request_context(request_id="req_abc123")
        try:
            assert get_log_context().request_id == "req_abc123"
        finally:
            clear_log_context(token)

    def test_bind_sets_user_id(self) -> None:
        """After binding with user_id, get_log_context() returns the user_id."""
        token = bind_request_context(request_id="req_abc123", user_id="u1")
        try:
            assert get_log_context().user_id == "u1"
        finally:
            clear_log_context(token)

    def test_bind_default_user_id_is_anonymous(self) -> None:
        """user_id defaults to 'anonymous' when not provided."""
        token = bind_request_context(request_id="req_def456")
        try:
            assert get_log_context().user_id == "anonymous"
        finally:
            clear_log_context(token)

    def test_bind_returns_token(self) -> None:
        """bind_request_context returns a Token for later cleanup."""
        from contextvars import Token

        token = bind_request_context(request_id="req_xyz789")
        try:
            assert isinstance(token, Token)
        finally:
            clear_log_context(token)


class TestClearLogContext:
    """clear_log_context restores the previous state."""

    def test_clear_restores_sentinel_after_bind(self) -> None:
        """After clearing, context returns to the 'no_request' sentinel."""
        token = bind_request_context(request_id="req_abc123")
        clear_log_context(token)
        context = get_log_context()
        assert context.request_id == "no_request"
        assert context.user_id == "anonymous"

    def test_clear_in_finally_block(self) -> None:
        """Context is cleaned up even when an exception is raised mid-request."""
        token = bind_request_context(request_id="req_abc123", user_id="u1")
        try:
            raise ValueError("simulated error")
        except ValueError:
            pass
        finally:
            clear_log_context(token)

        # After cleanup, sentinel is restored
        assert get_log_context().request_id == "no_request"

    def test_nested_contexts_restored_correctly(self) -> None:
        """Nested bind/clear pairs correctly restore the outer context."""
        outer_token = bind_request_context("req_outer", "u_outer")
        assert get_log_context().request_id == "req_outer"

        inner_token = bind_request_context("req_inner", "u_inner")
        assert get_log_context().request_id == "req_inner"

        clear_log_context(inner_token)
        assert get_log_context().request_id == "req_outer"

        clear_log_context(outer_token)
        assert get_log_context().request_id == "no_request"


class TestUpdateUserId:
    """update_user_id updates user_id while preserving request_id."""

    def test_update_user_id_preserves_request_id(self) -> None:
        """Updating user_id does not change the existing request_id."""
        bind_token = bind_request_context("req_abc123", "anonymous")
        update_token = update_user_id("u1")
        try:
            ctx = get_log_context()
            assert ctx.request_id == "req_abc123"
            assert ctx.user_id == "u1"
        finally:
            clear_log_context(update_token)
            clear_log_context(bind_token)

    def test_update_user_id_replaces_anonymous(self) -> None:
        """update_user_id replaces the 'anonymous' placeholder."""
        bind_token = bind_request_context("req_abc123")
        assert get_log_context().user_id == "anonymous"

        update_token = update_user_id("u_real")
        try:
            assert get_log_context().user_id == "u_real"
        finally:
            clear_log_context(update_token)
            clear_log_context(bind_token)

    def test_update_user_id_returns_token(self) -> None:
        """update_user_id returns a Token for cleanup."""
        from contextvars import Token

        bind_token = bind_request_context("req_abc123")
        update_token = update_user_id("u1")
        try:
            assert isinstance(update_token, Token)
        finally:
            clear_log_context(update_token)
            clear_log_context(bind_token)


class TestLogContextImmutability:
    """LogContext is a frozen dataclass -- mutation raises an error."""

    def test_log_context_is_frozen(self) -> None:
        """Attempting to mutate a LogContext raises an error."""
        import dataclasses

        ctx = LogContext(request_id="req_test", user_id="u_test")
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.request_id = "modified"  # type: ignore[misc]

    def test_log_context_equality(self) -> None:
        """Two LogContext instances with the same values are equal."""
        ctx1 = LogContext(request_id="req_test", user_id="u1")
        ctx2 = LogContext(request_id="req_test", user_id="u1")
        assert ctx1 == ctx2

    def test_log_context_different_values_not_equal(self) -> None:
        """LogContexts with different values are not equal."""
        ctx1 = LogContext(request_id="req_a", user_id="u1")
        ctx2 = LogContext(request_id="req_b", user_id="u1")
        assert ctx1 != ctx2


class TestContextIsolation:
    """Independent bind/clear cycles do not interfere with each other."""

    def test_two_independent_contexts_are_isolated(self) -> None:
        """
        Simulates two sequential requests in the same thread.
        Each bind/clear cycle is independent.
        """
        # First 'request'
        token_a = bind_request_context("req_first", "u_a")
        assert get_log_context().request_id == "req_first"
        clear_log_context(token_a)
        assert get_log_context().request_id == "no_request"

        # Second 'request'
        token_b = bind_request_context("req_second", "u_b")
        assert get_log_context().request_id == "req_second"
        assert get_log_context().user_id == "u_b"
        clear_log_context(token_b)
        assert get_log_context().request_id == "no_request"
