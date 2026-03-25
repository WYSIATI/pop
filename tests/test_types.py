"""Tests for types module — covering AgentError properties and _now."""

from __future__ import annotations

from pop.types import AgentError, ErrorClass, _now


class TestNow:
    def test_returns_utc_datetime(self) -> None:
        from datetime import timezone

        result = _now()
        assert result.tzinfo == timezone.utc


class TestAgentError:
    def test_is_rate_limit_true(self) -> None:
        err = AgentError(
            message="Rate limited",
            error_class=ErrorClass.RATE_LIMIT,
        )
        assert err.is_rate_limit is True
        assert err.is_tool_error is False
        assert err.is_fatal is False

    def test_is_tool_error_true(self) -> None:
        err = AgentError(
            message="Tool failed",
            error_class=ErrorClass.TRANSIENT,
        )
        assert err.is_tool_error is True
        assert err.is_rate_limit is False
        assert err.is_fatal is False

    def test_is_fatal_true(self) -> None:
        err = AgentError(
            message="Fatal error",
            error_class=ErrorClass.FATAL,
        )
        assert err.is_fatal is True
        assert err.is_rate_limit is False
        assert err.is_tool_error is False

    def test_other_error_class(self) -> None:
        err = AgentError(
            message="Validation error",
            error_class=ErrorClass.VALIDATION,
        )
        assert err.is_rate_limit is False
        assert err.is_tool_error is False
        assert err.is_fatal is False
