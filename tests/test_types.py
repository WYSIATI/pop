"""Tests for types module — covering AgentError properties, AgentState, and ToolDefinition."""

from __future__ import annotations

from types import MappingProxyType

import pytest

from pop.types import AgentError, AgentState, ErrorClass, Status, ToolDefinition

# ---------------------------------------------------------------------------
# H4: AgentState.tool_results removed, metadata is immutable
# ---------------------------------------------------------------------------


class TestAgentStateImmutability:
    def test_no_tool_results_field(self) -> None:
        """tool_results field should have been removed from AgentState."""
        state = AgentState()
        assert not hasattr(state, "tool_results")

    def test_metadata_is_immutable_mapping(self) -> None:
        """metadata should be an immutable MappingProxyType."""
        state = AgentState(metadata=MappingProxyType({"key": "value"}))
        assert isinstance(state.metadata, MappingProxyType)
        assert state.metadata["key"] == "value"
        with pytest.raises(TypeError):
            state.metadata["key"] = "new"  # type: ignore[index]

    def test_default_metadata_is_empty_immutable(self) -> None:
        """Default metadata should be an empty MappingProxyType."""
        state = AgentState()
        assert isinstance(state.metadata, MappingProxyType)
        assert len(state.metadata) == 0

    def test_with_message_preserves_metadata(self) -> None:
        """with_message should carry metadata to the new state."""
        from pop.types import Message

        meta = MappingProxyType({"foo": "bar"})
        state = AgentState(metadata=meta)
        new_state = state.with_message(Message.user("hello"))
        assert new_state.metadata == meta
        assert isinstance(new_state.metadata, MappingProxyType)

    def test_with_step_preserves_metadata(self) -> None:
        """with_step should carry metadata to the new state."""
        meta = MappingProxyType({"foo": "bar"})
        state = AgentState(metadata=meta)
        new_state = state.with_step()
        assert new_state.metadata == meta

    def test_with_status_preserves_metadata(self) -> None:
        """with_status should carry metadata to the new state."""
        meta = MappingProxyType({"foo": "bar"})
        state = AgentState(metadata=meta)
        new_state = state.with_status(Status.DONE)
        assert new_state.metadata == meta


# ---------------------------------------------------------------------------
# M6: ToolDefinition.function typed as Callable
# ---------------------------------------------------------------------------


class TestToolDefinitionFunctionType:
    def test_function_field_accepts_callable(self) -> None:
        """ToolDefinition.function should accept any callable."""
        td = ToolDefinition(
            name="test",
            description="test tool",
            parameters={},
            function=lambda x: x,
        )
        assert callable(td.function)

    def test_function_field_is_callable_type(self) -> None:
        """ToolDefinition.function annotation should be Callable, not Any."""
        import inspect

        hints = {
            k: v for k, v in inspect.get_annotations(ToolDefinition).items() if k == "function"
        }
        # The annotation should mention Callable, not Any
        func_annotation = hints.get("function")
        # With from __future__ import annotations, annotations are strings
        assert func_annotation is not None
        annotation_str = str(func_annotation)
        assert "Callable" in annotation_str


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
