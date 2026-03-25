"""Shared test fixtures for pop tests."""

from __future__ import annotations

import pytest

from pop.types import ModelResponse, TokenUsage, ToolCall


class MockModelResponses:
    """A configurable mock that returns pre-defined responses in sequence."""

    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = list(responses)
        self._index = 0

    def next_response(self) -> ModelResponse:
        if self._index >= len(self._responses):
            return ModelResponse(
                content="No more mock responses configured.",
                token_usage=TokenUsage(input_tokens=10, output_tokens=5),
            )
        response = self._responses[self._index]
        self._index += 1
        return response

    @property
    def call_count(self) -> int:
        return self._index


@pytest.fixture
def mock_final_response() -> ModelResponse:
    """A simple final answer response."""
    return ModelResponse(
        content="The answer is 42.",
        token_usage=TokenUsage(input_tokens=100, output_tokens=20),
        model="mock:test",
        finish_reason="stop",
    )


@pytest.fixture
def mock_tool_call_response() -> ModelResponse:
    """A response that requests a tool call."""
    return ModelResponse(
        content="",
        tool_calls=(ToolCall(name="search", args={"query": "test query"}, call_id="call_1"),),
        token_usage=TokenUsage(input_tokens=100, output_tokens=30),
        model="mock:test",
        finish_reason="tool_calls",
    )
