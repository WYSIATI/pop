"""Tests for workflow patterns: chain, route, parallel."""

from __future__ import annotations

import pytest

from pop.types import ModelResponse, TokenUsage


class MockAdapter:
    """A mock model adapter that returns pre-configured responses in order."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._call_index = 0
        self.messages_log: list[list] = []

    async def chat(self, messages: list, tools: list | None = None) -> ModelResponse:
        self.messages_log = [*self.messages_log, messages]
        if self._call_index >= len(self._responses):
            raise RuntimeError("MockAdapter exhausted: no more responses")
        text = self._responses[self._call_index]
        self._call_index += 1
        return ModelResponse(content=text, token_usage=TokenUsage(10, 5))


# ── chain tests ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_chain_single_step() -> None:
    """Single-step chain returns the model output directly."""
    from pop.workflows.patterns import chain

    adapter = MockAdapter(["Hello world"])
    result = await chain(adapter, ["Tell me something"], "hi")

    assert result == "Hello world"
    assert len(adapter.messages_log) == 1


@pytest.mark.asyncio
async def test_chain_multi_step() -> None:
    """Multi-step chain passes {prev} output between steps."""
    from pop.workflows.patterns import chain

    adapter = MockAdapter(["step1-out", "step2-out", "step3-out"])
    steps = [
        "First: {input}",
        "Second, prev was: {prev}",
        "Third, prev was: {prev}",
    ]
    result = await chain(adapter, steps, "original")

    assert result == "step3-out"
    assert len(adapter.messages_log) == 3

    # Verify second call included step1 output in the prompt
    second_call_content = adapter.messages_log[1][0].content
    assert "step1-out" in second_call_content

    # Verify third call included step2 output in the prompt
    third_call_content = adapter.messages_log[2][0].content
    assert "step2-out" in third_call_content


@pytest.mark.asyncio
async def test_chain_input_placeholder() -> None:
    """The {input} placeholder is replaced with input_text in every step."""
    from pop.workflows.patterns import chain

    adapter = MockAdapter(["response1", "response2"])
    steps = [
        "Process: {input}",
        "Refine {input} with context: {prev}",
    ]
    result = await chain(adapter, steps, "my-data")

    assert result == "response2"
    first_content = adapter.messages_log[0][0].content
    assert "my-data" in first_content

    second_content = adapter.messages_log[1][0].content
    assert "my-data" in second_content
    assert "response1" in second_content


@pytest.mark.asyncio
async def test_chain_empty_steps_raises() -> None:
    """Chain with empty steps list raises ValueError."""
    from pop.workflows.patterns import chain

    adapter = MockAdapter([])
    with pytest.raises(ValueError, match="at least one step"):
        await chain(adapter, [], "input")


# ── route tests ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_route_correct_dispatch() -> None:
    """Model classifies input and the correct handler is called."""
    from pop.workflows.patterns import route

    adapter = MockAdapter(["question"])
    handler_called_with: list[str] = []

    def question_handler(text: str) -> str:
        handler_called_with.append(text)
        return "answered"

    def statement_handler(text: str) -> str:
        return "noted"

    routes = {
        "question": question_handler,
        "statement": statement_handler,
    }
    result = await route(adapter, "What is Python?", routes)

    assert result == "answered"
    assert handler_called_with == ["What is Python?"]


@pytest.mark.asyncio
async def test_route_unknown_route_raises() -> None:
    """Model returns a classification not in routes, raises ValueError."""
    from pop.workflows.patterns import route

    adapter = MockAdapter(["unknown_category"])
    routes = {
        "question": lambda t: "q",
        "statement": lambda t: "s",
    }
    with pytest.raises(ValueError, match="unknown_category"):
        await route(adapter, "test input", routes)


@pytest.mark.asyncio
async def test_route_case_insensitive_keys() -> None:
    """Route keys with mixed case are matched case-insensitively."""
    from pop.workflows.patterns import route

    adapter = MockAdapter(["support"])

    def support_handler(text: str) -> str:
        return "handled"

    routes = {"Support": support_handler, "Sales": lambda t: "sales"}
    result = await route(adapter, "help me", routes)

    assert result == "handled"


@pytest.mark.asyncio
async def test_route_strips_whitespace() -> None:
    """Model response with extra whitespace is stripped before matching."""
    from pop.workflows.patterns import route

    adapter = MockAdapter(["  question  \n"])
    called = False

    def handler(text: str) -> str:
        nonlocal called
        called = True
        return "ok"

    routes = {"question": handler}
    result = await route(adapter, "input", routes)

    assert result == "ok"
    assert called


# ── parallel tests ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_parallel_multiple_tasks() -> None:
    """Multiple tasks run and return the correct number of results."""
    from pop.workflows.patterns import parallel

    adapter = MockAdapter(["r1", "r2", "r3"])
    tasks = ["task a", "task b", "task c"]
    results = await parallel(adapter, tasks)

    assert len(results) == 3
    assert set(results) == {"r1", "r2", "r3"}


@pytest.mark.asyncio
async def test_parallel_with_context() -> None:
    """The {context} placeholder is replaced in each task."""
    from pop.workflows.patterns import parallel

    adapter = MockAdapter(["out1", "out2"])
    tasks = ["Do {context} thing1", "Do {context} thing2"]
    results = await parallel(adapter, tasks, context="important")

    assert len(results) == 2
    # Verify context was substituted
    for messages in adapter.messages_log:
        assert "important" in messages[0].content


@pytest.mark.asyncio
async def test_parallel_preserves_order() -> None:
    """Results are returned in the same order as input tasks."""
    from pop.workflows.patterns import parallel

    adapter = MockAdapter(["first", "second", "third"])
    tasks = ["t1", "t2", "t3"]
    results = await parallel(adapter, tasks)

    assert results == ["first", "second", "third"]


@pytest.mark.asyncio
async def test_parallel_empty_tasks() -> None:
    """Parallel with no tasks returns empty list."""
    from pop.workflows.patterns import parallel

    adapter = MockAdapter([])
    results = await parallel(adapter, [])

    assert results == []
