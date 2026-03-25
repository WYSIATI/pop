"""Tests for the Runner execution engine."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from pop.hooks.base import Hook, HookManager
from pop.runner import Runner, run
from pop.types import (
    Action,
    ActionType,
    AgentResult,
    AgentState,
    DoneEvent,
    Status,
    Step,
    StreamEvent,
    TextDeltaEvent,
    ToolCall,
    ToolCallEvent,
    ToolResultEvent,
)

# ---------------------------------------------------------------------------
# Mock agent
# ---------------------------------------------------------------------------


class MockAgent:
    """A configurable mock Agent for testing the Runner."""

    def __init__(
        self,
        result: AgentResult,
        delay: float = 0.0,
    ) -> None:
        self._result = result
        self._delay = delay
        self.arun_calls: list[dict[str, Any]] = []
        self._hook_manager = HookManager()

    async def arun(self, task: str, **kwargs: Any) -> AgentResult:
        self.arun_calls.append({"task": task, **kwargs})
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        return self._result

    def run(self, task: str, **kwargs: Any) -> AgentResult:
        return self._result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_result(
    output: str = "done",
    steps: tuple[Step, ...] = (),
    run_id: str = "",
) -> AgentResult:
    return AgentResult(
        output=output,
        steps=steps,
        state=AgentState(status=Status.DONE),
        run_id=run_id,
    )


def _make_step_with_tool(
    index: int = 0,
    tool_name: str = "search",
    tool_args: dict[str, Any] | None = None,
    tool_result: str = "found it",
) -> Step:
    from datetime import datetime, timezone

    return Step(
        index=index,
        timestamp=datetime.now(timezone.utc),
        action=Action(
            type=ActionType.TOOL_CALL,
            tool_call=ToolCall(name=tool_name, args=tool_args or {}),
        ),
        tool_name=tool_name,
        tool_args=tool_args or {},
        tool_result=tool_result,
    )


def _make_final_step(index: int = 0, answer: str = "done") -> Step:
    from datetime import datetime, timezone

    return Step(
        index=index,
        timestamp=datetime.now(timezone.utc),
        action=Action(type=ActionType.FINAL_ANSWER, answer=answer),
    )


# ---------------------------------------------------------------------------
# 1. Sync run returns AgentResult
# ---------------------------------------------------------------------------


def test_sync_run_returns_agent_result():
    result = _make_result(output="hello sync")
    agent = MockAgent(result=result)
    runner = Runner(agent)

    got = runner.run("do something")

    assert got.output == "hello sync"


# ---------------------------------------------------------------------------
# 2. Async arun returns AgentResult
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_arun_returns_agent_result():
    result = _make_result(output="hello async")
    agent = MockAgent(result=result)
    runner = Runner(agent)

    got = await runner.arun("do something")

    assert got.output == "hello async"


# ---------------------------------------------------------------------------
# 3. Run ID generation when not provided
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_id_generated_when_not_provided():
    result = _make_result(output="ok")
    agent = MockAgent(result=result)
    runner = Runner(agent)

    got = await runner.arun("task")

    assert got.run_id != ""
    assert len(got.run_id) > 8  # UUID-like


# ---------------------------------------------------------------------------
# 4. Custom run_id is used
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_custom_run_id_is_used():
    result = _make_result(output="ok")
    agent = MockAgent(result=result)
    runner = Runner(agent)

    got = await runner.arun("task", run_id="my-custom-id")

    assert got.run_id == "my-custom-id"


# ---------------------------------------------------------------------------
# 5. on_step callback is called for each step
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_step_callback():
    step0 = _make_step_with_tool(index=0)
    step1 = _make_final_step(index=1, answer="42")
    result = _make_result(output="42", steps=(step0, step1))
    agent = MockAgent(result=result)
    runner = Runner(agent)

    collected_steps: list[Step] = []

    await runner.arun("task", on_step=collected_steps.append)

    assert len(collected_steps) == 2
    assert collected_steps[0].index == 0
    assert collected_steps[1].index == 1


# ---------------------------------------------------------------------------
# 6. Timeout enforcement
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_timeout_enforcement():
    result = _make_result(output="slow")
    agent = MockAgent(result=result, delay=5.0)
    runner = Runner(agent)

    with pytest.raises((asyncio.TimeoutError, TimeoutError)):
        await runner.arun("task", timeout=0.05)


# ---------------------------------------------------------------------------
# 7. Hooks firing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hooks_firing():
    result = _make_result(output="hooked")
    agent = MockAgent(result=result)

    hook = MagicMock(spec=Hook)
    runner = Runner(agent, hooks=[hook])

    await runner.arun("task")

    hook.on_run_start.assert_called_once()
    hook.on_run_end.assert_called_once()
    # on_run_end receives the result
    call_args = hook.on_run_end.call_args
    assert call_args[0][0].output == "hooked"


# ---------------------------------------------------------------------------
# 8. Stream yields events
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_yields_events():
    step0 = _make_step_with_tool(index=0, tool_name="search", tool_result="found")
    step1 = _make_final_step(index=1, answer="The answer")
    result = _make_result(output="The answer", steps=(step0, step1))
    agent = MockAgent(result=result)
    runner = Runner(agent)

    events: list[StreamEvent] = []
    async for event in runner.stream("task"):
        events.append(event)

    # Should have at least: ToolCallEvent, ToolResultEvent, TextDeltaEvent, DoneEvent
    event_types = [type(e) for e in events]
    assert ToolCallEvent in event_types
    assert ToolResultEvent in event_types
    assert TextDeltaEvent in event_types
    assert DoneEvent in event_types


# ---------------------------------------------------------------------------
# 9. Stream DoneEvent is always last
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_done_event_is_last():
    step0 = _make_step_with_tool(index=0)
    step1 = _make_final_step(index=1, answer="final")
    result = _make_result(output="final", steps=(step0, step1))
    agent = MockAgent(result=result)
    runner = Runner(agent)

    events: list[StreamEvent] = []
    async for event in runner.stream("task"):
        events.append(event)

    assert len(events) > 0
    assert isinstance(events[-1], DoneEvent)
    assert events[-1].result is not None
    assert events[-1].result.output == "final"


# ---------------------------------------------------------------------------
# 10. Convenience run function
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_convenience_run_function():
    result = _make_result(output="convenient")
    agent = MockAgent(result=result)

    got = await run(agent, "task")

    assert got.output == "convenient"


# ---------------------------------------------------------------------------
# 11. Runner with no hooks works fine
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runner_with_no_hooks():
    result = _make_result(output="no hooks")
    agent = MockAgent(result=result)
    runner = Runner(agent)

    got = await runner.arun("task")

    assert got.output == "no hooks"


# ---------------------------------------------------------------------------
# 12. _with_run_id creates new result when IDs differ
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_with_run_id_replaces_different_id():
    """When agent returns a different run_id, _with_run_id creates a new result."""
    from pop.runner import _with_run_id

    original = _make_result(output="test", run_id="original-id")
    updated = _with_run_id(original, "new-id")

    assert updated.run_id == "new-id"
    assert updated.output == "test"
    # Original should be unchanged
    assert original.run_id == "original-id"


@pytest.mark.asyncio
async def test_with_run_id_returns_same_when_matching():
    """When run_ids match, _with_run_id returns the same object."""
    from pop.runner import _with_run_id

    original = _make_result(output="test", run_id="same-id")
    updated = _with_run_id(original, "same-id")

    assert updated is original
