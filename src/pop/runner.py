"""Runner — higher-level execution engine for agents.

Adds streaming, step callbacks, run management, and timeout
enforcement on top of the core Agent.arun() loop.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING, Any

from pop.hooks.base import Hook, HookManager
from pop.types import (
    ActionType,
    AgentResult,
    DoneEvent,
    Step,
    StreamEvent,
    TextDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from pop.multi.patterns import AgentLike


class Runner:
    """Execution engine that drives agents with streaming, callbacks, and run management."""

    def __init__(self, agent: AgentLike, hooks: list[Hook] | None = None) -> None:
        self.agent = agent
        self._hook_manager = HookManager(hooks)

    def run(self, task: str, **kwargs: Any) -> AgentResult:
        """Sync execution. Handles running event loops (e.g. Jupyter)."""
        from pop._sync import run_sync

        return run_sync(self.arun(task, **kwargs))

    async def arun(
        self,
        task: str,
        *,
        run_id: str = "",
        on_step: Callable[[Step], None] | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Async execution with optional step callback and timeout."""
        run_id = run_id or str(uuid.uuid4())

        self._hook_manager.fire_run_start(task, run_id)

        try:
            result = await self._execute_with_timeout(
                task, run_id=run_id, timeout=timeout, **kwargs
            )
        except (asyncio.TimeoutError, TimeoutError):
            self._hook_manager.fire_run_end(AgentResult(output="", error="Timeout", run_id=run_id))
            raise

        result = _with_run_id(result, run_id)

        if on_step is not None:
            for step in result.steps:
                on_step(step)

        self._hook_manager.fire_run_end(result)
        return result

    async def stream(
        self,
        task: str,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Streaming execution yielding events as they happen."""
        result = await self.arun(task, **kwargs)

        for step in result.steps:
            if step.action.type == ActionType.TOOL_CALL and step.tool_name:
                yield ToolCallEvent(
                    name=step.tool_name,
                    args=dict(step.tool_args) if step.tool_args else {},
                )
                yield ToolResultEvent(
                    name=step.tool_name,
                    output=step.tool_result or "",
                )

        if result.output:
            yield TextDeltaEvent(delta=result.output)

        yield DoneEvent(result=result)

    async def _execute_with_timeout(
        self,
        task: str,
        *,
        run_id: str,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Run the agent with optional timeout enforcement."""
        coro = self.agent.arun(task, run_id=run_id, **kwargs)
        if timeout is not None:
            return await asyncio.wait_for(coro, timeout=timeout)
        return await coro


def _with_run_id(result: AgentResult, run_id: str) -> AgentResult:
    """Return a new AgentResult with the given run_id."""
    if result.run_id == run_id:
        return result
    return AgentResult(
        output=result.output,
        steps=result.steps,
        state=result.state,
        cost=result.cost,
        token_usage=result.token_usage,
        partial=result.partial,
        error=result.error,
        run_id=run_id,
    )


async def run(agent: AgentLike, task: str, **kwargs: Any) -> AgentResult:
    """Convenience: run an agent on a task."""
    runner = Runner(agent)
    return await runner.arun(task, **kwargs)
