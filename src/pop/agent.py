"""Agent class — the core ReAct loop with optional Reflexion.

Implements the Reasoning + Acting pattern: the LLM reasons about the task,
decides on an action (tool call, final answer, or ask human), and the loop
continues until completion or a budget is exceeded.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from pop.hooks.base import Hook, HookManager
from pop.models.router import ModelRouter
from pop.types import (
    Action,
    ActionType,
    AgentResult,
    AgentState,
    Message,
    ModelResponse,
    Status,
    Step,
    TokenUsage,
    ToolCall,
    ToolDefinition,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from pop.memory.base import MemoryBackend
    from pop.models.base import ModelAdapter

# Rough cost estimation per token (fallback when provider doesn't report cost)
_DEFAULT_COST_PER_INPUT_TOKEN = 0.000003
_DEFAULT_COST_PER_OUTPUT_TOKEN = 0.000015


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _estimate_cost(usage: TokenUsage) -> float:
    return (
        usage.input_tokens * _DEFAULT_COST_PER_INPUT_TOKEN
        + usage.output_tokens * _DEFAULT_COST_PER_OUTPUT_TOKEN
    )


class Agent:
    """The core ReAct loop agent with optional Reflexion."""

    def __init__(
        self,
        model: str | list[str] | ModelAdapter,
        tools: list[ToolDefinition] | None = None,
        instructions: str = "",
        memory: MemoryBackend | None = None,
        hooks: list[Hook] | None = None,
        max_steps: int = 10,
        max_cost: float | None = None,
        max_retries: int = 3,
        reflect_on_failure: bool = False,
        output_guardrails: list[Callable[[str], bool]] | None = None,
        confirm_before: list[str] | None = None,
        core_memory: dict[str, str] | None = None,
        conversation_window: int = 20,
        planning_model: str | None = None,
    ) -> None:
        self._tools: tuple[ToolDefinition, ...] = tuple(tools) if tools else ()
        self._tool_map: dict[str, ToolDefinition] = {t.name: t for t in self._tools}
        self._instructions = instructions
        self._memory = memory
        self._hook_manager = HookManager(hooks)
        self._max_steps = max_steps
        self._max_cost = max_cost
        self._max_retries = max_retries
        self._reflect_on_failure = reflect_on_failure
        self._output_guardrails: tuple[Callable[[str], bool], ...] = (
            tuple(output_guardrails) if output_guardrails else ()
        )
        self._confirm_before: tuple[str, ...] = tuple(confirm_before) if confirm_before else ()
        self._core_memory: dict[str, str] = dict(core_memory) if core_memory else {}
        self._conversation_window = conversation_window
        self._planning_model = planning_model

        # Resolve model adapter
        if isinstance(model, str):
            self._adapter: ModelAdapter = ModelRouter().from_model_string(model)
            self._fallback_models: list[str] = []
        elif isinstance(model, list):
            router = ModelRouter()
            self._adapter = router.from_model_string(model[0])
            self._fallback_models = model
        else:
            self._adapter = model
            self._fallback_models = []

    def run(self, task: str, **kwargs: Any) -> AgentResult:
        """Synchronous wrapper around arun()."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, self.arun(task, **kwargs))
                return future.result()

        return asyncio.run(self.arun(task, **kwargs))

    async def arun(self, task: str, **kwargs: Any) -> AgentResult:
        """Async execution of the agent loop."""
        run_id = kwargs.get("run_id", str(uuid.uuid4()))
        state = AgentState(status=Status.RUNNING)
        state = state.with_message(Message.user(task))

        self._hook_manager.fire_run_start(task, run_id)

        result = await self._loop(state, run_id)

        self._hook_manager.fire_run_end(result)
        return result

    async def _loop(self, state: AgentState, run_id: str) -> AgentResult:
        """The core ReAct loop."""
        steps: list[Step] = []

        while state.step_count < self._max_steps:
            if self._max_cost is not None and state.cost_usd >= self._max_cost:
                return self._build_partial_result(state, steps, run_id)

            messages = self._build_messages(state)
            tools_for_model = list(self._tools) if self._tools else None

            response = await self._adapter.chat(messages, tools_for_model)

            step_cost = _estimate_cost(response.token_usage)
            state = state.with_step(
                step_cost=step_cost,
                step_tokens=response.token_usage,
            )

            if self._max_cost is not None and state.cost_usd > self._max_cost:
                step = self._make_step(
                    index=len(steps),
                    response=response,
                    action=Action(type=ActionType.FINAL_ANSWER, answer=response.content),
                )
                steps = [*steps, step]
                self._hook_manager.fire_step(step)
                return self._build_partial_result(state, steps, run_id)

            # Determine action from response
            if response.tool_calls:
                tool_call = response.tool_calls[0]
                state = state.with_message(
                    Message.assistant(response.content, tool_calls=response.tool_calls)
                )
                step, state = await self._handle_tool_call(tool_call, response, state, len(steps))
                steps = [*steps, step]
                self._hook_manager.fire_step(step)
            else:
                # Final answer candidate
                action = Action(type=ActionType.FINAL_ANSWER, answer=response.content)
                step = self._make_step(index=len(steps), response=response, action=action)

                if not self._check_guardrails(response.content):
                    # Guardrail failed — add feedback and continue
                    state = state.with_message(Message.assistant(response.content))
                    state = state.with_message(
                        Message.user(
                            "Your previous output was rejected by a guardrail. "
                            "Please try again with a different response."
                        )
                    )
                    steps = [*steps, step]
                    self._hook_manager.fire_step(step)
                    continue

                state = state.with_message(Message.assistant(response.content))
                state = state.with_status(Status.DONE)
                steps = [*steps, step]
                self._hook_manager.fire_step(step)

                return AgentResult(
                    output=response.content,
                    steps=tuple(steps),
                    state=state,
                    cost=state.cost_usd,
                    token_usage=state.token_usage,
                    run_id=run_id,
                )

        return self._build_partial_result(state, steps, run_id)

    async def _handle_tool_call(
        self,
        tool_call: ToolCall,
        response: ModelResponse,
        state: AgentState,
        step_index: int,
    ) -> tuple[Step, AgentState]:
        """Execute a tool call and return the step and updated state."""
        tool_result_str: str | None = None
        error_str: str | None = None

        try:
            tool_result_str = await self._execute_tool(tool_call)
        except Exception as exc:
            error_str = f"Tool '{tool_call.name}' error: {exc}"
            tool_result_str = error_str

        call_id = tool_call.call_id or f"call_{step_index}"
        state = state.with_message(
            Message.tool_result(
                content=tool_result_str,
                tool_call_id=call_id,
                name=tool_call.name,
            )
        )

        if error_str and self._reflect_on_failure:
            state = state.with_message(
                Message.user(
                    "The tool call failed. Please reflect on the error above "
                    "and decide how to proceed."
                )
            )

        action = Action(
            type=ActionType.TOOL_CALL,
            tool_call=tool_call,
        )
        step = Step(
            index=step_index,
            timestamp=_now(),
            thought=response.content or None,
            action=action,
            tool_name=tool_call.name,
            tool_args=dict(tool_call.args),
            tool_result=tool_result_str,
            error=error_str,
            token_usage=response.token_usage,
            cost_usd=_estimate_cost(response.token_usage),
            model_used=response.model,
        )

        return step, state

    async def _execute_tool(self, tool_call: ToolCall) -> str:
        """Find tool, execute, and return result as string."""
        tool_def = self._tool_map.get(tool_call.name)
        if tool_def is None:
            raise ValueError(f"Unknown tool: '{tool_call.name}'")

        if tool_def.is_async:
            result = await tool_def.function(**tool_call.args)
        else:
            result = tool_def.function(**tool_call.args)

        return str(result)

    def _build_messages(self, state: AgentState) -> list[Message]:
        """Assemble the message list for the LLM."""
        messages: list[Message] = []

        system_parts: list[str] = []
        if self._instructions:
            system_parts.append(self._instructions)
        if self._core_memory:
            core_lines = [f"- {k}: {v}" for k, v in self._core_memory.items()]
            system_parts.append("Core memory:\n" + "\n".join(core_lines))

        if system_parts:
            messages.append(Message.system("\n\n".join(system_parts)))

        messages.extend(state.messages)
        return messages

    def _check_guardrails(self, output: str) -> bool:
        """Run all guardrail functions. Returns True if all pass."""
        return all(guardrail(output) for guardrail in self._output_guardrails)

    def _make_step(
        self,
        index: int,
        response: ModelResponse,
        action: Action,
    ) -> Step:
        """Create a Step record from a model response."""
        return Step(
            index=index,
            timestamp=_now(),
            thought=response.content or None,
            action=action,
            token_usage=response.token_usage,
            cost_usd=_estimate_cost(response.token_usage),
            model_used=response.model,
        )

    def _build_partial_result(
        self,
        state: AgentState,
        steps: list[Step],
        run_id: str,
    ) -> AgentResult:
        """Build a partial result when budget is exceeded."""
        last_output = ""
        if steps:
            last_step = steps[-1]
            if last_step.action.answer:
                last_output = last_step.action.answer
            elif last_step.tool_result:
                last_output = last_step.tool_result

        return AgentResult(
            output=last_output,
            steps=tuple(steps),
            state=state.with_status(Status.PAUSED),
            cost=state.cost_usd,
            token_usage=state.token_usage,
            partial=True,
            run_id=run_id,
        )
