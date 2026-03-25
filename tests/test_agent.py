"""Tests for the Agent class — the core ReAct loop."""

from __future__ import annotations

from typing import Any

import pytest

from pop.agent import Agent
from pop.types import (
    ActionType,
    Message,
    ModelResponse,
    TokenUsage,
    ToolCall,
    ToolDefinition,
)

# ---------------------------------------------------------------------------
# Mock adapter
# ---------------------------------------------------------------------------


class MockAdapter:
    """A configurable mock ModelAdapter for testing."""

    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = list(responses)
        self._call_index = 0
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
    ) -> ModelResponse:
        self.calls.append({"messages": messages, "tools": tools})
        if self._call_index >= len(self._responses):
            raise RuntimeError("MockAdapter exhausted — no more responses")
        response = self._responses[self._call_index]
        self._call_index += 1
        return response

    async def chat_stream(self, messages: Any, tools: Any = None):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Helper: simple tool definitions
# ---------------------------------------------------------------------------


def _add_tool() -> ToolDefinition:
    def add(a: int, b: int) -> str:
        return str(a + b)

    return ToolDefinition(
        name="add",
        description="Add two numbers",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        },
        function=add,
    )


def _failing_tool() -> ToolDefinition:
    def explode(msg: str) -> str:
        raise ValueError(f"Boom: {msg}")

    return ToolDefinition(
        name="explode",
        description="Always fails",
        parameters={
            "type": "object",
            "properties": {"msg": {"type": "string"}},
            "required": ["msg"],
        },
        function=explode,
    )


def _async_tool() -> ToolDefinition:
    async def greet(name: str) -> str:
        return f"Hello, {name}!"

    return ToolDefinition(
        name="greet",
        description="Greet someone",
        parameters={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
        function=greet,
        is_async=True,
    )


# ---------------------------------------------------------------------------
# 1. Simple final answer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_simple_final_answer():
    adapter = MockAdapter(
        [
            ModelResponse(content="The answer is 42.", token_usage=TokenUsage(10, 5)),
        ]
    )
    agent = Agent(model=adapter, max_steps=5)
    result = await agent.arun("What is the answer?")

    assert result.output == "The answer is 42."
    assert result.partial is False
    assert result.error is None
    assert len(result.steps) == 1


# ---------------------------------------------------------------------------
# 2. Single tool call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_tool_call():
    tool = _add_tool()
    adapter = MockAdapter(
        [
            ModelResponse(
                content="",
                tool_calls=(ToolCall(name="add", args={"a": 2, "b": 3}, call_id="c1"),),
                token_usage=TokenUsage(10, 5),
            ),
            ModelResponse(content="The sum is 5.", token_usage=TokenUsage(15, 8)),
        ]
    )
    agent = Agent(model=adapter, tools=[tool], max_steps=5)
    result = await agent.arun("Add 2 and 3")

    assert result.output == "The sum is 5."
    assert len(result.steps) == 2
    assert result.steps[0].tool_name == "add"
    assert result.steps[0].tool_result == "5"


# ---------------------------------------------------------------------------
# 3. Multi-step tool use
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multi_step_tool_use():
    tool = _add_tool()
    adapter = MockAdapter(
        [
            ModelResponse(
                content="",
                tool_calls=(ToolCall(name="add", args={"a": 1, "b": 2}, call_id="c1"),),
                token_usage=TokenUsage(10, 5),
            ),
            ModelResponse(
                content="",
                tool_calls=(ToolCall(name="add", args={"a": 3, "b": 4}, call_id="c2"),),
                token_usage=TokenUsage(10, 5),
            ),
            ModelResponse(content="Results: 3 and 7", token_usage=TokenUsage(15, 8)),
        ]
    )
    agent = Agent(model=adapter, tools=[tool], max_steps=5)
    result = await agent.arun("Add 1+2 then 3+4")

    assert result.output == "Results: 3 and 7"
    assert len(result.steps) == 3
    assert result.steps[0].tool_name == "add"
    assert result.steps[1].tool_name == "add"


# ---------------------------------------------------------------------------
# 4. Tool error recovery
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_error_recovery():
    tool = _failing_tool()
    adapter = MockAdapter(
        [
            ModelResponse(
                content="",
                tool_calls=(ToolCall(name="explode", args={"msg": "test"}, call_id="c1"),),
                token_usage=TokenUsage(10, 5),
            ),
            ModelResponse(content="I handled the error gracefully.", token_usage=TokenUsage(15, 8)),
        ]
    )
    agent = Agent(model=adapter, tools=[tool], max_steps=5)
    result = await agent.arun("Try exploding")

    assert result.output == "I handled the error gracefully."
    assert result.steps[0].error is not None
    assert "Boom: test" in result.steps[0].error


# ---------------------------------------------------------------------------
# 5. Max steps budget
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_max_steps_budget():
    tool = _add_tool()
    # Model keeps calling tool forever
    responses = [
        ModelResponse(
            content="",
            tool_calls=(ToolCall(name="add", args={"a": 1, "b": 1}, call_id=f"c{i}"),),
            token_usage=TokenUsage(10, 5),
        )
        for i in range(20)
    ]
    adapter = MockAdapter(responses)
    agent = Agent(model=adapter, tools=[tool], max_steps=3)
    result = await agent.arun("Keep adding")

    assert result.partial is True
    assert len(result.steps) <= 3


# ---------------------------------------------------------------------------
# 6. Max cost budget
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_max_cost_budget():
    tool = _add_tool()
    responses = [
        ModelResponse(
            content="",
            tool_calls=(ToolCall(name="add", args={"a": 1, "b": 1}, call_id=f"c{i}"),),
            token_usage=TokenUsage(10000, 5000),
        )
        for i in range(20)
    ]
    adapter = MockAdapter(responses)
    # Set a very low cost budget; we need cost_usd > 0 per step.
    # The agent should track cost even if pricing is not set.
    # We patch cost estimation in the agent or rely on token-based estimation.
    # For now, we inject a very small max_cost and rely on the agent's cost tracking.
    agent = Agent(model=adapter, tools=[tool], max_steps=20, max_cost=0.001)
    result = await agent.arun("Keep adding")

    assert result.partial is True


# ---------------------------------------------------------------------------
# 7. Guardrail enforcement
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_guardrail_enforcement():
    call_count = 0

    def no_bad_words(output: str) -> bool:
        nonlocal call_count
        call_count += 1
        return "bad" not in output

    adapter = MockAdapter(
        [
            ModelResponse(content="This is bad output.", token_usage=TokenUsage(10, 5)),
            ModelResponse(content="This is clean output.", token_usage=TokenUsage(10, 5)),
        ]
    )
    agent = Agent(model=adapter, max_steps=5, output_guardrails=[no_bad_words])
    result = await agent.arun("Say something")

    assert result.output == "This is clean output."
    assert call_count == 2


# ---------------------------------------------------------------------------
# 8. System instructions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_system_instructions():
    adapter = MockAdapter(
        [
            ModelResponse(content="OK", token_usage=TokenUsage(10, 5)),
        ]
    )
    agent = Agent(model=adapter, instructions="You are a helpful pirate.")
    await agent.arun("Hello")

    # The first message sent to the adapter should be the system message
    messages_sent = adapter.calls[0]["messages"]
    system_messages = [m for m in messages_sent if m.role.value == "system"]
    assert any("helpful pirate" in m.content for m in system_messages)


# ---------------------------------------------------------------------------
# 9. Core memory in context
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_core_memory_in_context():
    adapter = MockAdapter(
        [
            ModelResponse(content="OK", token_usage=TokenUsage(10, 5)),
        ]
    )
    agent = Agent(
        model=adapter,
        core_memory={"user_name": "Alice", "preference": "concise"},
    )
    await agent.arun("Hello")

    messages_sent = adapter.calls[0]["messages"]
    system_messages = [m for m in messages_sent if m.role.value == "system"]
    system_text = " ".join(m.content for m in system_messages)
    assert "Alice" in system_text
    assert "concise" in system_text


# ---------------------------------------------------------------------------
# 10. Tool schema passed to model
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_schema_passed_to_model():
    tool = _add_tool()
    adapter = MockAdapter(
        [
            ModelResponse(content="Done", token_usage=TokenUsage(10, 5)),
        ]
    )
    agent = Agent(model=adapter, tools=[tool])
    await agent.arun("Hello")

    tools_sent = adapter.calls[0]["tools"]
    assert tools_sent is not None
    assert len(tools_sent) == 1
    assert tools_sent[0].name == "add"


# ---------------------------------------------------------------------------
# 11. Step records have correct fields
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_step_records():
    tool = _add_tool()
    adapter = MockAdapter(
        [
            ModelResponse(
                content="Let me add.",
                tool_calls=(ToolCall(name="add", args={"a": 5, "b": 7}, call_id="c1"),),
                token_usage=TokenUsage(10, 5),
            ),
            ModelResponse(content="12", token_usage=TokenUsage(15, 8)),
        ]
    )
    agent = Agent(model=adapter, tools=[tool], max_steps=5)
    result = await agent.arun("Add 5 + 7")

    step0 = result.steps[0]
    assert step0.index == 0
    assert step0.tool_name == "add"
    assert step0.tool_args == {"a": 5, "b": 7}
    assert step0.tool_result == "12"
    assert step0.timestamp is not None
    assert step0.token_usage == TokenUsage(10, 5)

    step1 = result.steps[1]
    assert step1.index == 1
    assert step1.action.type == ActionType.FINAL_ANSWER


# ---------------------------------------------------------------------------
# 12. AgentResult contains cost and token usage
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_result_cost_and_tokens():
    adapter = MockAdapter(
        [
            ModelResponse(
                content="",
                tool_calls=(ToolCall(name="add", args={"a": 1, "b": 2}, call_id="c1"),),
                token_usage=TokenUsage(100, 50),
            ),
            ModelResponse(content="3", token_usage=TokenUsage(200, 80)),
        ]
    )
    tool = _add_tool()
    agent = Agent(model=adapter, tools=[tool], max_steps=5)
    result = await agent.arun("Add")

    assert result.token_usage.input_tokens == 300
    assert result.token_usage.output_tokens == 130
    assert result.cost >= 0.0


# ---------------------------------------------------------------------------
# 13. Reflexion: self-critique on tool failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reflexion_on_failure():
    tool = _failing_tool()
    adapter = MockAdapter(
        [
            ModelResponse(
                content="",
                tool_calls=(ToolCall(name="explode", args={"msg": "x"}, call_id="c1"),),
                token_usage=TokenUsage(10, 5),
            ),
            ModelResponse(content="Recovered after reflection.", token_usage=TokenUsage(15, 8)),
        ]
    )
    agent = Agent(
        model=adapter,
        tools=[tool],
        max_steps=5,
        reflect_on_failure=True,
    )
    result = await agent.arun("Try it")

    assert result.output == "Recovered after reflection."
    # When reflect_on_failure is True, the error context should include
    # a reflexion prompt in the messages sent to the LLM
    second_call_messages = adapter.calls[1]["messages"]
    message_texts = " ".join(m.content for m in second_call_messages)
    assert "reflect" in message_texts.lower() or "error" in message_texts.lower()


# ---------------------------------------------------------------------------
# 14. Sync run() wrapper
# ---------------------------------------------------------------------------


def test_sync_run():
    adapter = MockAdapter(
        [
            ModelResponse(content="sync result", token_usage=TokenUsage(10, 5)),
        ]
    )
    agent = Agent(model=adapter, max_steps=5)
    result = agent.run("Hello")

    assert result.output == "sync result"


# ---------------------------------------------------------------------------
# 15. Async tool execution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_tool_execution():
    tool = _async_tool()
    adapter = MockAdapter(
        [
            ModelResponse(
                content="",
                tool_calls=(ToolCall(name="greet", args={"name": "World"}, call_id="c1"),),
                token_usage=TokenUsage(10, 5),
            ),
            ModelResponse(content="Greeting sent!", token_usage=TokenUsage(15, 8)),
        ]
    )
    agent = Agent(model=adapter, tools=[tool], max_steps=5)
    result = await agent.arun("Greet World")

    assert result.steps[0].tool_result == "Hello, World!"
    assert result.output == "Greeting sent!"


# ---------------------------------------------------------------------------
# 16. Unknown tool raises error (agent.py line 266)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_tool_error():
    """Calling a tool not in the registry produces an error step."""
    adapter = MockAdapter(
        [
            ModelResponse(
                content="",
                tool_calls=(ToolCall(name="nonexistent", args={}, call_id="c1"),),
                token_usage=TokenUsage(10, 5),
            ),
            ModelResponse(content="Handled error.", token_usage=TokenUsage(15, 8)),
        ]
    )
    agent = Agent(model=adapter, tools=[], max_steps=5)
    result = await agent.arun("Try unknown tool")

    assert result.output == "Handled error."
    assert result.steps[0].error is not None
    assert "Unknown tool" in result.steps[0].error


# ---------------------------------------------------------------------------
# 17. Model as string list (agent.py lines 94-97)
# ---------------------------------------------------------------------------


def test_model_string_list_init(monkeypatch):
    """Agent can be initialized with a list of model strings."""
    from unittest.mock import patch

    fake_adapter = MockAdapter([])
    with patch.object(Agent, "__init__", wraps=Agent.__init__) as _:
        pass

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    from pop.models.router import ModelRouter

    with patch.object(ModelRouter, "from_model_string", return_value=fake_adapter):
        agent = Agent(model=["openai:gpt-4o", "openai:gpt-4o-mini"], max_steps=1)
    assert agent._fallback_models == ["openai:gpt-4o", "openai:gpt-4o-mini"]


# ---------------------------------------------------------------------------
# 18. Model as single string (agent.py lines 91-93)
# ---------------------------------------------------------------------------


def test_model_string_init(monkeypatch):
    """Agent can be initialized with a single model string."""
    from unittest.mock import patch

    fake_adapter = MockAdapter([])
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    from pop.models.router import ModelRouter

    with patch.object(ModelRouter, "from_model_string", return_value=fake_adapter):
        agent = Agent(model="openai:gpt-4o", max_steps=1)
    assert agent._fallback_models == []


# ---------------------------------------------------------------------------
# 19. Partial result uses tool_result (agent.py line 327-329)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_partial_result_uses_tool_result():
    """When budget exceeded after tool call, partial result has tool_result."""
    tool = _add_tool()
    adapter = MockAdapter(
        [
            ModelResponse(
                content="",
                tool_calls=(ToolCall(name="add", args={"a": 1, "b": 1}, call_id="c1"),),
                token_usage=TokenUsage(10000, 5000),
            ),
        ]
    )
    agent = Agent(model=adapter, tools=[tool], max_steps=1)
    result = await agent.arun("Add")

    assert result.partial is True
    # The partial result should have the tool result from the last step
    assert result.output == "2"


# ---------------------------------------------------------------------------
# 20. Max cost budget reached before first step (agent.py line 136)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_max_cost_budget_pre_step():
    """Agent returns partial if cost >= max_cost before LLM call."""
    tool = _add_tool()
    # First step uses lots of tokens, pushing cost over budget for next iteration
    adapter = MockAdapter(
        [
            ModelResponse(
                content="",
                tool_calls=(ToolCall(name="add", args={"a": 1, "b": 1}, call_id="c1"),),
                token_usage=TokenUsage(100000, 50000),
            ),
            # This response should not be reached since cost exceeds budget
            ModelResponse(content="Should not reach.", token_usage=TokenUsage(10, 5)),
        ]
    )
    # Very low budget so that after first step the cost exceeds it
    agent = Agent(model=adapter, tools=[tool], max_steps=10, max_cost=0.0001)
    result = await agent.arun("Add")

    assert result.partial is True


# ---------------------------------------------------------------------------
# 21. Sync run from inside running event loop (agent.py lines 110-113)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_run_from_running_loop():
    """Agent.run() works from within a running event loop using thread pool."""
    adapter = MockAdapter(
        [
            ModelResponse(content="from thread pool", token_usage=TokenUsage(10, 5)),
        ]
    )
    agent = Agent(model=adapter, max_steps=5)
    # We're already in an async context, so run() should detect running loop
    result = agent.run("Hello")
    assert result.output == "from thread pool"


# ---------------------------------------------------------------------------
# 22. Tool call with no call_id uses step index (agent.py line 225)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_call_no_call_id():
    """When tool_call.call_id is empty, agent generates one from step index."""
    tool = _add_tool()
    adapter = MockAdapter(
        [
            ModelResponse(
                content="",
                tool_calls=(ToolCall(name="add", args={"a": 1, "b": 2}, call_id=""),),
                token_usage=TokenUsage(10, 5),
            ),
            ModelResponse(content="3", token_usage=TokenUsage(15, 8)),
        ]
    )
    agent = Agent(model=adapter, tools=[tool], max_steps=5)
    result = await agent.arun("Add")

    assert result.output == "3"


# ---------------------------------------------------------------------------
# 23. Agent has `name` property (defaults to "agent")
# ---------------------------------------------------------------------------


def test_agent_name_defaults_to_agent():
    """Agent.name defaults to 'agent' when not specified."""
    adapter = MockAdapter([])
    agent = Agent(model=adapter, max_steps=1)
    assert agent.name == "agent"


# ---------------------------------------------------------------------------
# 24. Agent accepts `name` parameter in __init__
# ---------------------------------------------------------------------------


def test_agent_name_custom():
    """Agent.name can be set via __init__ parameter."""
    adapter = MockAdapter([])
    agent = Agent(model=adapter, name="researcher", max_steps=1)
    assert agent.name == "researcher"


# ---------------------------------------------------------------------------
# 25. Agent has public `instructions` property
# ---------------------------------------------------------------------------


def test_agent_instructions_public():
    """Agent.instructions is accessible as a public attribute."""
    adapter = MockAdapter([])
    agent = Agent(model=adapter, instructions="Be helpful.", max_steps=1)
    assert agent.instructions == "Be helpful."


def test_agent_instructions_defaults_to_empty():
    """Agent.instructions defaults to empty string."""
    adapter = MockAdapter([])
    agent = Agent(model=adapter, max_steps=1)
    assert agent.instructions == ""


# ---------------------------------------------------------------------------
# 26. Agent satisfies AgentLike protocol for multi-agent patterns
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_works_with_pipeline():
    """Agent instances work directly in multi-agent pipeline pattern."""
    from pop.multi.patterns import pipeline

    adapter1 = MockAdapter([ModelResponse(content="step one done", token_usage=TokenUsage(10, 5))])
    adapter2 = MockAdapter([ModelResponse(content="step two done", token_usage=TokenUsage(10, 5))])
    agent1 = Agent(model=adapter1, name="first", instructions="Do step one", max_steps=5)
    agent2 = Agent(model=adapter2, name="second", instructions="Do step two", max_steps=5)

    result = await pipeline([agent1, agent2], "start task")

    assert result.output == "step two done"
    assert len(result.agent_results) == 2


@pytest.mark.asyncio
async def test_agent_works_with_orchestrate():
    """Agent instances work directly in orchestrate pattern (boss/worker)."""
    from pop.multi.patterns import orchestrate

    boss_adapter = MockAdapter(
        [ModelResponse(content="orchestrated result", token_usage=TokenUsage(10, 5))]
    )
    worker_adapter = MockAdapter([])

    boss = Agent(model=boss_adapter, name="boss", instructions="Orchestrate workers", max_steps=5)
    worker = Agent(model=worker_adapter, name="worker_a", instructions="Do subtask A", max_steps=5)

    result = await orchestrate(boss, [worker], "do the thing")

    assert result.output == "orchestrated result"
    # Verify boss received enriched task with worker descriptions
    messages_sent = boss_adapter.calls[0]["messages"]
    all_text = " ".join(m.content for m in messages_sent)
    assert "worker_a" in all_text
    assert "Do subtask A" in all_text


# ---------------------------------------------------------------------------
# C2: Fallback chain — Agent with model list uses chat_with_fallback
# ---------------------------------------------------------------------------


class FailingAdapter:
    """Adapter that always raises on chat()."""

    def __init__(self, error_msg: str = "primary down") -> None:
        self._error_msg = error_msg

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
    ) -> ModelResponse:
        raise ConnectionError(self._error_msg)

    async def chat_stream(self, messages: Any, tools: Any = None):
        raise NotImplementedError


@pytest.mark.asyncio
async def test_fallback_chain_tries_second_model():
    """When Agent gets model=list and primary fails, fallback model is used."""
    from unittest.mock import AsyncMock, patch

    fallback_response = ModelResponse(
        content="fallback answer", token_usage=TokenUsage(10, 5), model="mock:fallback"
    )

    mock_chat_with_fallback = AsyncMock(return_value=fallback_response)

    with patch("pop.agent.ModelRouter") as mock_router:
        router_instance = mock_router.return_value
        router_instance.from_model_string.return_value = FailingAdapter()
        router_instance.chat_with_fallback = mock_chat_with_fallback

        agent = Agent(model=["mock:primary", "mock:fallback"], max_steps=5)
        result = await agent.arun("Hello")

    assert result.output == "fallback answer"
    mock_chat_with_fallback.assert_called()
    # Verify the fallback chain was called with both model strings
    call_args = mock_chat_with_fallback.call_args
    assert call_args[1]["model_strings"] == ["mock:primary", "mock:fallback"]


@pytest.mark.asyncio
async def test_single_model_does_not_use_fallback():
    """When Agent gets a single adapter (not a list), no fallback is involved."""
    adapter = MockAdapter(
        [
            ModelResponse(content="direct answer", token_usage=TokenUsage(10, 5)),
        ]
    )
    agent = Agent(model=adapter, max_steps=5)
    result = await agent.arun("Hello")

    assert result.output == "direct answer"
    assert len(adapter.calls) == 1


@pytest.mark.asyncio
async def test_fallback_chain_all_fail():
    """When all models in the fallback chain fail, agent raises RuntimeError."""
    from unittest.mock import AsyncMock, patch

    mock_chat_with_fallback = AsyncMock(
        side_effect=RuntimeError("All models failed. Errors: mock:a: ConnectionError: down")
    )

    with patch("pop.agent.ModelRouter") as mock_router:
        router_instance = mock_router.return_value
        router_instance.from_model_string.return_value = FailingAdapter()
        router_instance.chat_with_fallback = mock_chat_with_fallback

        agent = Agent(model=["mock:a", "mock:b"], max_steps=5)

        with pytest.raises(RuntimeError, match="All models failed"):
            await agent.arun("Hello")


# ---------------------------------------------------------------------------
# M7: Parallel tool calls — agent handles multiple tool calls in one response
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parallel_tool_calls():
    """When model returns multiple tool_calls, all should be executed."""
    add = _add_tool()
    greet = _async_tool()
    adapter = MockAdapter(
        [
            ModelResponse(
                content="Let me do both.",
                tool_calls=(
                    ToolCall(name="add", args={"a": 10, "b": 20}, call_id="c1"),
                    ToolCall(name="greet", args={"name": "Alice"}, call_id="c2"),
                ),
                token_usage=TokenUsage(10, 5),
            ),
            ModelResponse(content="Done: 30 and Hello, Alice!", token_usage=TokenUsage(15, 8)),
        ]
    )
    agent = Agent(model=adapter, tools=[add, greet], max_steps=5)
    result = await agent.arun("Add 10+20 and greet Alice")

    assert result.output == "Done: 30 and Hello, Alice!"
    # Two tool call steps should be recorded, plus the final answer step
    tool_steps = [s for s in result.steps if s.tool_name is not None]
    assert len(tool_steps) == 2
    assert tool_steps[0].tool_name == "add"
    assert tool_steps[0].tool_result == "30"
    assert tool_steps[1].tool_name == "greet"
    assert tool_steps[1].tool_result == "Hello, Alice!"


@pytest.mark.asyncio
async def test_parallel_tool_calls_with_error():
    """When one of multiple parallel tool calls fails, others still execute."""
    add = _add_tool()
    fail = _failing_tool()
    adapter = MockAdapter(
        [
            ModelResponse(
                content="",
                tool_calls=(
                    ToolCall(name="add", args={"a": 1, "b": 2}, call_id="c1"),
                    ToolCall(name="explode", args={"msg": "boom"}, call_id="c2"),
                ),
                token_usage=TokenUsage(10, 5),
            ),
            ModelResponse(content="Handled errors.", token_usage=TokenUsage(15, 8)),
        ]
    )
    agent = Agent(model=adapter, tools=[add, fail], max_steps=5)
    result = await agent.arun("Do both")

    assert result.output == "Handled errors."
    tool_steps = [s for s in result.steps if s.tool_name is not None]
    assert len(tool_steps) == 2
    # First tool call succeeds
    assert tool_steps[0].tool_result == "3"
    assert tool_steps[0].error is None
    # Second tool call fails
    assert tool_steps[1].error is not None
    assert "Boom: boom" in tool_steps[1].error
