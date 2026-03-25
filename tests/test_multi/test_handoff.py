"""Tests for the handoff function."""

from __future__ import annotations

import pytest

from pop.types import AgentResult, ToolDefinition


class MockAgent:
    """A mock agent for handoff testing."""

    def __init__(self, output: str, name: str = "mock") -> None:
        self._output = output
        self.name = name
        self.instructions = f"I am {name}"
        self.last_task: str | None = None

    async def arun(self, task: str, **kwargs: object) -> AgentResult:
        self.last_task = task
        return AgentResult(output=self._output)

    def run(self, task: str, **kwargs: object) -> AgentResult:
        self.last_task = task
        return AgentResult(output=self._output)


@pytest.mark.asyncio
async def test_handoff_creates_tool_definition() -> None:
    """handoff returns a valid ToolDefinition."""
    from pop.multi.handoff import handoff

    agent = MockAgent("result", name="helper")
    tool = handoff(agent)

    assert isinstance(tool, ToolDefinition)
    assert "helper" in tool.name


@pytest.mark.asyncio
async def test_handoff_tool_name_format() -> None:
    """Tool name follows handoff_to_{name} convention."""
    from pop.multi.handoff import handoff

    agent = MockAgent("result", name="researcher")
    tool = handoff(agent)

    assert tool.name == "handoff_to_researcher"


@pytest.mark.asyncio
async def test_handoff_description_includes_when() -> None:
    """Tool description incorporates the when parameter."""
    from pop.multi.handoff import handoff

    agent = MockAgent("result", name="helper")
    tool = handoff(agent, when="the user asks about math")

    assert "math" in tool.description


@pytest.mark.asyncio
async def test_handoff_tool_runs_target_agent() -> None:
    """Calling the handoff tool function runs the target agent."""
    from pop.multi.handoff import handoff

    agent = MockAgent("agent output", name="helper")
    tool = handoff(agent)

    result = await tool.function(task="do something")

    assert result == "agent output"
    assert agent.last_task == "do something"


@pytest.mark.asyncio
async def test_handoff_default_description() -> None:
    """Without when param, description still makes sense."""
    from pop.multi.handoff import handoff

    agent = MockAgent("result", name="helper")
    tool = handoff(agent)

    assert tool.description
    assert "helper" in tool.description.lower() or "handoff" in tool.description.lower()
