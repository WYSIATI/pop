"""Agent-to-agent handoff tool creation.

Creates a ToolDefinition that, when invoked by an LLM, transfers execution
to another agent and returns its output as the tool result.
"""

from __future__ import annotations

from typing import Any, Protocol

from pop.types import AgentResult, ToolDefinition


class AgentLike(Protocol):
    """Minimal protocol for an agent that can be handed off to."""

    name: str
    instructions: str

    async def arun(self, task: str, **kwargs: Any) -> AgentResult: ...


def handoff(agent: AgentLike, when: str = "") -> ToolDefinition:
    """Create a handoff tool that transfers execution to another agent.

    Args:
        agent: The agent to hand off to.
        when: Description of when to use this handoff (used in tool description).

    Returns:
        A ToolDefinition that, when called by the LLM, runs the target agent.
    """
    tool_name = f"handoff_to_{agent.name}"

    if when:
        description = f"Hand off to {agent.name}: use when {when}"
    else:
        description = f"Hand off to {agent.name} for further processing"

    async def _run_handoff(task: str) -> str:
        result = await agent.arun(task)
        return result.output

    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "The task or context to pass to the target agent",
            },
        },
        "required": ["task"],
    }

    return ToolDefinition(
        name=tool_name,
        description=description,
        parameters=parameters,
        function=_run_handoff,
        is_async=True,
    )
