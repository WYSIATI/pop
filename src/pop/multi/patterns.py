"""Multi-agent composition patterns: pipeline, orchestrate, debate, fan_out.

Higher-level patterns that compose multiple agents into collaborative workflows.
All functions are async and return immutable result types.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from pop.types import AgentResult


class AgentLike(Protocol):
    """Minimal protocol for agents used in multi-agent patterns."""

    name: str
    instructions: str

    async def arun(self, task: str, **kwargs: Any) -> AgentResult: ...


# ── Result types ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PipelineResult:
    """Result of a sequential agent pipeline."""

    output: str
    agent_results: tuple[AgentResult, ...]


@dataclass(frozen=True)
class DebateResult:
    """Result of a generator-critic debate loop."""

    output: str
    rounds: int
    approved: bool
    history: tuple[str, ...]  # alternating generator/critic outputs


@dataclass(frozen=True)
class FanOutResult:
    """Result of parallel agent execution."""

    output: str
    agent_outputs: tuple[str, ...]
    strategy: str


# ── Pattern implementations ─────────────────────────────────────────────


async def pipeline(
    agents: list[AgentLike],
    task: str,
) -> PipelineResult:
    """Run agents sequentially, passing each output as the next agent's task.

    Args:
        agents: Ordered list of agents to run in sequence.
        task: The initial task for the first agent.

    Returns:
        PipelineResult with the final output and all intermediate results.

    Raises:
        ValueError: If agents list is empty.
    """
    if not agents:
        raise ValueError("Pipeline requires at least one agent")

    current_task = task
    results: list[AgentResult] = []

    for agent in agents:
        result = await agent.arun(current_task)
        results = [*results, result]
        current_task = result.output

    return PipelineResult(
        output=results[-1].output,
        agent_results=tuple(results),
    )


async def debate(
    generator: AgentLike,
    critic: AgentLike,
    task: str,
    max_rounds: int = 3,
) -> DebateResult:
    """Generator proposes solutions, critic reviews until approved or max rounds.

    The critic's response is checked for "APPROVED" (case-insensitive).
    If not approved, the generator receives the critic's feedback for the next round.

    Args:
        generator: Agent that proposes solutions.
        critic: Agent that reviews and approves/rejects.
        task: The initial task.
        max_rounds: Maximum number of generate-critique rounds.

    Returns:
        DebateResult with final output, round count, and approval status.
    """
    history: list[str] = []
    current_task = task
    approved = False
    last_generation = ""
    completed_rounds = 0

    for _ in range(max_rounds):
        completed_rounds += 1
        gen_result = await generator.arun(current_task)
        last_generation = gen_result.output
        history = [*history, last_generation]

        critic_prompt = (
            f"Review this solution for the task: {task}\n\n"
            f"Solution:\n{last_generation}\n\n"
            f"Respond with APPROVED if acceptable, or REJECTED with feedback."
        )
        critic_result = await critic.arun(critic_prompt)
        critic_output = critic_result.output
        history = [*history, critic_output]

        if "APPROVED" in critic_output.upper():
            approved = True
            break

        current_task = (
            f"Original task: {task}\n\n"
            f"Your previous attempt:\n{last_generation}\n\n"
            f"Feedback:\n{critic_output}\n\n"
            f"Please improve your solution."
        )

    return DebateResult(
        output=last_generation,
        rounds=completed_rounds,
        approved=approved,
        history=tuple(history),
    )


async def fan_out(
    agents: list[AgentLike],
    task: str,
    strategy: str = "merge",
) -> FanOutResult:
    """Run all agents on the same task in parallel.

    Args:
        agents: Agents to run concurrently.
        task: The task for all agents.
        strategy: "merge" to concatenate outputs, "vote" to pick most common.

    Returns:
        FanOutResult with combined output and individual agent outputs.

    Raises:
        ValueError: If agents list is empty or strategy is unknown.
    """
    if not agents:
        raise ValueError("fan_out requires at least one agent")

    if strategy not in ("merge", "vote"):
        raise ValueError(f"Unknown strategy: '{strategy}'. Use 'merge' or 'vote'.")

    results = await asyncio.gather(*[agent.arun(task) for agent in agents])
    outputs = tuple(r.output for r in results)

    if strategy == "merge":
        combined = "\n\n".join(outputs)
    else:
        counter = Counter(outputs)
        combined = counter.most_common(1)[0][0]

    return FanOutResult(
        output=combined,
        agent_outputs=outputs,
        strategy=strategy,
    )


async def orchestrate(
    boss: AgentLike,
    workers: list[AgentLike],
    task: str,
) -> AgentResult:
    """Boss agent delegates subtasks to worker agents.

    The boss agent is given tools representing each worker. When the boss
    calls a worker tool, that worker runs the subtask and returns its result.

    For simplicity in testing with MockAgent (which ignores tools), this
    implementation runs the boss directly. A real boss Agent with LLM
    backing would use the worker tools during its agent loop.

    Args:
        boss: The orchestrating agent that decomposes and integrates.
        workers: Worker agents available for delegation.
        task: The top-level task to accomplish.

    Returns:
        The boss agent's final AgentResult.
    """
    worker_descriptions = "\n".join(f"- {w.name}: {w.instructions}" for w in workers)

    enriched_task = (
        f"{task}\n\n"
        f"Available workers:\n{worker_descriptions}\n\n"
        f"Delegate subtasks to workers and integrate their results."
    )

    return await boss.arun(enriched_task)
