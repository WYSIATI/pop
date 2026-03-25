"""Tests for multi-agent orchestration patterns."""

from __future__ import annotations

from typing import Callable

import pytest

from pop.types import AgentResult


class MockAgent:
    """A mock agent that returns a fixed output."""

    def __init__(self, output: str, name: str = "mock") -> None:
        self._output = output
        self.name = name
        self.instructions = f"I am {name}"
        self.call_count = 0

    async def arun(self, task: str, **kwargs: object) -> AgentResult:
        self.call_count += 1
        return AgentResult(output=self._output)

    def run(self, task: str, **kwargs: object) -> AgentResult:
        self.call_count += 1
        return AgentResult(output=self._output)


class TransformAgent:
    """A mock agent that transforms the input task."""

    def __init__(self, transform: Callable[[str], str], name: str = "transform") -> None:
        self._transform = transform
        self.name = name
        self.instructions = f"I am {name}"

    async def arun(self, task: str, **kwargs: object) -> AgentResult:
        return AgentResult(output=self._transform(task))

    def run(self, task: str, **kwargs: object) -> AgentResult:
        return AgentResult(output=self._transform(task))


# ── pipeline tests ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pipeline_sequential() -> None:
    """Three agents in sequence: output flows through each."""
    from pop.multi.patterns import pipeline

    agents = [
        TransformAgent(lambda t: f"[A]{t}", name="a"),
        TransformAgent(lambda t: f"[B]{t}", name="b"),
        TransformAgent(lambda t: f"[C]{t}", name="c"),
    ]
    result = await pipeline(agents, "start")

    assert result.output == "[C][B][A]start"


@pytest.mark.asyncio
async def test_pipeline_single_agent() -> None:
    """Single-agent pipeline degenerates to simple agent run."""
    from pop.multi.patterns import pipeline

    agent = MockAgent("done", name="solo")
    result = await pipeline([agent], "task")

    assert result.output == "done"
    assert len(result.agent_results) == 1


@pytest.mark.asyncio
async def test_pipeline_preserves_results() -> None:
    """PipelineResult contains all intermediate agent results."""
    from pop.multi.patterns import pipeline

    agents = [
        MockAgent("first", name="a"),
        MockAgent("second", name="b"),
        MockAgent("third", name="c"),
    ]
    result = await pipeline(agents, "task")

    assert len(result.agent_results) == 3
    assert result.agent_results[0].output == "first"
    assert result.agent_results[1].output == "second"
    assert result.agent_results[2].output == "third"
    assert result.output == "third"


# ── debate tests ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_debate_approved_first_round() -> None:
    """Critic approves on the first round."""
    from pop.multi.patterns import debate

    generator = MockAgent("my solution", name="gen")
    critic = MockAgent("APPROVED", name="critic")
    result = await debate(generator, critic, "solve this")

    assert result.approved is True
    assert result.rounds == 1
    assert result.output == "my solution"


@pytest.mark.asyncio
async def test_debate_multiple_rounds() -> None:
    """Critic rejects twice, approves on third round."""
    from pop.multi.patterns import debate

    round_counter = {"gen": 0, "critic": 0}

    class RoundAwareGenerator:
        name = "gen"
        instructions = "I generate"

        async def arun(self, task: str, **kwargs: object) -> AgentResult:
            round_counter["gen"] += 1
            return AgentResult(output=f"solution_v{round_counter['gen']}")

    class RoundAwareCritic:
        name = "critic"
        instructions = "I critique"

        async def arun(self, task: str, **kwargs: object) -> AgentResult:
            round_counter["critic"] += 1
            if round_counter["critic"] < 3:
                return AgentResult(output="REJECTED: needs improvement")
            return AgentResult(output="APPROVED")

    result = await debate(RoundAwareGenerator(), RoundAwareCritic(), "solve this")

    assert result.approved is True
    assert result.rounds == 3
    assert result.output == "solution_v3"


@pytest.mark.asyncio
async def test_debate_max_rounds() -> None:
    """Reaches max_rounds without approval."""
    from pop.multi.patterns import debate

    generator = MockAgent("attempt", name="gen")
    critic = MockAgent("REJECTED: not good enough", name="critic")
    result = await debate(generator, critic, "solve this", max_rounds=2)

    assert result.approved is False
    assert result.rounds == 2
    assert len(result.history) == 4  # gen, critic, gen, critic


# ── fan_out tests ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fan_out_merge_strategy() -> None:
    """Merge strategy concatenates all outputs."""
    from pop.multi.patterns import fan_out

    agents = [
        MockAgent("alpha", name="a"),
        MockAgent("beta", name="b"),
        MockAgent("gamma", name="c"),
    ]
    result = await fan_out(agents, "task", strategy="merge")

    assert "alpha" in result.output
    assert "beta" in result.output
    assert "gamma" in result.output
    assert result.strategy == "merge"
    assert len(result.agent_outputs) == 3


@pytest.mark.asyncio
async def test_fan_out_vote_strategy() -> None:
    """Vote strategy selects the most common output."""
    from pop.multi.patterns import fan_out

    agents = [
        MockAgent("winner", name="a"),
        MockAgent("winner", name="b"),
        MockAgent("loser", name="c"),
    ]
    result = await fan_out(agents, "task", strategy="vote")

    assert result.output == "winner"
    assert result.strategy == "vote"


@pytest.mark.asyncio
async def test_fan_out_parallel_execution() -> None:
    """All agents are called when fan_out runs."""
    from pop.multi.patterns import fan_out

    agents = [
        MockAgent("a", name="a"),
        MockAgent("b", name="b"),
        MockAgent("c", name="c"),
    ]
    result = await fan_out(agents, "task")

    for agent in agents:
        assert agent.call_count == 1
    assert len(result.agent_outputs) == 3


# ── orchestrate tests ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_orchestrate_boss_delegates_to_workers() -> None:
    """Boss agent receives worker tools and can delegate."""
    from pop.multi.patterns import orchestrate

    worker_a = MockAgent("worker_a_result", name="worker_a")
    worker_b = MockAgent("worker_b_result", name="worker_b")
    boss = MockAgent("final answer from boss", name="boss")

    result = await orchestrate(boss, [worker_a, worker_b], "do the thing")

    assert result.output == "final answer from boss"


# ── error case tests ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pipeline_empty_raises() -> None:
    """Empty pipeline raises ValueError."""
    from pop.multi.patterns import pipeline

    with pytest.raises(ValueError, match="at least one agent"):
        await pipeline([], "task")


@pytest.mark.asyncio
async def test_fan_out_empty_raises() -> None:
    """Empty fan_out raises ValueError."""
    from pop.multi.patterns import fan_out

    with pytest.raises(ValueError, match="at least one agent"):
        await fan_out([], "task")


@pytest.mark.asyncio
async def test_fan_out_unknown_strategy_raises() -> None:
    """Unknown strategy raises ValueError."""
    from pop.multi.patterns import fan_out

    agents = [MockAgent("test", name="a")]
    with pytest.raises(ValueError, match="Unknown strategy"):
        await fan_out(agents, "task", strategy="invalid")
