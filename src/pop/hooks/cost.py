"""CostTrackingHook — accumulates cost and warns on budget."""

from __future__ import annotations

import sys

from pop.types import AgentResult, Step, TokenUsage


class CostTrackingHook:
    """Tracks cumulative cost and token usage across an agent run.

    Optionally warns when spending exceeds 80% of a given budget.
    """

    def __init__(self, budget: float | None = None) -> None:
        self._budget = budget
        self._total_cost: float = 0.0
        self._total_tokens: TokenUsage = TokenUsage()
        self._step_count: int = 0

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def total_tokens(self) -> int:
        return self._total_tokens.total

    @property
    def step_count(self) -> int:
        return self._step_count

    def on_step(self, step: Step) -> None:
        self._total_cost = self._total_cost + step.cost_usd
        self._total_tokens = self._total_tokens + step.token_usage
        self._step_count = self._step_count + 1

        if self._budget is not None and self._total_cost >= self._budget * 0.8:
            print(
                f"Warning: cost ${self._total_cost:.4f} has reached "
                f"{self._total_cost / self._budget * 100:.0f}% of "
                f"${self._budget:.4f} budget",
                file=sys.stderr,
            )

    def on_run_end(self, result: AgentResult) -> None:
        print(
            f"Cost summary: ${self._total_cost:.4f}, "
            f"{self._total_tokens.total} tokens, "
            f"{self._step_count} steps",
            file=sys.stderr,
        )
