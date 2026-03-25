"""Tests for the pop hook system."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from pop.hooks import ConsoleHook, CostTrackingHook, FileLogHook, HookManager
from pop.types import (
    Action,
    ActionType,
    AgentResult,
    Step,
    TokenUsage,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_step(
    index: int = 0,
    tool_name: str | None = None,
    tool_args: dict | None = None,
    tool_result: str | None = None,
    error: str | None = None,
    cost_usd: float = 0.0,
    token_usage: TokenUsage | None = None,
    latency_ms: float = 100.0,
    action_type: ActionType = ActionType.TOOL_CALL,
) -> Step:
    return Step(
        index=index,
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        action=Action(type=action_type),
        tool_name=tool_name,
        tool_args=tool_args,
        tool_result=tool_result,
        error=error,
        cost_usd=cost_usd,
        token_usage=token_usage or TokenUsage(),
        latency_ms=latency_ms,
    )


def _make_result(
    output: str = "done",
    steps: tuple[Step, ...] = (),
    cost: float = 0.0,
    token_usage: TokenUsage | None = None,
) -> AgentResult:
    return AgentResult(
        output=output,
        steps=steps,
        cost=cost,
        token_usage=token_usage or TokenUsage(),
    )


# ---------------------------------------------------------------------------
# HookManager — no hooks (zero overhead)
# ---------------------------------------------------------------------------


class TestHookManagerEmpty:
    def test_fire_run_start_no_hooks(self) -> None:
        manager = HookManager()
        manager.fire_run_start("some task", "run-1")  # should not raise

    def test_fire_step_no_hooks(self) -> None:
        manager = HookManager()
        manager.fire_step(_make_step())  # should not raise

    def test_fire_run_end_no_hooks(self) -> None:
        manager = HookManager()
        manager.fire_run_end(_make_result())  # should not raise


# ---------------------------------------------------------------------------
# HookManager — fires events to all hooks
# ---------------------------------------------------------------------------


class _RecordingHook:
    """A hook that records every call for assertions."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []

    def on_run_start(self, task: str, run_id: str = "") -> None:
        self.calls.append(("on_run_start", (task, run_id)))

    def on_step(self, step: Step) -> None:
        self.calls.append(("on_step", (step,)))

    def on_run_end(self, result: AgentResult) -> None:
        self.calls.append(("on_run_end", (result,)))


class TestHookManagerFiring:
    def test_fires_run_start_to_all(self) -> None:
        h1, h2 = _RecordingHook(), _RecordingHook()
        manager = HookManager(hooks=[h1, h2])
        manager.fire_run_start("task-a", "run-1")
        assert h1.calls == [("on_run_start", ("task-a", "run-1"))]
        assert h2.calls == [("on_run_start", ("task-a", "run-1"))]

    def test_fires_step_to_all(self) -> None:
        h1, h2 = _RecordingHook(), _RecordingHook()
        manager = HookManager(hooks=[h1, h2])
        step = _make_step(index=3)
        manager.fire_step(step)
        assert h1.calls == [("on_step", (step,))]
        assert h2.calls == [("on_step", (step,))]

    def test_fires_run_end_to_all(self) -> None:
        h1, h2 = _RecordingHook(), _RecordingHook()
        manager = HookManager(hooks=[h1, h2])
        result = _make_result()
        manager.fire_run_end(result)
        assert h1.calls == [("on_run_end", (result,))]
        assert h2.calls == [("on_run_end", (result,))]

    def test_fires_in_registration_order(self) -> None:
        order: list[int] = []

        class _OrderedHook:
            def __init__(self, idx: int) -> None:
                self._idx = idx

            def on_run_start(self, task: str, run_id: str = "") -> None:
                order.append(self._idx)

        manager = HookManager(hooks=[_OrderedHook(1), _OrderedHook(2), _OrderedHook(3)])
        manager.fire_run_start("t", "r")
        assert order == [1, 2, 3]


# ---------------------------------------------------------------------------
# Partial hooks (only some methods implemented)
# ---------------------------------------------------------------------------


class _StepOnlyHook:
    """A hook that only implements on_step."""

    def __init__(self) -> None:
        self.steps: list[Step] = []

    def on_step(self, step: Step) -> None:
        self.steps.append(step)


class TestPartialHook:
    def test_partial_hook_skips_missing_methods(self) -> None:
        hook = _StepOnlyHook()
        manager = HookManager(hooks=[hook])
        # These should not raise even though the hook lacks on_run_start/on_run_end
        manager.fire_run_start("task", "run-1")
        step = _make_step(index=1)
        manager.fire_step(step)
        manager.fire_run_end(_make_result())
        assert hook.steps == [step]


# ---------------------------------------------------------------------------
# ConsoleHook
# ---------------------------------------------------------------------------


class TestConsoleHook:
    def test_on_run_start(self, capsys: pytest.CaptureFixture[str]) -> None:
        hook = ConsoleHook()
        hook.on_run_start("summarize docs", "run-42")
        captured = capsys.readouterr()
        assert "Starting agent run: summarize docs" in captured.err
        assert captured.out == ""

    def test_on_step_tool_call(self, capsys: pytest.CaptureFixture[str]) -> None:
        hook = ConsoleHook()
        step = _make_step(
            index=1,
            tool_name="search",
            tool_args={"query": "hello"},
            tool_result="found 3 results",
            action_type=ActionType.TOOL_CALL,
        )
        hook.on_step(step)
        captured = capsys.readouterr()
        assert "Step 1: calling search" in captured.err
        assert "found 3 results" in captured.err

    def test_on_step_final_answer(self, capsys: pytest.CaptureFixture[str]) -> None:
        hook = ConsoleHook()
        step = _make_step(index=2, action_type=ActionType.FINAL_ANSWER)
        hook.on_step(step)
        captured = capsys.readouterr()
        assert "Step 2: final answer" in captured.err

    def test_on_step_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        hook = ConsoleHook()
        step = _make_step(index=3, error="timeout")
        hook.on_step(step)
        captured = capsys.readouterr()
        assert "Step 3: ERROR" in captured.err
        assert "timeout" in captured.err

    def test_on_run_end(self, capsys: pytest.CaptureFixture[str]) -> None:
        hook = ConsoleHook()
        steps = (_make_step(index=0), _make_step(index=1))
        result = _make_result(
            steps=steps,
            cost=0.0042,
            token_usage=TokenUsage(input_tokens=500, output_tokens=200),
        )
        hook.on_run_end(result)
        captured = capsys.readouterr()
        assert "2 steps" in captured.err
        assert "$0.0042" in captured.err
        assert "700 tokens" in captured.err

    def test_on_step_truncates_long_result(self, capsys: pytest.CaptureFixture[str]) -> None:
        hook = ConsoleHook()
        long_result = "x" * 200
        step = _make_step(
            index=0,
            tool_name="read",
            tool_args={},
            tool_result=long_result,
            action_type=ActionType.TOOL_CALL,
        )
        hook.on_step(step)
        captured = capsys.readouterr()
        # Result should be truncated to 100 chars
        assert len("x" * 100) == 100
        assert ("x" * 101) not in captured.err

    def test_on_step_ask_human_action(self, capsys: pytest.CaptureFixture[str]) -> None:
        """ASK_HUMAN action type hits the fallback branch (line 42)."""
        hook = ConsoleHook()
        step = _make_step(index=4, action_type=ActionType.ASK_HUMAN)
        hook.on_step(step)
        captured = capsys.readouterr()
        assert "Step 4: ask_human" in captured.err


# ---------------------------------------------------------------------------
# CostTrackingHook
# ---------------------------------------------------------------------------


class TestCostTrackingHook:
    def test_accumulates_cost(self) -> None:
        hook = CostTrackingHook()
        hook.on_step(_make_step(cost_usd=0.01, token_usage=TokenUsage(100, 50)))
        hook.on_step(_make_step(cost_usd=0.02, token_usage=TokenUsage(200, 100)))
        assert hook.total_cost == pytest.approx(0.03)
        assert hook.total_tokens == 450
        assert hook.step_count == 2

    def test_warns_at_80_percent_budget(self, capsys: pytest.CaptureFixture[str]) -> None:
        hook = CostTrackingHook(budget=1.0)
        hook.on_step(_make_step(cost_usd=0.81))
        captured = capsys.readouterr()
        assert "warning" in captured.err.lower() or "Warning" in captured.err

    def test_no_warning_below_80_percent(self, capsys: pytest.CaptureFixture[str]) -> None:
        hook = CostTrackingHook(budget=1.0)
        hook.on_step(_make_step(cost_usd=0.5))
        captured = capsys.readouterr()
        assert "warning" not in captured.err.lower()

    def test_no_warning_without_budget(self, capsys: pytest.CaptureFixture[str]) -> None:
        hook = CostTrackingHook()
        hook.on_step(_make_step(cost_usd=999.0))
        captured = capsys.readouterr()
        assert "warning" not in captured.err.lower()

    def test_on_run_end_logs_summary(self, capsys: pytest.CaptureFixture[str]) -> None:
        hook = CostTrackingHook()
        hook.on_step(_make_step(cost_usd=0.05, token_usage=TokenUsage(100, 50)))
        result = _make_result()
        hook.on_run_end(result)
        captured = capsys.readouterr()
        assert "$0.0500" in captured.err or "0.05" in captured.err


# ---------------------------------------------------------------------------
# FileLogHook
# ---------------------------------------------------------------------------


class TestFileLogHook:
    def test_writes_step_as_json_line(self, tmp_path: Path) -> None:
        log_file = tmp_path / "log.jsonl"
        hook = FileLogHook(log_file)
        step = _make_step(
            index=0,
            tool_name="search",
            latency_ms=42.5,
            cost_usd=0.001,
        )
        hook.on_step(step)
        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["index"] == 0
        assert data["tool_name"] == "search"
        assert data["latency_ms"] == 42.5
        assert data["cost_usd"] == 0.001
        assert data["error"] is None

    def test_appends_multiple_steps(self, tmp_path: Path) -> None:
        log_file = tmp_path / "log.jsonl"
        hook = FileLogHook(log_file)
        hook.on_step(_make_step(index=0))
        hook.on_step(_make_step(index=1))
        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_writes_summary_on_run_end(self, tmp_path: Path) -> None:
        log_file = tmp_path / "log.jsonl"
        hook = FileLogHook(log_file)
        result = _make_result(
            output="all done",
            cost=0.05,
            token_usage=TokenUsage(500, 200),
        )
        hook.on_run_end(result)
        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["event"] == "run_end"
        assert data["cost_usd"] == 0.05

    def test_creates_file_if_not_exists(self, tmp_path: Path) -> None:
        log_file = tmp_path / "subdir" / "log.jsonl"
        hook = FileLogHook(log_file)
        hook.on_step(_make_step(index=0))
        assert log_file.exists()

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        log_file = str(tmp_path / "log.jsonl")
        hook = FileLogHook(log_file)
        hook.on_step(_make_step(index=0))
        assert Path(log_file).exists()
