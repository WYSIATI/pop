"""ConsoleHook — pretty-prints agent activity to stderr."""

from __future__ import annotations

import sys

from pop.types import ActionType, AgentResult, Step


class ConsoleHook:
    """Prints human-readable agent activity to stderr.

    Uses stderr so agent stdout output remains clean for piping.
    """

    def on_run_start(self, task: str, run_id: str = "") -> None:
        print(f"Starting agent run: {task}", file=sys.stderr)

    def on_step(self, step: Step) -> None:
        if step.error:
            print(
                f"Step {step.index}: ERROR — {step.error}",
                file=sys.stderr,
            )
            return

        if step.action.type == ActionType.TOOL_CALL and step.tool_name:
            args_str = str(step.tool_args or {})
            print(
                f"Step {step.index}: calling {step.tool_name}({args_str})",
                file=sys.stderr,
            )
            if step.tool_result is not None:
                truncated = step.tool_result[:100]
                print(f"  → {truncated}", file=sys.stderr)
            return

        if step.action.type == ActionType.FINAL_ANSWER:
            print(f"Step {step.index}: final answer", file=sys.stderr)
            return

        print(f"Step {step.index}: {step.action.type.value}", file=sys.stderr)

    def on_run_end(self, result: AgentResult) -> None:
        step_count = len(result.steps)
        total_tokens = result.token_usage.total
        print(
            f"Done! {step_count} steps, ${result.cost:.4f}, {total_tokens} tokens",
            file=sys.stderr,
        )
