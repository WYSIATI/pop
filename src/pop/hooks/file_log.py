"""FileLogHook — writes JSON Lines to a file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pop.types import AgentResult, Step


class FileLogHook:
    """Appends JSON Lines to a file for structured logging.

    Creates the file (and parent directories) if they don't exist.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def on_step(self, step: Step) -> None:
        record = {
            "event": "step",
            "index": step.index,
            "tool_name": step.tool_name,
            "latency_ms": step.latency_ms,
            "cost_usd": step.cost_usd,
            "error": step.error,
        }
        self._append(record)

    def on_run_end(self, result: AgentResult) -> None:
        record = {
            "event": "run_end",
            "output": result.output,
            "cost_usd": result.cost,
            "total_tokens": result.token_usage.total,
            "step_count": len(result.steps),
        }
        self._append(record)

    def _append(self, record: dict[str, object]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
