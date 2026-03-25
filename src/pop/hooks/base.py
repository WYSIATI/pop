"""Hook protocol and manager for opt-in agent middleware."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pop.types import AgentResult, Step


@runtime_checkable
class Hook(Protocol):
    """Protocol for agent lifecycle hooks.

    Hooks are opt-in: implement only the methods you need.
    """

    def on_run_start(self, task: str, run_id: str = "") -> None: ...

    def on_step(self, step: Step) -> None: ...

    def on_run_end(self, result: AgentResult) -> None: ...


class HookManager:
    """Dispatches lifecycle events to registered hooks.

    Zero hooks registered means zero overhead — all fire_* methods
    short-circuit immediately when the hooks list is empty.
    """

    def __init__(self, hooks: list[Hook] | None = None) -> None:
        self._hooks: tuple[Hook, ...] = tuple(hooks) if hooks else ()

    def fire_run_start(self, task: str, run_id: str = "") -> None:
        if not self._hooks:
            return
        for hook in self._hooks:
            if hasattr(hook, "on_run_start"):
                hook.on_run_start(task, run_id)

    def fire_step(self, step: Step) -> None:
        if not self._hooks:
            return
        for hook in self._hooks:
            if hasattr(hook, "on_step"):
                hook.on_step(step)

    def fire_run_end(self, result: AgentResult) -> None:
        if not self._hooks:
            return
        for hook in self._hooks:
            if hasattr(hook, "on_run_end"):
                hook.on_run_end(result)
