"""Hook base class and manager for opt-in agent middleware."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pop.types import AgentResult, Step


class Hook:
    """Base class for agent lifecycle hooks.

    Hooks are opt-in: override only the methods you need.
    Default implementations are no-ops, so subclasses don't need
    to implement methods they don't care about.

    Previous design used a Protocol with hasattr checks, but since
    Protocol declares all methods, hasattr was always True for valid
    implementors — making the checks redundant. A base class with
    no-op defaults is simpler and actually achieves opt-in behavior.
    """

    def on_run_start(self, task: str, run_id: str = "") -> None:
        """Called when an agent run starts."""

    def on_step(self, step: Step) -> None:
        """Called after each agent step completes."""

    def on_run_end(self, result: AgentResult) -> None:
        """Called when an agent run finishes."""


class HookManager:
    """Dispatches lifecycle events to registered hooks.

    Zero hooks registered means zero overhead — all fire_* methods
    short-circuit immediately when the hooks tuple is empty.
    """

    def __init__(self, hooks: list[Hook] | None = None) -> None:
        self._hooks: tuple[Hook, ...] = tuple(hooks) if hooks else ()

    def fire_run_start(self, task: str, run_id: str = "") -> None:
        if not self._hooks:
            return
        for hook in self._hooks:
            method = getattr(hook, "on_run_start", None)
            if method is not None:
                method(task, run_id)

    def fire_step(self, step: Step) -> None:
        if not self._hooks:
            return
        for hook in self._hooks:
            method = getattr(hook, "on_step", None)
            if method is not None:
                method(step)

    def fire_run_end(self, result: AgentResult) -> None:
        if not self._hooks:
            return
        for hook in self._hooks:
            method = getattr(hook, "on_run_end", None)
            if method is not None:
                method(result)
