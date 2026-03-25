"""Hook system for opt-in middleware."""

from __future__ import annotations

from pop.hooks.base import Hook, HookManager

__all__ = [
    "ConsoleHook",
    "CostTrackingHook",
    "FileLogHook",
    "Hook",
    "HookManager",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "ConsoleHook": ("pop.hooks.console", "ConsoleHook"),
    "CostTrackingHook": ("pop.hooks.cost", "CostTrackingHook"),
    "FileLogHook": ("pop.hooks.file_log", "FileLogHook"),
}


def __getattr__(name: str) -> object:
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        import importlib

        module = importlib.import_module(module_path)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'pop.hooks' has no attribute {name!r}")
