"""pop -- Fast, lean AI agents. 5 lines to production."""

from __future__ import annotations

__version__ = "1.2.0"

__all__ = [
    # Core
    "Agent",
    "tool",
    "Runner",
    "run",
    # Multi-agent
    "handoff",
    "pipeline",
    "orchestrate",
    "debate",
    "fan_out",
    # Workflows
    "chain",
    "route",
    "parallel",
    # Models
    "chat",
    "model",
    "register_provider",
    # Types
    "AgentResult",
    "Step",
    "TokenUsage",
    "Message",
    # Stream events
    "ThinkEvent",
    "ToolCallEvent",
    "ToolResultEvent",
    "TextDeltaEvent",
    "DoneEvent",
]

# Lazy import mapping: attribute name -> (module, name)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Core
    "Agent": ("pop.agent", "Agent"),
    "tool": ("pop.tool", "tool"),
    "Runner": ("pop.runner", "Runner"),
    "run": ("pop.runner", "run"),
    # Multi-agent
    "handoff": ("pop.multi", "handoff"),
    "pipeline": ("pop.multi", "pipeline"),
    "orchestrate": ("pop.multi", "orchestrate"),
    "debate": ("pop.multi", "debate"),
    "fan_out": ("pop.multi", "fan_out"),
    # Workflows
    "chain": ("pop.workflows", "chain"),
    "route": ("pop.workflows", "route"),
    "parallel": ("pop.workflows", "parallel"),
    # Models
    "chat": ("pop.models", "chat"),
    "model": ("pop.models", "model"),
    "register_provider": ("pop.models", "register_provider"),
    # Types
    "AgentResult": ("pop.types", "AgentResult"),
    "Step": ("pop.types", "Step"),
    "TokenUsage": ("pop.types", "TokenUsage"),
    "Message": ("pop.types", "Message"),
    # Stream events
    "ThinkEvent": ("pop.types", "ThinkEvent"),
    "ToolCallEvent": ("pop.types", "ToolCallEvent"),
    "ToolResultEvent": ("pop.types", "ToolResultEvent"),
    "TextDeltaEvent": ("pop.types", "TextDeltaEvent"),
    "DoneEvent": ("pop.types", "DoneEvent"),
}


def __getattr__(name: str) -> object:
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        import importlib

        module = importlib.import_module(module_path)
        value = getattr(module, attr_name)
        # Cache on the module to avoid repeated lookups
        globals()[name] = value
        return value
    raise AttributeError(f"module 'pop' has no attribute {name!r}")
