"""pop -- Fast, lean AI agents. 5 lines to production."""

__version__ = "0.1.0"

# Core
from pop.agent import Agent
from pop.tool import tool
from pop.runner import Runner, run

# Multi-agent
from pop.multi import handoff, pipeline, orchestrate, debate, fan_out

# Workflows
from pop.workflows import chain, route, parallel

# Models
from pop.models import chat, model, register_provider

# Types (commonly used)
from pop.types import AgentResult, Step, TokenUsage, Message

# Stream events
from pop.types import ThinkEvent, ToolCallEvent, ToolResultEvent, TextDeltaEvent, DoneEvent

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
