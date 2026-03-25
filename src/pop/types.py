"""Core types for the pop agent framework.

All types are immutable (frozen dataclasses) to enable deterministic replay,
checkpoint/resume, and safe concurrent access. This is a deliberate design
choice — see FRAMEWORK_ARCHITECTURE.md Section 5.6 for rationale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class Role(str, Enum):
    """Message roles in the conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ActionType(str, Enum):
    """The type of action the agent decides to take."""

    TOOL_CALL = "tool_call"
    FINAL_ANSWER = "final_answer"
    ASK_HUMAN = "ask_human"


class Status(str, Enum):
    """Agent run status."""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"
    PAUSED = "paused"


class ErrorClass(str, Enum):
    """Classification of errors for recovery strategy selection."""

    TRANSIENT = "transient"
    VALIDATION = "validation"
    BUDGET = "budget"
    FATAL = "fatal"
    RATE_LIMIT = "rate_limit"
    AUTH = "auth"


@dataclass(frozen=True)
class ToolCall:
    """A tool call requested by the LLM."""

    name: str
    args: dict[str, Any]
    call_id: str = ""


@dataclass(frozen=True)
class Message:
    """A message in the conversation history.

    Immutable to ensure conversation history cannot be mutated in place.
    New messages are appended by creating new lists.
    """

    role: Role
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str = ""
    name: str = ""

    @staticmethod
    def system(content: str) -> Message:
        return Message(role=Role.SYSTEM, content=content)

    @staticmethod
    def user(content: str) -> Message:
        return Message(role=Role.USER, content=content)

    @staticmethod
    def assistant(content: str, tool_calls: tuple[ToolCall, ...] = ()) -> Message:
        return Message(role=Role.ASSISTANT, content=content, tool_calls=tool_calls)

    @staticmethod
    def tool_result(content: str, tool_call_id: str, name: str = "") -> Message:
        return Message(
            role=Role.TOOL,
            content=content,
            tool_call_id=tool_call_id,
            name=name,
        )


@dataclass(frozen=True)
class TokenUsage:
    """Token usage for a single LLM call or accumulated across a run."""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


@dataclass(frozen=True)
class Action:
    """The action decided by the agent at each step."""

    type: ActionType
    tool_call: ToolCall | None = None
    answer: str | None = None
    question: str | None = None


@dataclass(frozen=True)
class Step:
    """An immutable record of a single agent loop iteration.

    This is the bridge between runtime behavior and eval metrics.
    Every eval dimension maps to fields in this record:
    - Task accuracy → final step's action
    - Cost efficiency → cost_usd
    - Latency → latency_ms
    - Reliability → error + recovery_action
    - Tool calling accuracy → tool_name + tool_args
    """

    index: int
    timestamp: datetime
    thought: str | None = None
    action: Action = field(default_factory=lambda: Action(type=ActionType.FINAL_ANSWER))
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_result: str | None = None
    error: str | None = None
    recovery_action: str | None = None
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    model_used: str = ""


@dataclass(frozen=True)
class ToolDefinition:
    """A tool that an agent can use, with its JSON Schema for the LLM."""

    name: str
    description: str
    parameters: dict[str, Any]
    function: Any  # The actual callable
    is_async: bool = False


@dataclass(frozen=True)
class ModelResponse:
    """Normalized response from any LLM provider."""

    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    model: str = ""
    finish_reason: str = ""


@dataclass(frozen=True)
class StreamEvent:
    """Base class for streaming events."""


@dataclass(frozen=True)
class ThinkEvent(StreamEvent):
    """The agent is thinking (reasoning text delta)."""

    thought: str = ""


@dataclass(frozen=True)
class ToolCallEvent(StreamEvent):
    """The agent is calling a tool."""

    name: str = ""
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolResultEvent(StreamEvent):
    """A tool returned a result."""

    name: str = ""
    output: str = ""


@dataclass(frozen=True)
class TextDeltaEvent(StreamEvent):
    """A chunk of the agent's final text response."""

    delta: str = ""


@dataclass(frozen=True)
class DoneEvent(StreamEvent):
    """The agent run is complete."""

    result: AgentResult | None = None


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class AgentState:
    """The complete state of an agent run at a point in time.

    Immutable snapshots enable checkpoint, replay, and forking.
    Each step creates a new AgentState rather than mutating the existing one.
    """

    messages: tuple[Message, ...] = ()
    tool_results: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    step_count: int = 0
    status: Status = Status.PENDING
    cost_usd: float = 0.0
    token_usage: TokenUsage = field(default_factory=TokenUsage)

    def with_message(self, message: Message) -> AgentState:
        """Return a new state with an appended message."""
        return AgentState(
            messages=(*self.messages, message),
            tool_results=self.tool_results,
            metadata=self.metadata,
            step_count=self.step_count,
            status=self.status,
            cost_usd=self.cost_usd,
            token_usage=self.token_usage,
        )

    def with_step(
        self,
        *,
        step_cost: float = 0.0,
        step_tokens: TokenUsage | None = None,
    ) -> AgentState:
        """Return a new state with incremented step count and accumulated cost."""
        return AgentState(
            messages=self.messages,
            tool_results=self.tool_results,
            metadata=self.metadata,
            step_count=self.step_count + 1,
            status=Status.RUNNING,
            cost_usd=self.cost_usd + step_cost,
            token_usage=self.token_usage + (step_tokens or TokenUsage()),
        )

    def with_status(self, status: Status) -> AgentState:
        """Return a new state with updated status."""
        return AgentState(
            messages=self.messages,
            tool_results=self.tool_results,
            metadata=self.metadata,
            step_count=self.step_count,
            status=status,
            cost_usd=self.cost_usd,
            token_usage=self.token_usage,
        )


@dataclass(frozen=True)
class AgentResult:
    """The final result of an agent run.

    Contains the output, full step trace, state, and cost/token metrics.
    This is what agent.run() returns to the caller.
    """

    output: str
    steps: tuple[Step, ...] = ()
    state: AgentState = field(default_factory=AgentState)
    cost: float = 0.0
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    partial: bool = False
    error: str | None = None
    run_id: str = ""


@dataclass(frozen=True)
class AgentError:
    """A structured error with classification and actionable message."""

    message: str
    error_class: ErrorClass
    suggestion: str = ""
    original_error: Exception | None = None

    @property
    def is_rate_limit(self) -> bool:
        return self.error_class == ErrorClass.RATE_LIMIT

    @property
    def is_tool_error(self) -> bool:
        return self.error_class == ErrorClass.TRANSIENT

    @property
    def is_fatal(self) -> bool:
        return self.error_class == ErrorClass.FATAL
