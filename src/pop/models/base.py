"""Base protocol and types for model adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Protocol

from pop.types import Message, ModelResponse, ToolCall, ToolDefinition, TokenUsage


@dataclass(frozen=True)
class StreamChunk:
    """A single chunk from a streaming LLM response."""

    delta_content: str = ""
    delta_tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str = ""
    usage: TokenUsage | None = None


class ModelAdapter(Protocol):
    """Protocol that all LLM provider adapters must satisfy."""

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
    ) -> ModelResponse: ...

    async def chat_stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
    ) -> AsyncIterator[StreamChunk]: ...
