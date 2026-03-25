"""Anthropic adapter — uses httpx, not the anthropic SDK."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

import httpx

from pop.models.base import StreamChunk
from pop.types import Message, ModelResponse, Role, TokenUsage, ToolCall, ToolDefinition

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

# ---------------------------------------------------------------------------
# Conversion helpers (pure functions, easy to test)
# ---------------------------------------------------------------------------


def messages_to_anthropic(
    messages: list[Message],
) -> tuple[str, list[dict[str, Any]]]:
    """Convert pop Messages to Anthropic format.

    Returns (system_prompt, messages) since Anthropic treats system as a
    top-level parameter rather than a message role.
    """
    system_parts: list[str] = []
    result: list[dict[str, Any]] = []

    for msg in messages:
        if msg.role == Role.SYSTEM:
            system_parts = [*system_parts, msg.content]
            continue

        if msg.role == Role.USER:
            result = [*result, {"role": "user", "content": msg.content}]

        elif msg.role == Role.ASSISTANT:
            if msg.tool_calls:
                content_blocks: list[dict[str, Any]] = []
                if msg.content:
                    content_blocks = [*content_blocks, {"type": "text", "text": msg.content}]
                for tc in msg.tool_calls:
                    content_blocks = [
                        *content_blocks,
                        {
                            "type": "tool_use",
                            "id": tc.call_id,
                            "name": tc.name,
                            "input": tc.args,
                        },
                    ]
                result = [*result, {"role": "assistant", "content": content_blocks}]
            else:
                result = [*result, {"role": "assistant", "content": msg.content}]

        elif msg.role == Role.TOOL:
            result = [
                *result,
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id,
                            "content": msg.content,
                        }
                    ],
                },
            ]

    system_prompt = "\n".join(system_parts)
    return system_prompt, result


def tools_to_anthropic(tools: list[ToolDefinition] | None) -> list[dict[str, Any]]:
    """Convert pop ToolDefinitions to Anthropic tool format."""
    if not tools:
        return []
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.parameters,
        }
        for t in tools
    ]


def parse_anthropic_response(raw: dict[str, Any]) -> ModelResponse:
    """Parse an Anthropic API response dict into a ModelResponse."""
    content_blocks = raw.get("content", [])

    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for block in content_blocks:
        if block["type"] == "text":
            text_parts = [*text_parts, block["text"]]
        elif block["type"] == "tool_use":
            tool_calls = [
                *tool_calls,
                ToolCall(
                    name=block["name"],
                    args=block.get("input", {}),
                    call_id=block["id"],
                ),
            ]

    usage_data = raw.get("usage", {})
    token_usage = TokenUsage(
        input_tokens=usage_data.get("input_tokens", 0),
        output_tokens=usage_data.get("output_tokens", 0),
    )

    return ModelResponse(
        content="".join(text_parts),
        tool_calls=tuple(tool_calls),
        token_usage=token_usage,
        model=raw.get("model", ""),
        finish_reason=raw.get("stop_reason", ""),
    )


# ---------------------------------------------------------------------------
# Adapter class
# ---------------------------------------------------------------------------


class AnthropicAdapter:
    """Adapter for the Anthropic Messages API."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str = "https://api.anthropic.com/v1",
        api_key_env: str = "ANTHROPIC_API_KEY",
        max_tokens: int = 4096,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._max_tokens = max_tokens
        resolved_key = api_key or os.environ.get(api_key_env, "")
        if not resolved_key:
            raise ValueError(
                f"No API key provided. Set {api_key_env} environment variable "
                f"or pass api_key explicitly."
            )
        self._api_key = resolved_key

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
    ) -> ModelResponse:
        system_prompt, anthropic_messages = messages_to_anthropic(messages)

        payload: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": anthropic_messages,
        }
        if system_prompt:
            payload["system"] = system_prompt
        anthropic_tools = tools_to_anthropic(tools)
        if anthropic_tools:
            payload["tools"] = anthropic_tools

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._base_url}/messages",
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=120.0,
            )
            response.raise_for_status()
            return parse_anthropic_response(response.json())

    async def chat_stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        system_prompt, anthropic_messages = messages_to_anthropic(messages)

        payload: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": anthropic_messages,
            "stream": True,
        }
        if system_prompt:
            payload["system"] = system_prompt
        anthropic_tools = tools_to_anthropic(tools)
        if anthropic_tools:
            payload["tools"] = anthropic_tools

        async with (
            httpx.AsyncClient() as client,
            client.stream(
                "POST",
                f"{self._base_url}/messages",
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=120.0,
            ) as response,
        ):
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = json.loads(line[6:])
                event_type = data.get("type", "")
                if event_type == "content_block_delta":
                    delta = data.get("delta", {})
                    if delta.get("type") == "text_delta":
                        yield StreamChunk(delta_content=delta.get("text", ""))
                elif event_type == "message_delta":
                    yield StreamChunk(
                        finish_reason=data.get("delta", {}).get("stop_reason", ""),
                    )
                elif event_type == "message_stop":
                    break
