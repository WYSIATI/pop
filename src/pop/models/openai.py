"""OpenAI-compatible adapter — uses httpx, not the openai SDK."""

from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator

import httpx

from pop.models.base import ModelAdapter, StreamChunk
from pop.types import Message, ModelResponse, Role, ToolCall, ToolDefinition, TokenUsage


# ---------------------------------------------------------------------------
# Conversion helpers (pure functions, easy to test)
# ---------------------------------------------------------------------------

def messages_to_openai(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert pop Messages to OpenAI chat message dicts."""
    result: list[dict[str, Any]] = []
    for msg in messages:
        entry: dict[str, Any] = {"role": msg.role.value, "content": msg.content}

        if msg.role == Role.ASSISTANT and msg.tool_calls:
            entry["tool_calls"] = [
                {
                    "id": tc.call_id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.args),
                    },
                }
                for tc in msg.tool_calls
            ]

        if msg.role == Role.TOOL:
            entry = {
                "role": "tool",
                "content": msg.content,
                "tool_call_id": msg.tool_call_id,
            }

        result = [*result, entry]
    return result


def tools_to_openai(tools: list[ToolDefinition] | None) -> list[dict[str, Any]]:
    """Convert pop ToolDefinitions to OpenAI tool format."""
    if not tools:
        return []
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in tools
    ]


def parse_openai_response(raw: dict[str, Any]) -> ModelResponse:
    """Parse an OpenAI API response dict into a ModelResponse."""
    choice = raw["choices"][0]
    message = choice["message"]

    content = message.get("content") or ""

    tool_calls: tuple[ToolCall, ...] = ()
    raw_tool_calls = message.get("tool_calls")
    if raw_tool_calls:
        tool_calls = tuple(
            ToolCall(
                name=tc["function"]["name"],
                args=json.loads(tc["function"]["arguments"]),
                call_id=tc["id"],
            )
            for tc in raw_tool_calls
        )

    usage_data = raw.get("usage", {})
    token_usage = TokenUsage(
        input_tokens=usage_data.get("prompt_tokens", 0),
        output_tokens=usage_data.get("completion_tokens", 0),
    )

    return ModelResponse(
        content=content,
        tool_calls=tool_calls,
        token_usage=token_usage,
        model=raw.get("model", ""),
        finish_reason=choice.get("finish_reason", ""),
    )


# ---------------------------------------------------------------------------
# Adapter class
# ---------------------------------------------------------------------------

class OpenAIAdapter:
    """Adapter for OpenAI-compatible APIs."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        api_key_env: str = "OPENAI_API_KEY",
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
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
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages_to_openai(messages),
        }
        openai_tools = tools_to_openai(tools)
        if openai_tools:
            payload["tools"] = openai_tools

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=120.0,
            )
            response.raise_for_status()
            return parse_openai_response(response.json())

    async def chat_stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages_to_openai(messages),
            "stream": True,
        }
        openai_tools = tools_to_openai(tools)
        if openai_tools:
            payload["tools"] = openai_tools

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=120.0,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    chunk = json.loads(data)
                    delta = chunk["choices"][0].get("delta", {})
                    yield StreamChunk(
                        delta_content=delta.get("content", ""),
                        finish_reason=chunk["choices"][0].get("finish_reason", "") or "",
                    )
