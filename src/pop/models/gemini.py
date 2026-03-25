"""Google Gemini adapter — uses httpx, not the google SDK."""

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


def messages_to_gemini(
    messages: list[Message],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Convert pop Messages to Gemini format.

    Returns (system_instruction, contents) since Gemini treats system as a
    top-level 'systemInstruction' parameter rather than a message role.
    """
    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []

    for msg in messages:
        if msg.role == Role.SYSTEM:
            system_parts = [*system_parts, msg.content]
            continue

        if msg.role == Role.USER:
            contents = [*contents, {"role": "user", "parts": [{"text": msg.content}]}]

        elif msg.role == Role.ASSISTANT:
            parts: list[dict[str, Any]] = []
            if msg.content:
                parts = [*parts, {"text": msg.content}]
            for tc in msg.tool_calls:
                parts = [
                    *parts,
                    {"functionCall": {"name": tc.name, "args": tc.args}},
                ]
            if not parts:
                parts = [{"text": ""}]
            contents = [*contents, {"role": "model", "parts": parts}]

        elif msg.role == Role.TOOL:
            contents = [
                *contents,
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": msg.name or msg.tool_call_id,
                                "response": {"result": msg.content},
                            }
                        }
                    ],
                },
            ]

    system_instruction: dict[str, Any] | None = None
    if system_parts:
        system_instruction = {"parts": [{"text": "\n".join(system_parts)}]}

    return system_instruction, contents


def tools_to_gemini(tools: list[ToolDefinition] | None) -> list[dict[str, Any]]:
    """Convert pop ToolDefinitions to Gemini tool format."""
    if not tools:
        return []
    return [
        {
            "functionDeclarations": [
                {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                }
                for t in tools
            ]
        }
    ]


def parse_gemini_response(raw: dict[str, Any]) -> ModelResponse:
    """Parse a Gemini API response dict into a ModelResponse."""
    candidates = raw.get("candidates", [])
    if not candidates:
        return ModelResponse()

    parts = candidates[0].get("content", {}).get("parts", [])

    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for part in parts:
        if "text" in part:
            text_parts = [*text_parts, part["text"]]
        elif "functionCall" in part:
            fc = part["functionCall"]
            tool_calls = [
                *tool_calls,
                ToolCall(
                    name=fc["name"],
                    args=fc.get("args", {}),
                    call_id=fc["name"],
                ),
            ]

    usage_data = raw.get("usageMetadata", {})
    token_usage = TokenUsage(
        input_tokens=usage_data.get("promptTokenCount", 0),
        output_tokens=usage_data.get("candidatesTokenCount", 0),
    )

    finish_reason = candidates[0].get("finishReason", "")

    return ModelResponse(
        content="".join(text_parts),
        tool_calls=tuple(tool_calls),
        token_usage=token_usage,
        model=raw.get("modelVersion", ""),
        finish_reason=finish_reason,
    )


# ---------------------------------------------------------------------------
# Adapter class
# ---------------------------------------------------------------------------


class GeminiAdapter:
    """Adapter for the Google Gemini API."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        api_key_env: str = "GEMINI_API_KEY",
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
        self._client = httpx.AsyncClient(
            headers={"Content-Type": "application/json"},
            timeout=120.0,
        )

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
    ) -> ModelResponse:
        system_instruction, contents = messages_to_gemini(messages)

        payload: dict[str, Any] = {"contents": contents}
        if system_instruction is not None:
            payload["systemInstruction"] = system_instruction
        gemini_tools = tools_to_gemini(tools)
        if gemini_tools:
            payload["tools"] = gemini_tools

        url = f"{self._base_url}/models/{self._model}:generateContent?key={self._api_key}"
        response = await self._client.post(url, json=payload)
        response.raise_for_status()
        return parse_gemini_response(response.json())

    async def chat_stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        system_instruction, contents = messages_to_gemini(messages)

        payload: dict[str, Any] = {"contents": contents}
        if system_instruction is not None:
            payload["systemInstruction"] = system_instruction
        gemini_tools = tools_to_gemini(tools)
        if gemini_tools:
            payload["tools"] = gemini_tools

        url = (
            f"{self._base_url}/models/{self._model}:streamGenerateContent"
            f"?alt=sse&key={self._api_key}"
        )

        async with self._client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = json.loads(line[6:])
                candidates = data.get("candidates", [])
                if not candidates:
                    continue
                parts = candidates[0].get("content", {}).get("parts", [])
                for part in parts:
                    if "text" in part:
                        yield StreamChunk(delta_content=part["text"])
                finish_reason = candidates[0].get("finishReason", "")
                if finish_reason and finish_reason != "STOP":
                    yield StreamChunk(finish_reason=finish_reason)
                elif finish_reason == "STOP":
                    yield StreamChunk(finish_reason="stop")
