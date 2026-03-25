"""Tests for OpenAI adapter — conversion logic only, no real API calls."""

from __future__ import annotations

import pytest

from pop.types import Message, Role, ToolCall, ToolDefinition, TokenUsage, ModelResponse
from pop.models.openai import (
    messages_to_openai,
    tools_to_openai,
    parse_openai_response,
)


# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------

class TestMessagesToOpenAI:
    def test_user_message(self) -> None:
        msgs = [Message.user("hello")]
        result = messages_to_openai(msgs)
        assert result == [{"role": "user", "content": "hello"}]

    def test_system_message(self) -> None:
        msgs = [Message.system("you are helpful")]
        result = messages_to_openai(msgs)
        assert result == [{"role": "system", "content": "you are helpful"}]

    def test_assistant_message_plain(self) -> None:
        msgs = [Message.assistant("hi there")]
        result = messages_to_openai(msgs)
        assert result == [{"role": "assistant", "content": "hi there"}]

    def test_assistant_with_tool_calls(self) -> None:
        tc = ToolCall(name="get_weather", args={"city": "SF"}, call_id="tc_1")
        msgs = [Message.assistant("", tool_calls=(tc,))]
        result = messages_to_openai(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert len(result[0]["tool_calls"]) == 1
        assert result[0]["tool_calls"][0]["id"] == "tc_1"
        assert result[0]["tool_calls"][0]["type"] == "function"
        assert result[0]["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_tool_result_message(self) -> None:
        msgs = [Message.tool_result("sunny", tool_call_id="tc_1", name="get_weather")]
        result = messages_to_openai(msgs)
        assert result == [
            {"role": "tool", "content": "sunny", "tool_call_id": "tc_1"}
        ]

    def test_multi_turn_conversation(self) -> None:
        msgs = [
            Message.system("be helpful"),
            Message.user("hi"),
            Message.assistant("hello!"),
        ]
        result = messages_to_openai(msgs)
        assert len(result) == 3
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"


# ---------------------------------------------------------------------------
# Tool definition conversion
# ---------------------------------------------------------------------------

class TestToolsToOpenAI:
    def test_single_tool(self) -> None:
        tool = ToolDefinition(
            name="get_weather",
            description="Get weather for a city",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
            function=lambda x: x,
        )
        result = tools_to_openai([tool])
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_weather"
        assert result[0]["function"]["description"] == "Get weather for a city"
        assert result[0]["function"]["parameters"]["type"] == "object"

    def test_empty_tools(self) -> None:
        result = tools_to_openai([])
        assert result == []

    def test_none_tools(self) -> None:
        result = tools_to_openai(None)
        assert result == []


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

class TestParseOpenAIResponse:
    def test_text_response(self) -> None:
        raw = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        resp = parse_openai_response(raw)
        assert resp.content == "Hello!"
        assert resp.tool_calls == ()
        assert resp.model == "gpt-4o"
        assert resp.finish_reason == "stop"
        assert resp.token_usage.input_tokens == 10
        assert resp.token_usage.output_tokens == 5

    def test_tool_call_response(self) -> None:
        raw = {
            "id": "chatcmpl-456",
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "SF"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 10,
                "total_tokens": 30,
            },
        }
        resp = parse_openai_response(raw)
        assert resp.content == ""
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "get_weather"
        assert resp.tool_calls[0].args == {"city": "SF"}
        assert resp.tool_calls[0].call_id == "call_abc"
        assert resp.finish_reason == "tool_calls"

    def test_multiple_tool_calls(self) -> None:
        raw = {
            "id": "chatcmpl-789",
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "tool_a",
                                    "arguments": "{}",
                                },
                            },
                            {
                                "id": "call_2",
                                "type": "function",
                                "function": {
                                    "name": "tool_b",
                                    "arguments": '{"x": 1}',
                                },
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": 30,
                "completion_tokens": 15,
                "total_tokens": 45,
            },
        }
        resp = parse_openai_response(raw)
        assert len(resp.tool_calls) == 2
        assert resp.tool_calls[0].name == "tool_a"
        assert resp.tool_calls[1].name == "tool_b"
        assert resp.tool_calls[1].args == {"x": 1}

    def test_missing_usage(self) -> None:
        raw = {
            "id": "chatcmpl-000",
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
        }
        resp = parse_openai_response(raw)
        assert resp.token_usage.input_tokens == 0
        assert resp.token_usage.output_tokens == 0

    def test_null_content_becomes_empty_string(self) -> None:
        raw = {
            "id": "chatcmpl-x",
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": None},
                    "finish_reason": "stop",
                }
            ],
        }
        resp = parse_openai_response(raw)
        assert resp.content == ""


# ---------------------------------------------------------------------------
# Adapter class tests
# ---------------------------------------------------------------------------


class TestOpenAIAdapterInit:
    def test_init_with_explicit_api_key(self) -> None:
        from pop.models.openai import OpenAIAdapter

        adapter = OpenAIAdapter("gpt-4o", api_key="sk-test-key")
        assert adapter._model == "gpt-4o"
        assert adapter._api_key == "sk-test-key"

    def test_init_with_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pop.models.openai import OpenAIAdapter

        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        adapter = OpenAIAdapter("gpt-4o")
        assert adapter._api_key == "sk-from-env"

    def test_init_missing_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pop.models.openai import OpenAIAdapter

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="No API key provided"):
            OpenAIAdapter("gpt-4o")

    def test_init_custom_base_url(self) -> None:
        from pop.models.openai import OpenAIAdapter

        adapter = OpenAIAdapter(
            "gpt-4o",
            api_key="sk-test",
            base_url="https://custom.api.com/v1/",
        )
        assert adapter._base_url == "https://custom.api.com/v1"

    def test_init_custom_api_key_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pop.models.openai import OpenAIAdapter

        monkeypatch.setenv("MY_KEY", "sk-custom")
        adapter = OpenAIAdapter("gpt-4o", api_key_env="MY_KEY")
        assert adapter._api_key == "sk-custom"


class TestOpenAIAdapterChat:
    @pytest.mark.asyncio
    async def test_chat_sends_correct_request(self) -> None:
        import httpx
        from unittest.mock import AsyncMock, patch
        from pop.models.openai import OpenAIAdapter

        mock_response_data = {
            "id": "chatcmpl-test",
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        mock_response = AsyncMock()
        mock_response.json = lambda: mock_response_data
        mock_response.raise_for_status = lambda: None

        with patch("pop.models.openai.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            adapter = OpenAIAdapter("gpt-4o", api_key="sk-test")
            result = await adapter.chat([Message.user("Hi")])

            assert result.content == "Hello!"
            assert result.model == "gpt-4o"
            mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_with_tools(self) -> None:
        import httpx
        from unittest.mock import AsyncMock, patch
        from pop.models.openai import OpenAIAdapter

        mock_response_data = {
            "id": "chatcmpl-test",
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "search",
                                    "arguments": '{"q": "test"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        mock_response = AsyncMock()
        mock_response.json = lambda: mock_response_data
        mock_response.raise_for_status = lambda: None

        with patch("pop.models.openai.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            tool_def = ToolDefinition(
                name="search",
                description="Search",
                parameters={"type": "object", "properties": {"q": {"type": "string"}}},
                function=lambda q: q,
            )

            adapter = OpenAIAdapter("gpt-4o", api_key="sk-test")
            result = await adapter.chat([Message.user("Search")], tools=[tool_def])

            assert len(result.tool_calls) == 1
            assert result.tool_calls[0].name == "search"

            # Verify tools were included in the payload
            call_kwargs = mock_client.post.call_args
            payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
            assert "tools" in payload


class TestOpenAIAdapterChatStream:
    @pytest.mark.asyncio
    async def test_chat_stream_yields_chunks(self) -> None:
        from unittest.mock import AsyncMock, patch, MagicMock
        from pop.models.openai import OpenAIAdapter

        lines = [
            'data: {"choices": [{"delta": {"content": "Hello"}, "finish_reason": null}]}',
            'data: {"choices": [{"delta": {"content": " world"}, "finish_reason": null}]}',
            "data: [DONE]",
        ]

        async def mock_aiter_lines():
            for line in lines:
                yield line

        mock_response = AsyncMock()
        mock_response.raise_for_status = lambda: None
        mock_response.aiter_lines = mock_aiter_lines

        with patch("pop.models.openai.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)

            mock_stream_ctx = AsyncMock()
            mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
            mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_client.stream = MagicMock(return_value=mock_stream_ctx)
            mock_client_cls.return_value = mock_client

            adapter = OpenAIAdapter("gpt-4o", api_key="sk-test")
            chunks = []
            async for chunk in adapter.chat_stream([Message.user("Hi")]):
                chunks.append(chunk)

            assert len(chunks) == 2
            assert chunks[0].delta_content == "Hello"
            assert chunks[1].delta_content == " world"

    @pytest.mark.asyncio
    async def test_chat_stream_skips_non_data_lines(self) -> None:
        from unittest.mock import AsyncMock, patch, MagicMock
        from pop.models.openai import OpenAIAdapter

        lines = [
            "event: message",
            "",
            'data: {"choices": [{"delta": {"content": "ok"}, "finish_reason": null}]}',
            "data: [DONE]",
        ]

        async def mock_aiter_lines():
            for line in lines:
                yield line

        mock_response = AsyncMock()
        mock_response.raise_for_status = lambda: None
        mock_response.aiter_lines = mock_aiter_lines

        with patch("pop.models.openai.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)

            mock_stream_ctx = AsyncMock()
            mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
            mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_client.stream = MagicMock(return_value=mock_stream_ctx)
            mock_client_cls.return_value = mock_client

            adapter = OpenAIAdapter("gpt-4o", api_key="sk-test")
            chunks = []
            async for chunk in adapter.chat_stream([Message.user("Hi")]):
                chunks.append(chunk)

            assert len(chunks) == 1
            assert chunks[0].delta_content == "ok"
