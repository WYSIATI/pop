"""Tests for Anthropic adapter — conversion logic only, no real API calls."""

from __future__ import annotations

import pytest

from pop.types import Message, Role, ToolCall, ToolDefinition, TokenUsage
from pop.models.anthropic import (
    messages_to_anthropic,
    tools_to_anthropic,
    parse_anthropic_response,
)


# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------

class TestMessagesToAnthropic:
    def test_system_extracted_separately(self) -> None:
        msgs = [
            Message.system("you are helpful"),
            Message.user("hello"),
        ]
        system, messages = messages_to_anthropic(msgs)
        assert system == "you are helpful"
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "hello"

    def test_no_system_message(self) -> None:
        msgs = [Message.user("hello")]
        system, messages = messages_to_anthropic(msgs)
        assert system == ""
        assert len(messages) == 1

    def test_multiple_system_messages_concatenated(self) -> None:
        msgs = [
            Message.system("rule 1"),
            Message.system("rule 2"),
            Message.user("hi"),
        ]
        system, messages = messages_to_anthropic(msgs)
        assert "rule 1" in system
        assert "rule 2" in system
        assert len(messages) == 1

    def test_assistant_plain(self) -> None:
        msgs = [Message.user("hi"), Message.assistant("hello")]
        system, messages = messages_to_anthropic(msgs)
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "hello"

    def test_assistant_with_tool_calls(self) -> None:
        tc = ToolCall(name="search", args={"q": "test"}, call_id="tu_1")
        msgs = [
            Message.user("search for test"),
            Message.assistant("", tool_calls=(tc,)),
        ]
        system, messages = messages_to_anthropic(msgs)
        assistant_msg = messages[1]
        assert assistant_msg["role"] == "assistant"
        # Should have content blocks with tool_use type
        assert isinstance(assistant_msg["content"], list)
        tool_block = assistant_msg["content"][0]
        assert tool_block["type"] == "tool_use"
        assert tool_block["id"] == "tu_1"
        assert tool_block["name"] == "search"
        assert tool_block["input"] == {"q": "test"}

    def test_tool_result_message(self) -> None:
        msgs = [
            Message.user("hi"),
            Message.assistant(
                "",
                tool_calls=(ToolCall(name="search", args={}, call_id="tu_1"),),
            ),
            Message.tool_result("found it", tool_call_id="tu_1", name="search"),
        ]
        system, messages = messages_to_anthropic(msgs)
        tool_msg = messages[2]
        assert tool_msg["role"] == "user"
        assert isinstance(tool_msg["content"], list)
        assert tool_msg["content"][0]["type"] == "tool_result"
        assert tool_msg["content"][0]["tool_use_id"] == "tu_1"
        assert tool_msg["content"][0]["content"] == "found it"


# ---------------------------------------------------------------------------
# Tool definition conversion
# ---------------------------------------------------------------------------

class TestToolsToAnthropic:
    def test_single_tool(self) -> None:
        tool = ToolDefinition(
            name="calculator",
            description="Do math",
            parameters={
                "type": "object",
                "properties": {"expr": {"type": "string"}},
                "required": ["expr"],
            },
            function=lambda x: x,
        )
        result = tools_to_anthropic([tool])
        assert len(result) == 1
        assert result[0]["name"] == "calculator"
        assert result[0]["description"] == "Do math"
        assert result[0]["input_schema"]["type"] == "object"

    def test_empty_tools(self) -> None:
        assert tools_to_anthropic([]) == []

    def test_none_tools(self) -> None:
        assert tools_to_anthropic(None) == []


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

class TestParseAnthropicResponse:
    def test_text_response(self) -> None:
        raw = {
            "id": "msg_123",
            "type": "message",
            "model": "claude-3-opus-20240229",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        resp = parse_anthropic_response(raw)
        assert resp.content == "Hello!"
        assert resp.tool_calls == ()
        assert resp.model == "claude-3-opus-20240229"
        assert resp.finish_reason == "end_turn"
        assert resp.token_usage.input_tokens == 10
        assert resp.token_usage.output_tokens == 5

    def test_tool_use_response(self) -> None:
        raw = {
            "id": "msg_456",
            "type": "message",
            "model": "claude-3-sonnet-20240229",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me search for that."},
                {
                    "type": "tool_use",
                    "id": "tu_abc",
                    "name": "web_search",
                    "input": {"query": "python asyncio"},
                },
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 20, "output_tokens": 15},
        }
        resp = parse_anthropic_response(raw)
        assert resp.content == "Let me search for that."
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "web_search"
        assert resp.tool_calls[0].args == {"query": "python asyncio"}
        assert resp.tool_calls[0].call_id == "tu_abc"
        assert resp.finish_reason == "tool_use"

    def test_multiple_tool_use_blocks(self) -> None:
        raw = {
            "id": "msg_789",
            "type": "message",
            "model": "claude-3-opus-20240229",
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tu_1",
                    "name": "tool_a",
                    "input": {},
                },
                {
                    "type": "tool_use",
                    "id": "tu_2",
                    "name": "tool_b",
                    "input": {"x": 1},
                },
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 30, "output_tokens": 20},
        }
        resp = parse_anthropic_response(raw)
        assert len(resp.tool_calls) == 2
        assert resp.tool_calls[0].name == "tool_a"
        assert resp.tool_calls[1].name == "tool_b"
        assert resp.tool_calls[1].args == {"x": 1}

    def test_missing_usage(self) -> None:
        raw = {
            "id": "msg_x",
            "type": "message",
            "model": "claude-3-haiku-20240307",
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
        }
        resp = parse_anthropic_response(raw)
        assert resp.token_usage.input_tokens == 0
        assert resp.token_usage.output_tokens == 0

    def test_empty_content(self) -> None:
        raw = {
            "id": "msg_y",
            "type": "message",
            "model": "claude-3-haiku-20240307",
            "role": "assistant",
            "content": [],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 0},
        }
        resp = parse_anthropic_response(raw)
        assert resp.content == ""
        assert resp.tool_calls == ()


# ---------------------------------------------------------------------------
# Adapter class tests
# ---------------------------------------------------------------------------


class TestAnthropicAdapterInit:
    def test_init_with_explicit_api_key(self) -> None:
        from pop.models.anthropic import AnthropicAdapter

        adapter = AnthropicAdapter("claude-3-opus", api_key="sk-ant-test")
        assert adapter._model == "claude-3-opus"
        assert adapter._api_key == "sk-ant-test"

    def test_init_with_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pop.models.anthropic import AnthropicAdapter

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env")
        adapter = AnthropicAdapter("claude-3-opus")
        assert adapter._api_key == "sk-ant-env"

    def test_init_missing_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pop.models.anthropic import AnthropicAdapter

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ValueError, match="No API key provided"):
            AnthropicAdapter("claude-3-opus")

    def test_init_custom_base_url(self) -> None:
        from pop.models.anthropic import AnthropicAdapter

        adapter = AnthropicAdapter(
            "claude-3-opus",
            api_key="sk-test",
            base_url="https://custom.anthropic.com/v1/",
        )
        assert adapter._base_url == "https://custom.anthropic.com/v1"

    def test_init_custom_max_tokens(self) -> None:
        from pop.models.anthropic import AnthropicAdapter

        adapter = AnthropicAdapter("claude-3-opus", api_key="sk-test", max_tokens=8192)
        assert adapter._max_tokens == 8192

    def test_init_custom_api_key_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pop.models.anthropic import AnthropicAdapter

        monkeypatch.setenv("MY_ANT_KEY", "sk-ant-custom")
        adapter = AnthropicAdapter("claude-3-opus", api_key_env="MY_ANT_KEY")
        assert adapter._api_key == "sk-ant-custom"


class TestAnthropicAdapterChat:
    @pytest.mark.asyncio
    async def test_chat_sends_correct_request(self) -> None:
        from unittest.mock import AsyncMock, patch
        from pop.models.anthropic import AnthropicAdapter

        mock_response_data = {
            "id": "msg_test",
            "type": "message",
            "model": "claude-3-opus",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

        mock_response = AsyncMock()
        mock_response.json = lambda: mock_response_data
        mock_response.raise_for_status = lambda: None

        with patch("pop.models.anthropic.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            adapter = AnthropicAdapter("claude-3-opus", api_key="sk-test")
            result = await adapter.chat([Message.user("Hi")])

            assert result.content == "Hello!"
            mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_with_system_and_tools(self) -> None:
        from unittest.mock import AsyncMock, patch
        from pop.models.anthropic import AnthropicAdapter

        mock_response_data = {
            "id": "msg_test",
            "type": "message",
            "model": "claude-3-opus",
            "role": "assistant",
            "content": [{"type": "text", "text": "OK"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

        mock_response = AsyncMock()
        mock_response.json = lambda: mock_response_data
        mock_response.raise_for_status = lambda: None

        with patch("pop.models.anthropic.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            tool_def = ToolDefinition(
                name="search",
                description="Search",
                parameters={"type": "object", "properties": {}},
                function=lambda: None,
            )

            adapter = AnthropicAdapter("claude-3-opus", api_key="sk-test")
            result = await adapter.chat(
                [Message.system("Be helpful"), Message.user("Hi")],
                tools=[tool_def],
            )

            call_kwargs = mock_client.post.call_args
            payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
            assert "system" in payload
            assert "tools" in payload


class TestAnthropicAdapterChatStream:
    @pytest.mark.asyncio
    async def test_chat_stream_yields_text_chunks(self) -> None:
        from unittest.mock import AsyncMock, patch, MagicMock
        from pop.models.anthropic import AnthropicAdapter

        lines = [
            'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello"}}',
            'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": " world"}}',
            'data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}}',
            'data: {"type": "message_stop"}',
        ]

        async def mock_aiter_lines():
            for line in lines:
                yield line

        mock_response = AsyncMock()
        mock_response.raise_for_status = lambda: None
        mock_response.aiter_lines = mock_aiter_lines

        with patch("pop.models.anthropic.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)

            mock_stream_ctx = AsyncMock()
            mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
            mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_client.stream = MagicMock(return_value=mock_stream_ctx)
            mock_client_cls.return_value = mock_client

            adapter = AnthropicAdapter("claude-3-opus", api_key="sk-test")
            chunks = []
            async for chunk in adapter.chat_stream([Message.user("Hi")]):
                chunks.append(chunk)

            assert len(chunks) == 3
            assert chunks[0].delta_content == "Hello"
            assert chunks[1].delta_content == " world"
            assert chunks[2].finish_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_chat_stream_skips_non_data_lines(self) -> None:
        from unittest.mock import AsyncMock, patch, MagicMock
        from pop.models.anthropic import AnthropicAdapter

        lines = [
            "event: content_block_delta",
            "",
            'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "ok"}}',
            'data: {"type": "message_stop"}',
        ]

        async def mock_aiter_lines():
            for line in lines:
                yield line

        mock_response = AsyncMock()
        mock_response.raise_for_status = lambda: None
        mock_response.aiter_lines = mock_aiter_lines

        with patch("pop.models.anthropic.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)

            mock_stream_ctx = AsyncMock()
            mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
            mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_client.stream = MagicMock(return_value=mock_stream_ctx)
            mock_client_cls.return_value = mock_client

            adapter = AnthropicAdapter("claude-3-opus", api_key="sk-test")
            chunks = []
            async for chunk in adapter.chat_stream([Message.user("Hi")]):
                chunks.append(chunk)

            assert len(chunks) == 1
            assert chunks[0].delta_content == "ok"
