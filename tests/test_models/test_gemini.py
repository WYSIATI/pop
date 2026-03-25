"""Tests for Gemini adapter — conversion logic only, no real API calls."""

from __future__ import annotations

import pytest

from pop.models.gemini import (
    messages_to_gemini,
    parse_gemini_response,
    tools_to_gemini,
)
from pop.types import Message, ToolCall, ToolDefinition

# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------


class TestMessagesToGemini:
    def test_user_message(self) -> None:
        msgs = [Message.user("hello")]
        system_instruction, contents = messages_to_gemini(msgs)
        assert system_instruction is None
        assert len(contents) == 1
        assert contents[0]["role"] == "user"
        assert contents[0]["parts"] == [{"text": "hello"}]

    def test_system_extracted_to_system_instruction(self) -> None:
        msgs = [
            Message.system("you are helpful"),
            Message.user("hello"),
        ]
        system_instruction, contents = messages_to_gemini(msgs)
        assert system_instruction is not None
        assert system_instruction["parts"] == [{"text": "you are helpful"}]
        assert len(contents) == 1
        assert contents[0]["role"] == "user"

    def test_multiple_system_messages_concatenated(self) -> None:
        msgs = [
            Message.system("rule 1"),
            Message.system("rule 2"),
            Message.user("hi"),
        ]
        system_instruction, contents = messages_to_gemini(msgs)
        assert system_instruction is not None
        assert "rule 1" in system_instruction["parts"][0]["text"]
        assert "rule 2" in system_instruction["parts"][0]["text"]
        assert len(contents) == 1

    def test_assistant_plain(self) -> None:
        msgs = [Message.user("hi"), Message.assistant("hello")]
        _system, contents = messages_to_gemini(msgs)
        assert contents[1]["role"] == "model"
        assert contents[1]["parts"] == [{"text": "hello"}]

    def test_assistant_with_tool_calls(self) -> None:
        tc = ToolCall(name="search", args={"q": "test"}, call_id="tc_1")
        msgs = [
            Message.user("search for test"),
            Message.assistant("Let me search", tool_calls=(tc,)),
        ]
        _system, contents = messages_to_gemini(msgs)
        model_msg = contents[1]
        assert model_msg["role"] == "model"
        assert model_msg["parts"][0] == {"text": "Let me search"}
        assert model_msg["parts"][1] == {"functionCall": {"name": "search", "args": {"q": "test"}}}

    def test_tool_result_message(self) -> None:
        msgs = [
            Message.user("hi"),
            Message.assistant(
                "",
                tool_calls=(ToolCall(name="search", args={}, call_id="tc_1"),),
            ),
            Message.tool_result("found it", tool_call_id="tc_1", name="search"),
        ]
        _system, contents = messages_to_gemini(msgs)
        tool_msg = contents[2]
        assert tool_msg["role"] == "user"
        assert tool_msg["parts"][0]["functionResponse"]["name"] == "search"
        assert tool_msg["parts"][0]["functionResponse"]["response"] == {"result": "found it"}

    def test_tool_result_uses_tool_call_id_as_fallback_name(self) -> None:
        msgs = [
            Message.tool_result("result", tool_call_id="tc_1"),
        ]
        _system, contents = messages_to_gemini(msgs)
        assert contents[0]["parts"][0]["functionResponse"]["name"] == "tc_1"


# ---------------------------------------------------------------------------
# Tool definition conversion
# ---------------------------------------------------------------------------


class TestToolsToGemini:
    def test_converts_tool_definitions(self) -> None:
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
        result = tools_to_gemini([tool])
        assert len(result) == 1
        decls = result[0]["functionDeclarations"]
        assert len(decls) == 1
        assert decls[0]["name"] == "calculator"
        assert decls[0]["description"] == "Do math"
        assert decls[0]["parameters"]["type"] == "object"

    def test_empty_list_returns_empty(self) -> None:
        assert tools_to_gemini([]) == []

    def test_none_returns_empty(self) -> None:
        assert tools_to_gemini(None) == []

    def test_multiple_tools_in_single_declaration_block(self) -> None:
        tools = [
            ToolDefinition(
                name="tool_a",
                description="A",
                parameters={"type": "object", "properties": {}},
                function=lambda: None,
            ),
            ToolDefinition(
                name="tool_b",
                description="B",
                parameters={"type": "object", "properties": {}},
                function=lambda: None,
            ),
        ]
        result = tools_to_gemini(tools)
        assert len(result) == 1
        decls = result[0]["functionDeclarations"]
        assert len(decls) == 2
        assert decls[0]["name"] == "tool_a"
        assert decls[1]["name"] == "tool_b"


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


class TestParseGeminiResponse:
    def test_text_only_response(self) -> None:
        raw = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "Hello!"}], "role": "model"},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
            },
            "modelVersion": "gemini-2.0-flash",
        }
        resp = parse_gemini_response(raw)
        assert resp.content == "Hello!"
        assert resp.tool_calls == ()
        assert resp.model == "gemini-2.0-flash"
        assert resp.finish_reason == "STOP"

    def test_response_with_function_call(self) -> None:
        raw = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "web_search",
                                    "args": {"query": "python asyncio"},
                                }
                            }
                        ],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 20,
                "candidatesTokenCount": 15,
            },
        }
        resp = parse_gemini_response(raw)
        assert resp.content == ""
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "web_search"
        assert resp.tool_calls[0].args == {"query": "python asyncio"}

    def test_token_usage_extraction(self) -> None:
        raw = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "ok"}], "role": "model"},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 42,
                "candidatesTokenCount": 17,
            },
        }
        resp = parse_gemini_response(raw)
        assert resp.token_usage.input_tokens == 42
        assert resp.token_usage.output_tokens == 17
        assert resp.token_usage.total == 59

    def test_missing_usage_metadata(self) -> None:
        raw = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "ok"}], "role": "model"},
                    "finishReason": "STOP",
                }
            ],
        }
        resp = parse_gemini_response(raw)
        assert resp.token_usage.input_tokens == 0
        assert resp.token_usage.output_tokens == 0

    def test_empty_candidates(self) -> None:
        raw = {"candidates": []}
        resp = parse_gemini_response(raw)
        assert resp.content == ""
        assert resp.tool_calls == ()

    def test_mixed_text_and_function_call(self) -> None:
        raw = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "Let me search."},
                            {
                                "functionCall": {
                                    "name": "search",
                                    "args": {"q": "test"},
                                }
                            },
                        ],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
            },
        }
        resp = parse_gemini_response(raw)
        assert resp.content == "Let me search."
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "search"


# ---------------------------------------------------------------------------
# Adapter class tests
# ---------------------------------------------------------------------------


class TestGeminiAdapterInit:
    def test_init_with_explicit_api_key(self) -> None:
        from pop.models.gemini import GeminiAdapter

        adapter = GeminiAdapter("gemini-2.0-flash", api_key="test-key-123")
        assert adapter._model == "gemini-2.0-flash"
        assert adapter._api_key == "test-key-123"

    def test_init_with_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pop.models.gemini import GeminiAdapter

        monkeypatch.setenv("GEMINI_API_KEY", "env-key-456")
        adapter = GeminiAdapter("gemini-2.0-flash")
        assert adapter._api_key == "env-key-456"

    def test_init_missing_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pop.models.gemini import GeminiAdapter

        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="No API key provided"):
            GeminiAdapter("gemini-2.0-flash")

    def test_init_custom_base_url(self) -> None:
        from pop.models.gemini import GeminiAdapter

        adapter = GeminiAdapter(
            "gemini-2.0-flash",
            api_key="test-key",
            base_url="https://custom.googleapis.com/v1beta/",
        )
        assert adapter._base_url == "https://custom.googleapis.com/v1beta"

    def test_init_custom_api_key_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pop.models.gemini import GeminiAdapter

        monkeypatch.setenv("MY_GEMINI_KEY", "custom-env-key")
        adapter = GeminiAdapter("gemini-2.0-flash", api_key_env="MY_GEMINI_KEY")
        assert adapter._api_key == "custom-env-key"

    def test_adapter_has_client_attribute(self) -> None:
        from pop.models.gemini import GeminiAdapter

        adapter = GeminiAdapter("gemini-2.0-flash", api_key="test-key")
        assert hasattr(adapter, "_client")
        import httpx

        assert isinstance(adapter._client, httpx.AsyncClient)


# ---------------------------------------------------------------------------
# Router integration
# ---------------------------------------------------------------------------


class TestGeminiRouterRegistration:
    def test_router_has_gemini_by_default(self) -> None:
        from pop.models.router import ModelRouter

        router = ModelRouter()
        assert "gemini" in router.providers

    def test_from_model_string_gemini(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pop.models.gemini import GeminiAdapter
        from pop.models.router import ModelRouter

        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        router = ModelRouter()
        adapter = router.from_model_string("gemini:gemini-2.0-flash")
        assert isinstance(adapter, GeminiAdapter)
        assert adapter._model == "gemini-2.0-flash"


# ---------------------------------------------------------------------------
# Adapter chat (mocked)
# ---------------------------------------------------------------------------


class TestGeminiAdapterChat:
    @pytest.mark.asyncio
    async def test_chat_sends_correct_request(self) -> None:
        from unittest.mock import AsyncMock, patch

        from pop.models.gemini import GeminiAdapter

        mock_response_data = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "Hello!"}], "role": "model"},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
            },
        }

        mock_response = AsyncMock()
        mock_response.json = lambda: mock_response_data
        mock_response.raise_for_status = lambda: None

        with patch("pop.models.gemini.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            adapter = GeminiAdapter("gemini-2.0-flash", api_key="test-key")
            result = await adapter.chat([Message.user("Hi")])

            assert result.content == "Hello!"
            mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_with_system_and_tools(self) -> None:
        from unittest.mock import AsyncMock, patch

        from pop.models.gemini import GeminiAdapter

        mock_response_data = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "OK"}], "role": "model"},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
            },
        }

        mock_response = AsyncMock()
        mock_response.json = lambda: mock_response_data
        mock_response.raise_for_status = lambda: None

        with patch("pop.models.gemini.httpx.AsyncClient") as mock_client_cls:
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

            adapter = GeminiAdapter("gemini-2.0-flash", api_key="test-key")
            await adapter.chat(
                [Message.system("Be helpful"), Message.user("Hi")],
                tools=[tool_def],
            )

            call_args = mock_client.post.call_args
            url = call_args[0][0] if call_args[0] else call_args.kwargs.get("url", "")
            payload = call_args.kwargs.get("json") or call_args[1].get("json")
            assert "systemInstruction" in payload
            assert "tools" in payload
            assert "key=test-key" in url


class TestGeminiAdapterChatStream:
    @pytest.mark.asyncio
    async def test_chat_stream_yields_text_chunks(self) -> None:
        from unittest.mock import AsyncMock, MagicMock, patch

        from pop.models.gemini import GeminiAdapter

        lines = [
            'data: {"candidates": [{"content": {"parts": [{"text": "Hello"}], "role": "model"}}]}',
            'data: {"candidates": [{"content": {"parts": [{"text": " world"}], "role": "model"}}]}',
            'data: {"candidates": [{"content": {"parts": [{"text": "!"}],'
            ' "role": "model"}, "finishReason": "STOP"}]}',
        ]

        async def mock_aiter_lines():
            for line in lines:
                yield line

        mock_response = AsyncMock()
        mock_response.raise_for_status = lambda: None
        mock_response.aiter_lines = mock_aiter_lines

        with patch("pop.models.gemini.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)

            mock_stream_ctx = AsyncMock()
            mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
            mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_client.stream = MagicMock(return_value=mock_stream_ctx)
            mock_client_cls.return_value = mock_client

            adapter = GeminiAdapter("gemini-2.0-flash", api_key="test-key")
            chunks = []
            async for chunk in adapter.chat_stream([Message.user("Hi")]):
                chunks.append(chunk)

            assert len(chunks) >= 3
            assert chunks[0].delta_content == "Hello"
            assert chunks[1].delta_content == " world"
            assert chunks[2].delta_content == "!"
