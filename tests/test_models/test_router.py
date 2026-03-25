"""Tests for the model router — written FIRST per TDD."""

from __future__ import annotations

import pytest

from pop.models.base import ModelAdapter, StreamChunk
from pop.models.router import ModelRouter, parse_model_string
from pop.types import Message, ModelResponse, ToolDefinition

# ---------------------------------------------------------------------------
# parse_model_string
# ---------------------------------------------------------------------------


class TestParseModelString:
    def test_valid_openai(self) -> None:
        provider, model = parse_model_string("openai:gpt-4o")
        assert provider == "openai"
        assert model == "gpt-4o"

    def test_valid_anthropic(self) -> None:
        provider, model = parse_model_string("anthropic:claude-3-opus")
        assert provider == "anthropic"
        assert model == "claude-3-opus"

    def test_valid_with_slashes(self) -> None:
        provider, model = parse_model_string("openai:ft:gpt-4o:my-org")
        assert provider == "openai"
        assert model == "ft:gpt-4o:my-org"

    def test_missing_colon_raises(self) -> None:
        with pytest.raises(ValueError, match="must be in 'provider:model' format"):
            parse_model_string("gpt-4o")

    def test_empty_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Provider cannot be empty"):
            parse_model_string(":gpt-4o")

    def test_empty_model_raises(self) -> None:
        with pytest.raises(ValueError, match="Model cannot be empty"):
            parse_model_string("openai:")

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_model_string("")


# ---------------------------------------------------------------------------
# Helpers — fake adapters for testing
# ---------------------------------------------------------------------------


class FakeAdapter:
    """A minimal fake that satisfies the ModelAdapter protocol for testing."""

    def __init__(self, name: str = "fake", should_fail: bool = False) -> None:
        self._name = name
        self._should_fail = should_fail

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
    ) -> ModelResponse:
        if self._should_fail:
            raise ConnectionError(f"{self._name} failed")
        return ModelResponse(content=f"response from {self._name}", model=self._name)

    async def chat_stream(self, messages, tools=None):  # type: ignore[override]
        yield StreamChunk(delta_content="chunk")


def _factory_ok(model: str, **kwargs: object) -> ModelAdapter:
    return FakeAdapter(name=f"ok-{model}")  # type: ignore[return-value]


def _factory_fail(model: str, **kwargs: object) -> ModelAdapter:
    return FakeAdapter(name=f"fail-{model}", should_fail=True)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# ModelRouter — registry
# ---------------------------------------------------------------------------


class TestModelRouterRegistry:
    def test_register_and_get_adapter(self) -> None:
        router = ModelRouter()
        router.register("fake", _factory_ok)
        adapter = router.get_adapter("fake", "test-model")
        assert adapter is not None

    def test_unknown_provider_raises(self) -> None:
        router = ModelRouter()
        with pytest.raises(ValueError, match="Unknown provider 'nope'"):
            router.get_adapter("nope", "model")

    def test_register_overwrites(self) -> None:
        router = ModelRouter()
        router.register("p", _factory_ok)
        router.register("p", _factory_fail)
        adapter = router.get_adapter("p", "m")
        # Should be the second factory (fail variant)
        assert adapter._name == "fail-m"  # type: ignore[attr-defined]

    def test_from_model_string(self) -> None:
        router = ModelRouter()
        router.register("openai", _factory_ok)
        adapter = router.from_model_string("openai:gpt-4o")
        assert adapter is not None


# ---------------------------------------------------------------------------
# ModelRouter — fallback chain
# ---------------------------------------------------------------------------


class TestFallbackChain:
    @pytest.mark.asyncio
    async def test_first_succeeds(self) -> None:
        router = ModelRouter()
        router.register("ok", _factory_ok)
        response = await router.chat_with_fallback(
            model_strings=["ok:model-a"],
            messages=[Message.user("hi")],
        )
        assert response.content == "response from ok-model-a"

    @pytest.mark.asyncio
    async def test_fallback_on_failure(self) -> None:
        router = ModelRouter()
        router.register("bad", _factory_fail)
        router.register("good", _factory_ok)
        response = await router.chat_with_fallback(
            model_strings=["bad:model-a", "good:model-b"],
            messages=[Message.user("hi")],
        )
        assert response.content == "response from ok-model-b"

    @pytest.mark.asyncio
    async def test_all_fail_raises(self) -> None:
        router = ModelRouter()
        router.register("bad", _factory_fail)
        with pytest.raises(RuntimeError, match="All models failed"):
            await router.chat_with_fallback(
                model_strings=["bad:m1", "bad:m2"],
                messages=[Message.user("hi")],
            )

    @pytest.mark.asyncio
    async def test_empty_chain_raises(self) -> None:
        router = ModelRouter()
        with pytest.raises(ValueError, match="at least one model"):
            await router.chat_with_fallback(
                model_strings=[],
                messages=[Message.user("hi")],
            )


# ---------------------------------------------------------------------------
# register_provider convenience
# ---------------------------------------------------------------------------


class TestRegisterProvider:
    def test_register_provider_creates_factory(self) -> None:
        router = ModelRouter()
        router.register_provider(
            name="custom",
            base_url="https://custom.api.com/v1",
            api_key_env="CUSTOM_API_KEY",
            protocol="openai",
        )
        assert "custom" in router.providers

    def test_register_provider_unknown_protocol_raises(self) -> None:
        router = ModelRouter()
        with pytest.raises(ValueError, match="Unknown protocol 'bad_proto'"):
            router.register_provider(
                name="x",
                base_url="https://x.com",
                api_key_env="X_KEY",
                protocol="bad_proto",
            )

    def test_register_provider_openai_creates_working_factory(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CUSTOM_OAI_KEY", "sk-test-key")
        router = ModelRouter()
        router.register_provider(
            name="custom_oai",
            base_url="https://custom.openai.com/v1",
            api_key_env="CUSTOM_OAI_KEY",
            protocol="openai",
        )
        adapter = router.get_adapter("custom_oai", "gpt-4o")
        assert adapter._model == "gpt-4o"  # type: ignore[attr-defined]
        assert adapter._api_key == "sk-test-key"  # type: ignore[attr-defined]

    def test_register_provider_anthropic_creates_working_factory(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CUSTOM_ANT_KEY", "sk-ant-test-key")
        router = ModelRouter()
        router.register_provider(
            name="custom_ant",
            base_url="https://custom.anthropic.com/v1",
            api_key_env="CUSTOM_ANT_KEY",
            protocol="anthropic",
        )
        adapter = router.get_adapter("custom_ant", "claude-3-opus")
        assert adapter._model == "claude-3-opus"  # type: ignore[attr-defined]
        assert adapter._api_key == "sk-ant-test-key"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# models/__init__.py convenience functions
# ---------------------------------------------------------------------------


class TestModelsInit:
    def test_register_provider_global(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pop.models import _router, register_provider

        monkeypatch.setenv("TEST_GLOBAL_KEY", "sk-global")
        register_provider(
            name="test_global",
            base_url="https://test.com/v1",
            api_key_env="TEST_GLOBAL_KEY",
            protocol="openai",
        )
        assert "test_global" in _router.providers

    def test_model_function(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pop.models import _router, model

        monkeypatch.setenv("TEST_MODEL_KEY", "sk-model")
        _router.register_provider(
            name="test_model_prov",
            base_url="https://test.com/v1",
            api_key_env="TEST_MODEL_KEY",
            protocol="openai",
        )
        adapter = model("test_model_prov", "gpt-4o")
        assert adapter is not None

    @pytest.mark.asyncio
    async def test_chat_convenience(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pop.models import _router

        # Register a fake provider
        _router.register("test_chat_prov", _factory_ok)

        result = await _router.from_model_string("test_chat_prov:some-model").chat(
            [Message.user("Hi")]
        )
        assert "response from" in result.content


# ---------------------------------------------------------------------------
# H1: ModelRouter auto-registers default providers
# ---------------------------------------------------------------------------


class TestDefaultProviders:
    def test_router_has_openai_by_default(self) -> None:
        """ModelRouter() should have 'openai' registered out of the box."""
        router = ModelRouter()
        assert "openai" in router.providers

    def test_router_has_anthropic_by_default(self) -> None:
        """ModelRouter() should have 'anthropic' registered out of the box."""
        router = ModelRouter()
        assert "anthropic" in router.providers

    def test_router_has_deepseek_by_default(self) -> None:
        """ModelRouter() should have 'deepseek' registered out of the box."""
        router = ModelRouter()
        assert "deepseek" in router.providers

    def test_router_has_kimi_by_default(self) -> None:
        """ModelRouter() should have 'kimi' registered out of the box."""
        router = ModelRouter()
        assert "kimi" in router.providers

    def test_router_has_glm_by_default(self) -> None:
        """ModelRouter() should have 'glm' registered out of the box."""
        router = ModelRouter()
        assert "glm" in router.providers

    def test_from_model_string_openai(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """from_model_string('openai:gpt-4o') returns an OpenAIAdapter."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        router = ModelRouter()
        adapter = router.from_model_string("openai:gpt-4o")

        from pop.models.openai import OpenAIAdapter

        assert isinstance(adapter, OpenAIAdapter)
        assert adapter._model == "gpt-4o"  # type: ignore[attr-defined]

    def test_from_model_string_anthropic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """from_model_string('anthropic:claude-sonnet') returns an AnthropicAdapter."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
        router = ModelRouter()
        adapter = router.from_model_string("anthropic:claude-sonnet")

        from pop.models.anthropic import AnthropicAdapter

        assert isinstance(adapter, AnthropicAdapter)
        assert adapter._model == "claude-sonnet"  # type: ignore[attr-defined]

    def test_custom_provider_overrides_default(self) -> None:
        """Registering 'openai' manually should override the default factory."""
        router = ModelRouter()
        router.register("openai", _factory_ok)
        adapter = router.get_adapter("openai", "custom-model")
        assert adapter._name == "ok-custom-model"  # type: ignore[attr-defined]
