"""Tests for the DeepSeek adapter."""

from __future__ import annotations

import pytest

from pop.models.deepseek import DeepSeekAdapter
from pop.models.openai import OpenAIAdapter
from pop.models.router import ModelRouter


class TestDeepSeekAdapterInit:
    def test_inherits_from_openai_adapter(self) -> None:
        """DeepSeekAdapter should be a subclass of OpenAIAdapter."""
        assert issubclass(DeepSeekAdapter, OpenAIAdapter)

    def test_default_base_url(self) -> None:
        adapter = DeepSeekAdapter("deepseek-chat", api_key="sk-test")
        assert adapter._base_url == "https://api.deepseek.com/v1"

    def test_default_api_key_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-ds-from-env")
        adapter = DeepSeekAdapter("deepseek-chat")
        assert adapter._api_key == "sk-ds-from-env"

    def test_custom_base_url(self) -> None:
        adapter = DeepSeekAdapter(
            "deepseek-chat",
            api_key="sk-test",
            base_url="https://custom.deepseek.com/v1",
        )
        assert adapter._base_url == "https://custom.deepseek.com/v1"

    def test_explicit_api_key(self) -> None:
        adapter = DeepSeekAdapter("deepseek-chat", api_key="sk-explicit")
        assert adapter._api_key == "sk-explicit"

    def test_missing_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        with pytest.raises(ValueError, match="No API key provided"):
            DeepSeekAdapter("deepseek-chat")

    def test_model_name_stored(self) -> None:
        adapter = DeepSeekAdapter("deepseek-coder", api_key="sk-test")
        assert adapter._model == "deepseek-coder"


class TestDeepSeekRouterRegistration:
    def test_router_has_deepseek_by_default(self) -> None:
        router = ModelRouter()
        assert "deepseek" in router.providers

    def test_from_model_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-ds-test")
        router = ModelRouter()
        adapter = router.from_model_string("deepseek:deepseek-chat")
        assert isinstance(adapter, DeepSeekAdapter)
        assert adapter._model == "deepseek-chat"  # type: ignore[attr-defined]
