"""Tests for the Grok (xAI) adapter."""

from __future__ import annotations

import pytest

from pop.models.grok import GrokAdapter
from pop.models.openai import OpenAIAdapter
from pop.models.router import ModelRouter


class TestGrokAdapterInit:
    def test_inherits_from_openai_adapter(self) -> None:
        """GrokAdapter should be a subclass of OpenAIAdapter."""
        assert issubclass(GrokAdapter, OpenAIAdapter)

    def test_default_base_url(self) -> None:
        adapter = GrokAdapter("grok-3", api_key="xai-test")
        assert adapter._base_url == "https://api.x.ai/v1"

    def test_default_api_key_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XAI_API_KEY", "xai-from-env")
        adapter = GrokAdapter("grok-3")
        assert adapter._api_key == "xai-from-env"

    def test_custom_base_url(self) -> None:
        adapter = GrokAdapter(
            "grok-3",
            api_key="xai-test",
            base_url="https://custom.x.ai/v1",
        )
        assert adapter._base_url == "https://custom.x.ai/v1"

    def test_explicit_api_key(self) -> None:
        adapter = GrokAdapter("grok-3", api_key="xai-explicit")
        assert adapter._api_key == "xai-explicit"

    def test_missing_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="No API key provided"):
            GrokAdapter("grok-3")

    def test_model_name_stored(self) -> None:
        adapter = GrokAdapter("grok-3-mini", api_key="xai-test")
        assert adapter._model == "grok-3-mini"


class TestGrokRouterRegistration:
    def test_router_has_grok_by_default(self) -> None:
        router = ModelRouter()
        assert "grok" in router.providers

    def test_from_model_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XAI_API_KEY", "xai-test")
        router = ModelRouter()
        adapter = router.from_model_string("grok:grok-3")
        assert isinstance(adapter, GrokAdapter)
        assert adapter._model == "grok-3"  # type: ignore[attr-defined]
