"""Tests for the Kimi/Moonshot adapter."""

from __future__ import annotations

import pytest

from pop.models.kimi import KimiAdapter
from pop.models.openai import OpenAIAdapter
from pop.models.router import ModelRouter


class TestKimiAdapterInit:
    def test_inherits_from_openai_adapter(self) -> None:
        """KimiAdapter should be a subclass of OpenAIAdapter."""
        assert issubclass(KimiAdapter, OpenAIAdapter)

    def test_default_base_url(self) -> None:
        adapter = KimiAdapter("moonshot-v1-8k", api_key="sk-test")
        assert adapter._base_url == "https://api.moonshot.cn/v1"

    def test_default_api_key_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KIMI_API_KEY", "sk-kimi-from-env")
        adapter = KimiAdapter("moonshot-v1-8k")
        assert adapter._api_key == "sk-kimi-from-env"

    def test_custom_base_url(self) -> None:
        adapter = KimiAdapter(
            "moonshot-v1-8k",
            api_key="sk-test",
            base_url="https://custom.moonshot.cn/v1",
        )
        assert adapter._base_url == "https://custom.moonshot.cn/v1"

    def test_explicit_api_key(self) -> None:
        adapter = KimiAdapter("moonshot-v1-8k", api_key="sk-explicit")
        assert adapter._api_key == "sk-explicit"

    def test_missing_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("KIMI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="No API key provided"):
            KimiAdapter("moonshot-v1-8k")

    def test_model_name_stored(self) -> None:
        adapter = KimiAdapter("moonshot-v1-32k", api_key="sk-test")
        assert adapter._model == "moonshot-v1-32k"


class TestKimiRouterRegistration:
    def test_router_has_kimi_by_default(self) -> None:
        router = ModelRouter()
        assert "kimi" in router.providers

    def test_from_model_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KIMI_API_KEY", "sk-kimi-test")
        router = ModelRouter()
        adapter = router.from_model_string("kimi:moonshot-v1-8k")
        assert isinstance(adapter, KimiAdapter)
        assert adapter._model == "moonshot-v1-8k"  # type: ignore[attr-defined]
