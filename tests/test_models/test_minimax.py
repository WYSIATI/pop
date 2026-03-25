"""Tests for MiniMax adapter — thin OpenAI-compatible subclass."""

from __future__ import annotations

import pytest

from pop.models.minimax import MiniMaxAdapter
from pop.models.openai import OpenAIAdapter
from pop.models.router import ModelRouter

# ---------------------------------------------------------------------------
# Adapter init
# ---------------------------------------------------------------------------


class TestMiniMaxAdapterInit:
    def test_correct_base_url(self) -> None:
        adapter = MiniMaxAdapter("minimax-01", api_key="mm-test-key")
        assert adapter._base_url == "https://api.minimax.chat/v1"

    def test_correct_api_key_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MINIMAX_API_KEY", "mm-from-env")
        adapter = MiniMaxAdapter("minimax-01")
        assert adapter._api_key == "mm-from-env"

    def test_missing_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        with pytest.raises(ValueError, match="No API key provided"):
            MiniMaxAdapter("minimax-01")

    def test_explicit_api_key(self) -> None:
        adapter = MiniMaxAdapter("minimax-01", api_key="mm-explicit")
        assert adapter._api_key == "mm-explicit"
        assert adapter._model == "minimax-01"


# ---------------------------------------------------------------------------
# Inheritance
# ---------------------------------------------------------------------------


class TestMiniMaxInheritance:
    def test_inherits_from_openai_adapter(self) -> None:
        assert issubclass(MiniMaxAdapter, OpenAIAdapter)

    def test_instance_is_openai_adapter(self) -> None:
        adapter = MiniMaxAdapter("minimax-01", api_key="mm-test-key")
        assert isinstance(adapter, OpenAIAdapter)


# ---------------------------------------------------------------------------
# Router registration
# ---------------------------------------------------------------------------


class TestMiniMaxRouter:
    def test_router_has_minimax_by_default(self) -> None:
        router = ModelRouter()
        assert "minimax" in router.providers

    def test_from_model_string_minimax(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MINIMAX_API_KEY", "mm-test-key")
        router = ModelRouter()
        adapter = router.from_model_string("minimax:minimax-01")
        assert isinstance(adapter, MiniMaxAdapter)
        assert adapter._model == "minimax-01"
