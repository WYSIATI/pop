"""Tests for the GLM/Zhipu AI adapter."""

from __future__ import annotations

import pytest

from pop.models.glm import GLMAdapter
from pop.models.openai import OpenAIAdapter
from pop.models.router import ModelRouter


class TestGLMAdapterInit:
    def test_inherits_from_openai_adapter(self) -> None:
        """GLMAdapter should be a subclass of OpenAIAdapter."""
        assert issubclass(GLMAdapter, OpenAIAdapter)

    def test_default_base_url(self) -> None:
        adapter = GLMAdapter("glm-4", api_key="sk-test")
        assert adapter._base_url == "https://open.bigmodel.cn/api/paas/v4"

    def test_default_api_key_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GLM_API_KEY", "sk-glm-from-env")
        adapter = GLMAdapter("glm-4")
        assert adapter._api_key == "sk-glm-from-env"

    def test_custom_base_url(self) -> None:
        adapter = GLMAdapter(
            "glm-4",
            api_key="sk-test",
            base_url="https://custom.bigmodel.cn/v1",
        )
        assert adapter._base_url == "https://custom.bigmodel.cn/v1"

    def test_explicit_api_key(self) -> None:
        adapter = GLMAdapter("glm-4", api_key="sk-explicit")
        assert adapter._api_key == "sk-explicit"

    def test_missing_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GLM_API_KEY", raising=False)
        with pytest.raises(ValueError, match="No API key provided"):
            GLMAdapter("glm-4")

    def test_model_name_stored(self) -> None:
        adapter = GLMAdapter("glm-4-flash", api_key="sk-test")
        assert adapter._model == "glm-4-flash"


class TestGLMRouterRegistration:
    def test_router_has_glm_by_default(self) -> None:
        router = ModelRouter()
        assert "glm" in router.providers

    def test_from_model_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GLM_API_KEY", "sk-glm-test")
        router = ModelRouter()
        adapter = router.from_model_string("glm:glm-4")
        assert isinstance(adapter, GLMAdapter)
        assert adapter._model == "glm-4"  # type: ignore[attr-defined]
