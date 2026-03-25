"""GLM/Zhipu AI adapter — OpenAI-compatible API."""

from __future__ import annotations

from pop.models.openai import OpenAIAdapter


class GLMAdapter(OpenAIAdapter):
    """GLM (Zhipu AI) uses an OpenAI-compatible API."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str = "https://open.bigmodel.cn/api/paas/v4",
        api_key_env: str = "GLM_API_KEY",
    ) -> None:
        super().__init__(model, api_key=api_key, base_url=base_url, api_key_env=api_key_env)
