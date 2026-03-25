"""Kimi/Moonshot adapter — OpenAI-compatible API."""

from __future__ import annotations

from pop.models.openai import OpenAIAdapter


class KimiAdapter(OpenAIAdapter):
    """Kimi (Moonshot AI) uses an OpenAI-compatible API."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str = "https://api.moonshot.cn/v1",
        api_key_env: str = "KIMI_API_KEY",
    ) -> None:
        super().__init__(model, api_key=api_key, base_url=base_url, api_key_env=api_key_env)
