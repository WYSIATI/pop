"""Grok (xAI) adapter — OpenAI-compatible API."""

from __future__ import annotations

from pop.models.openai import OpenAIAdapter


class GrokAdapter(OpenAIAdapter):
    """Grok uses an OpenAI-compatible API with xAI's base URL."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str = "https://api.x.ai/v1",
        api_key_env: str = "XAI_API_KEY",
    ) -> None:
        super().__init__(model, api_key=api_key, base_url=base_url, api_key_env=api_key_env)
