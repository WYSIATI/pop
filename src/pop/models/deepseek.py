"""DeepSeek adapter — OpenAI-compatible API."""

from __future__ import annotations

from pop.models.openai import OpenAIAdapter


class DeepSeekAdapter(OpenAIAdapter):
    """DeepSeek uses an OpenAI-compatible API with a different base URL."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str = "https://api.deepseek.com/v1",
        api_key_env: str = "DEEPSEEK_API_KEY",
    ) -> None:
        super().__init__(model, api_key=api_key, base_url=base_url, api_key_env=api_key_env)
