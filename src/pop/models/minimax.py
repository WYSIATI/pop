"""MiniMax adapter — OpenAI-compatible API."""

from __future__ import annotations

from pop.models.openai import OpenAIAdapter


class MiniMaxAdapter(OpenAIAdapter):
    """MiniMax uses an OpenAI-compatible chat completions API."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str = "https://api.minimax.chat/v1",
        api_key_env: str = "MINIMAX_API_KEY",
    ) -> None:
        super().__init__(model, api_key=api_key, base_url=base_url, api_key_env=api_key_env)
