"""Model router — maps provider:model strings to adapters with fallback."""

from __future__ import annotations

from typing import Any, Callable

from pop.models.base import ModelAdapter
from pop.types import Message, ModelResponse, ToolDefinition


AdapterFactory = Callable[..., ModelAdapter]


def parse_model_string(model_str: str) -> tuple[str, str]:
    """Split 'provider:model' into (provider, model).

    The first colon is the delimiter — everything after it is the model name,
    which may itself contain colons (e.g. 'openai:ft:gpt-4o:my-org').
    """
    if ":" not in model_str:
        raise ValueError(
            f"Model string '{model_str}' must be in 'provider:model' format"
        )

    provider, _, model = model_str.partition(":")

    if not provider:
        raise ValueError("Provider cannot be empty in model string")
    if not model:
        raise ValueError("Model cannot be empty in model string")

    return provider, model


class ModelRouter:
    """Registry of provider adapter factories with fallback support."""

    def __init__(self) -> None:
        self._providers: dict[str, AdapterFactory] = {}

    @property
    def providers(self) -> dict[str, AdapterFactory]:
        return dict(self._providers)

    def register(self, name: str, factory: AdapterFactory) -> None:
        """Register an adapter factory for a provider name."""
        self._providers = {**self._providers, name: factory}

    def get_adapter(self, provider: str, model: str, **kwargs: Any) -> ModelAdapter:
        """Get an adapter instance for the given provider and model."""
        factory = self._providers.get(provider)
        if factory is None:
            raise ValueError(
                f"Unknown provider '{provider}'. "
                f"Available: {', '.join(sorted(self._providers.keys())) or 'none'}"
            )
        return factory(model, **kwargs)

    def from_model_string(self, model_str: str, **kwargs: Any) -> ModelAdapter:
        """Parse a 'provider:model' string and return the adapter."""
        provider, model = parse_model_string(model_str)
        return self.get_adapter(provider, model, **kwargs)

    def register_provider(
        self,
        name: str,
        base_url: str,
        api_key_env: str,
        protocol: str,
    ) -> None:
        """Register a custom provider using a known protocol (openai or anthropic)."""
        if protocol == "openai":
            from pop.models.openai import OpenAIAdapter

            def openai_factory(model: str, **kwargs: Any) -> ModelAdapter:
                return OpenAIAdapter(  # type: ignore[return-value]
                    model,
                    base_url=base_url,
                    api_key_env=api_key_env,
                    **kwargs,
                )

            self.register(name, openai_factory)

        elif protocol == "anthropic":
            from pop.models.anthropic import AnthropicAdapter

            def anthropic_factory(model: str, **kwargs: Any) -> ModelAdapter:
                return AnthropicAdapter(  # type: ignore[return-value]
                    model,
                    base_url=base_url,
                    api_key_env=api_key_env,
                    **kwargs,
                )

            self.register(name, anthropic_factory)

        else:
            raise ValueError(
                f"Unknown protocol '{protocol}'. Supported: 'openai', 'anthropic'"
            )

    async def chat_with_fallback(
        self,
        model_strings: list[str],
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
    ) -> ModelResponse:
        """Try each model in sequence, falling back on failure."""
        if not model_strings:
            raise ValueError("Fallback chain requires at least one model string")

        errors: list[tuple[str, Exception]] = []

        for model_str in model_strings:
            adapter = self.from_model_string(model_str)
            try:
                return await adapter.chat(messages, tools)
            except Exception as exc:
                errors = [*errors, (model_str, exc)]

        error_summary = "; ".join(
            f"{ms}: {type(e).__name__}: {e}" for ms, e in errors
        )
        raise RuntimeError(f"All models failed. Errors: {error_summary}")
