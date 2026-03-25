"""Model router and provider adapters."""

from __future__ import annotations

from typing import Any

from pop.models.base import ModelAdapter, StreamChunk
from pop.models.router import ModelRouter
from pop.types import Message, ModelResponse, ToolDefinition

# Global router instance
_router = ModelRouter()


def register_provider(
    name: str,
    base_url: str,
    api_key_env: str,
    protocol: str,
) -> None:
    """Register a custom provider on the global router."""
    _router.register_provider(name, base_url, api_key_env, protocol)


def model(provider: str, name: str, **kwargs: Any) -> ModelAdapter:
    """Create a configured ModelAdapter for the given provider and model."""
    return _router.get_adapter(provider, name, **kwargs)


async def chat(model_str: str, message: str) -> ModelResponse:
    """Convenience: single LLM call with a user message string."""
    adapter = _router.from_model_string(model_str)
    return await adapter.chat([Message.user(message)])


__all__ = [
    "ModelAdapter",
    "ModelRouter",
    "StreamChunk",
    "chat",
    "model",
    "register_provider",
]
