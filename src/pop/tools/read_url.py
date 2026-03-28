"""Built-in URL reading tool using httpx."""

from __future__ import annotations

import re
from typing import Any

from pop.types import ToolDefinition


def ReadURL(timeout: int = 10) -> ToolDefinition:
    """Create a tool that fetches and returns text content from a URL.

    Uses httpx (already a pop dependency) — no extras needed.
    """

    def _read_url(url: str) -> str:
        """Fetch and return the text content of a URL.

        Args:
            url: The URL to fetch.
        """
        import httpx

        response = httpx.get(url, timeout=timeout, follow_redirects=True)
        response.raise_for_status()

        text = response.text
        # Strip HTML tags for a cleaner text result
        text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        # Truncate to avoid flooding context
        max_chars = 8000
        if len(text) > max_chars:
            text = text[:max_chars] + "... [truncated]"

        return text

    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to fetch",
            },
        },
        "required": ["url"],
    }

    return ToolDefinition(
        name="read_url",
        description="Fetch and return the text content of a URL.",
        parameters=parameters,
        function=_read_url,
        is_async=False,
    )
