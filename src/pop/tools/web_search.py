"""Built-in web search tool using DuckDuckGo."""

from __future__ import annotations

from typing import Any

from pop.types import ToolDefinition


def WebSearch(max_results: int = 5) -> ToolDefinition:
    """Create a web search tool powered by DuckDuckGo.

    Requires: pip install pop-framework[tools]
    """

    def _search(query: str) -> str:
        """Search the web for current information.

        Args:
            query: The search query string.
        """
        try:
            from duckduckgo_search import DDGS
        except ImportError as exc:
            raise ImportError(
                "WebSearch requires 'duckduckgo-search'. "
                "Install with: pip install pop-framework[tools]"
            ) from exc

        results = DDGS().text(query, max_results=max_results)
        if not results:
            return "No results found."
        lines = [
            f"{i + 1}. {r['title']}\n   {r['href']}\n   {r['body']}"
            for i, r in enumerate(results)
        ]
        return "\n\n".join(lines)

    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query string",
            },
        },
        "required": ["query"],
    }

    return ToolDefinition(
        name="web_search",
        description="Search the web for current information using DuckDuckGo.",
        parameters=parameters,
        function=_search,
        is_async=False,
    )
