"""Memory backend protocol — the contract all memory implementations must satisfy."""

from __future__ import annotations

from typing import Protocol


class MemoryBackend(Protocol):
    """Protocol for memory storage backends.

    Implementations must provide all methods. The framework ships two:
    - InMemoryStore: ephemeral, zero-config (default)
    - MarkdownMemory: persistent, human-readable markdown files on disk
    """

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """Search memory and return top_k most relevant content strings."""
        ...

    def store(
        self, content: str, tags: list[str] | None = None, tier: str = "episodes"
    ) -> str:
        """Store content and return a unique entry ID."""
        ...

    def update_core(self, key: str, content: str) -> None:
        """Create or overwrite a core memory entry by key."""
        ...

    def get_core(self) -> dict[str, str]:
        """Return all core memory entries as {key: content}."""
        ...

    def get_conversation(self, session_id: str, window: int = 20) -> list[str]:
        """Return the last `window` messages for a session."""
        ...

    def save_conversation(self, session_id: str, messages: list[str]) -> None:
        """Persist conversation messages for a session."""
        ...
