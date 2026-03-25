"""In-memory (ephemeral) implementation of MemoryBackend.

Everything is lost when the process exits. This is the default backend
when the user does not configure persistent storage.
"""

from __future__ import annotations

import uuid


class InMemoryStore:
    """Non-persistent memory backend backed by plain dicts."""

    def __init__(self) -> None:
        self._core: dict[str, str] = {}
        self._conversations: dict[str, list[str]] = {}
        self._entries: list[_Entry] = []

    def store(
        self,
        content: str,
        tags: list[str] | None = None,
        tier: str = "episodes",
    ) -> str:
        entry_id = uuid.uuid4().hex[:12]
        entry = _Entry(
            entry_id=entry_id,
            content=content,
            tags=tuple(tags) if tags else (),
            tier=tier,
        )
        self._entries = [*self._entries, entry]
        return entry_id

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        keywords = query.lower().split()
        scored: list[tuple[int, str]] = []

        for entry in self._entries:
            score = _score_entry(entry, keywords)
            if score > 0:
                scored.append((score, entry.content))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [content for _, content in scored[:top_k]]

    def update_core(self, key: str, content: str) -> None:
        self._core = {**self._core, key: content}

    def get_core(self) -> dict[str, str]:
        return dict(self._core)

    def get_conversation(self, session_id: str, window: int = 20) -> list[str]:
        messages = self._conversations.get(session_id, [])
        return list(messages[-window:])

    def save_conversation(self, session_id: str, messages: list[str]) -> None:
        self._conversations = {**self._conversations, session_id: list(messages)}


class _Entry:
    """Internal storage record (not exposed outside this module)."""

    __slots__ = ("content", "entry_id", "tags", "tier")

    def __init__(
        self,
        entry_id: str,
        content: str,
        tags: tuple[str, ...],
        tier: str,
    ) -> None:
        self.entry_id = entry_id
        self.content = content
        self.tags = tags
        self.tier = tier


def _score_entry(entry: _Entry, keywords: list[str]) -> int:
    """Score an entry by counting word-boundary keyword matches."""
    import re

    combined = f"{' '.join(entry.tags)} {entry.content}".lower()
    score = 0
    for kw in keywords:
        pattern = re.compile(rf"\b{re.escape(kw)}\b")
        score += len(pattern.findall(combined))
    return score
