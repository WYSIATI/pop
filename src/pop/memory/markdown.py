"""Markdown-file-based persistent memory backend.

Uses the filesystem as the database — every memory entry is a human-readable
.md file with YAML-style frontmatter. No external dependencies required.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)

_TIER_DIRS = ("core", "conversations", "episodes", "knowledge")

_TIER_TO_TYPE: dict[str, str] = {
    "episodes": "episode",
    "knowledge": "knowledge",
}


class MarkdownMemory:
    """Persistent memory that stores everything as markdown files on disk.

    Directory layout:
        {base_dir}/
        ├── core/           # Tier 1 — always in context
        ├── conversations/  # Tier 2 — session transcripts
        ├── episodes/       # Tier 3 — past experiences
        └── knowledge/      # Tier 4 — domain knowledge
    """

    def __init__(self, base_dir: str) -> None:
        self._base = Path(base_dir)
        self._ensure_dirs()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store(
        self,
        content: str,
        tags: list[str] | None = None,
        tier: str = "episodes",
    ) -> str:
        entry_id = uuid.uuid4().hex[:12]
        entry_type = _TIER_TO_TYPE.get(tier, tier)
        frontmatter = _build_frontmatter(entry_type, tags or [])
        file_content = f"---\n{frontmatter}---\n\n{content}\n"

        tier_dir = self._base / tier
        filename = f"{entry_type}_{entry_id}.md"
        tier_dir.joinpath(filename).write_text(file_content, encoding="utf-8")

        return entry_id

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        keywords = query.lower().split()
        scored: list[tuple[int, str]] = []

        for tier in ("episodes", "knowledge"):
            tier_dir = self._base / tier
            if not tier_dir.exists():
                continue
            for md_file in tier_dir.glob("*.md"):
                raw = md_file.read_text(encoding="utf-8")
                body = _extract_body(raw)
                fm_text = _extract_frontmatter_text(raw)
                score = _score_text(body, fm_text, keywords)
                if score > 0:
                    scored.append((score, body))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [body for _, body in scored[:top_k]]

    def update_core(self, key: str, content: str) -> None:
        core_file = self._base / "core" / f"{key}.md"
        core_file.write_text(content, encoding="utf-8")

    def get_core(self) -> dict[str, str]:
        core_dir = self._base / "core"
        result: dict[str, str] = {}
        if not core_dir.exists():
            return result
        for md_file in sorted(core_dir.glob("*.md")):
            key = md_file.stem
            result[key] = md_file.read_text(encoding="utf-8")
        return result

    def save_conversation(self, session_id: str, messages: list[str]) -> None:
        conv_file = self._base / "conversations" / f"session_{session_id}.md"
        lines = [_format_message(i, msg) for i, msg in enumerate(messages)]
        conv_file.write_text("\n\n".join(lines) + "\n", encoding="utf-8")

    def get_conversation(self, session_id: str, window: int = 20) -> list[str]:
        conv_file = self._base / "conversations" / f"session_{session_id}.md"
        if not conv_file.exists():
            return []
        raw = conv_file.read_text(encoding="utf-8")
        messages = _parse_conversation(raw)
        return messages[-window:]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_dirs(self) -> None:
        for tier in _TIER_DIRS:
            (self._base / tier).mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# Pure functions (no side effects, easy to test)
# ------------------------------------------------------------------


def _build_frontmatter(entry_type: str, tags: list[str]) -> str:
    """Build YAML-style frontmatter string (without the --- delimiters)."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        f"type: {entry_type}",
        f"tags: [{', '.join(tags)}]",
        f"timestamp: {timestamp}",
    ]
    return "\n".join(lines) + "\n"


def _extract_body(raw: str) -> str:
    """Extract the body content after frontmatter."""
    match = _FRONTMATTER_RE.match(raw)
    if match:
        return raw[match.end() :].strip()
    return raw.strip()


def _extract_frontmatter_text(raw: str) -> str:
    """Extract the raw frontmatter text for keyword matching."""
    match = _FRONTMATTER_RE.match(raw)
    if match:
        return match.group(1)
    return ""


def _score_text(body: str, frontmatter: str, keywords: list[str]) -> int:
    """Score content by counting keyword occurrences in body and frontmatter."""
    body_lower = body.lower()
    fm_lower = frontmatter.lower()
    score = 0
    for kw in keywords:
        score += body_lower.count(kw)
        score += fm_lower.count(kw)
    return score


def _format_message(index: int, message: str) -> str:
    """Format a single conversation message for storage."""
    return f"**[{index}]** {message}"


def _parse_conversation(raw: str) -> list[str]:
    """Parse stored conversation back into a list of message strings."""
    pattern = re.compile(r"\*\*\[\d+\]\*\* ")
    parts = pattern.split(raw)
    # First element is empty (before first match), skip it
    return [part.strip() for part in parts if part.strip()]
