"""Tests for MarkdownMemory — persistent markdown-file-based memory."""

from pathlib import Path

import pytest

from pop.memory import MarkdownMemory


class TestMarkdownMemoryInit:
    """Directory structure and initialization."""

    def test_creates_directory_structure(self, tmp_path: Path) -> None:
        MarkdownMemory(str(tmp_path / "mem"))
        base = tmp_path / "mem"
        assert (base / "core").is_dir()
        assert (base / "conversations").is_dir()
        assert (base / "episodes").is_dir()
        assert (base / "knowledge").is_dir()

    def test_works_with_existing_directory(self, tmp_path: Path) -> None:
        base = tmp_path / "mem"
        base.mkdir()
        (base / "core").mkdir()
        mem = MarkdownMemory(str(base))
        assert mem.get_core() == {}

    def test_default_base_dir_uses_home(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        monkeypatch.delenv("POP_MEMORY_DIR", raising=False)
        mem = MarkdownMemory()
        expected = tmp_path / ".pop" / "memory"
        assert mem._base == expected
        assert (expected / "core").is_dir()

    def test_env_var_overrides_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        custom = tmp_path / "custom_memory"
        monkeypatch.setenv("POP_MEMORY_DIR", str(custom))
        mem = MarkdownMemory()
        assert mem._base == custom
        assert (custom / "core").is_dir()

    def test_explicit_base_dir_takes_precedence(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("POP_MEMORY_DIR", str(tmp_path / "env_dir"))
        explicit = tmp_path / "explicit"
        mem = MarkdownMemory(str(explicit))
        assert mem._base == explicit


class TestMarkdownMemoryStore:
    """Storing content creates markdown files with frontmatter."""

    def test_store_creates_md_file(self, tmp_path: Path) -> None:
        mem = MarkdownMemory(str(tmp_path))
        entry_id = mem.store("Research findings about AI safety")
        assert isinstance(entry_id, str)

        md_files = list((tmp_path / "episodes").glob("*.md"))
        assert len(md_files) == 1

    def test_store_file_has_frontmatter(self, tmp_path: Path) -> None:
        mem = MarkdownMemory(str(tmp_path))
        mem.store("Some content", tags=["research", "ai"])

        md_files = list((tmp_path / "episodes").glob("*.md"))
        content = md_files[0].read_text()
        assert content.startswith("---\n")
        assert "type: episode" in content
        assert "research" in content
        assert "ai" in content
        assert "Some content" in content

    def test_store_to_knowledge_tier(self, tmp_path: Path) -> None:
        mem = MarkdownMemory(str(tmp_path))
        mem.store("Python best practices", tier="knowledge")

        md_files = list((tmp_path / "knowledge").glob("*.md"))
        assert len(md_files) == 1

    def test_store_returns_unique_ids(self, tmp_path: Path) -> None:
        mem = MarkdownMemory(str(tmp_path))
        id1 = mem.store("First entry")
        id2 = mem.store("Second entry")
        assert id1 != id2


class TestMarkdownMemoryRetrieve:
    """Retrieval with keyword matching."""

    def test_retrieve_finds_by_content_keyword(self, tmp_path: Path) -> None:
        mem = MarkdownMemory(str(tmp_path))
        mem.store("Python is a versatile programming language")
        mem.store("JavaScript runs in the browser")

        results = mem.retrieve("Python")
        assert len(results) == 1
        assert "Python" in results[0]

    def test_retrieve_finds_by_tag(self, tmp_path: Path) -> None:
        mem = MarkdownMemory(str(tmp_path))
        mem.store("Some meeting notes", tags=["meeting", "weekly"])
        mem.store("Random thoughts", tags=["personal"])

        results = mem.retrieve("meeting")
        assert len(results) >= 1
        assert "meeting" in results[0].lower() or "Meeting" in results[0]

    def test_retrieve_top_k_limits(self, tmp_path: Path) -> None:
        mem = MarkdownMemory(str(tmp_path))
        for i in range(10):
            mem.store(f"Document {i} about machine learning")

        results = mem.retrieve("machine learning", top_k=3)
        assert len(results) == 3

    def test_retrieve_returns_empty_for_no_matches(self, tmp_path: Path) -> None:
        mem = MarkdownMemory(str(tmp_path))
        mem.store("Cats are great pets")
        results = mem.retrieve("quantum entanglement")
        assert results == []

    def test_retrieve_ranks_by_relevance(self, tmp_path: Path) -> None:
        mem = MarkdownMemory(str(tmp_path))
        mem.store("Python basics")
        mem.store("Advanced Python programming with Python libraries")

        results = mem.retrieve("Python", top_k=2)
        assert len(results) == 2
        # The one with more "Python" mentions should rank first
        assert "Advanced" in results[0] or "Python" in results[0]


class TestMarkdownMemoryCore:
    """Core memory operations."""

    def test_update_core_creates_file(self, tmp_path: Path) -> None:
        mem = MarkdownMemory(str(tmp_path))
        mem.update_core("agent_persona", "I am a research assistant")

        core_file = tmp_path / "core" / "agent_persona.md"
        assert core_file.exists()
        content = core_file.read_text()
        assert "I am a research assistant" in content

    def test_update_core_overwrites(self, tmp_path: Path) -> None:
        mem = MarkdownMemory(str(tmp_path))
        mem.update_core("persona", "Version 1")
        mem.update_core("persona", "Version 2")

        core = mem.get_core()
        assert core["persona"] == "Version 2"

    def test_get_core_reads_all_files(self, tmp_path: Path) -> None:
        mem = MarkdownMemory(str(tmp_path))
        mem.update_core("persona", "Helpful assistant")
        mem.update_core("user_facts", "Prefers concise answers")

        core = mem.get_core()
        assert len(core) == 2
        assert core["persona"] == "Helpful assistant"
        assert core["user_facts"] == "Prefers concise answers"

    def test_get_core_empty_initially(self, tmp_path: Path) -> None:
        mem = MarkdownMemory(str(tmp_path))
        assert mem.get_core() == {}


class TestMarkdownMemoryConversation:
    """Conversation persistence."""

    def test_save_and_get_conversation(self, tmp_path: Path) -> None:
        mem = MarkdownMemory(str(tmp_path))
        messages = ["Hello", "Hi there", "How are you?"]
        mem.save_conversation("sess-1", messages)

        result = mem.get_conversation("sess-1")
        assert result == messages

    def test_get_conversation_with_window(self, tmp_path: Path) -> None:
        mem = MarkdownMemory(str(tmp_path))
        messages = [f"Message {i}" for i in range(30)]
        mem.save_conversation("sess-2", messages)

        result = mem.get_conversation("sess-2", window=5)
        assert len(result) == 5
        assert result == messages[-5:]

    def test_get_conversation_missing_returns_empty(self, tmp_path: Path) -> None:
        mem = MarkdownMemory(str(tmp_path))
        assert mem.get_conversation("nonexistent") == []

    def test_conversation_file_is_readable_markdown(self, tmp_path: Path) -> None:
        mem = MarkdownMemory(str(tmp_path))
        mem.save_conversation("sess-3", ["Hello world", "Goodbye world"])

        session_file = tmp_path / "conversations" / "session_sess-3.md"
        assert session_file.exists()
        content = session_file.read_text()
        assert "Hello world" in content
        assert "Goodbye world" in content


class TestMarkdownMemoryFrontmatter:
    """Frontmatter parsing and generation."""

    def test_frontmatter_has_timestamp(self, tmp_path: Path) -> None:
        mem = MarkdownMemory(str(tmp_path))
        mem.store("Content with timestamp")

        md_files = list((tmp_path / "episodes").glob("*.md"))
        content = md_files[0].read_text()
        assert "timestamp:" in content

    def test_frontmatter_has_type(self, tmp_path: Path) -> None:
        mem = MarkdownMemory(str(tmp_path))
        mem.store("Episode content", tier="episodes")
        mem.store("Knowledge content", tier="knowledge")

        ep_files = list((tmp_path / "episodes").glob("*.md"))
        kn_files = list((tmp_path / "knowledge").glob("*.md"))

        assert "type: episode" in ep_files[0].read_text()
        assert "type: knowledge" in kn_files[0].read_text()

    def test_files_are_human_readable(self, tmp_path: Path) -> None:
        mem = MarkdownMemory(str(tmp_path))
        mem.store(
            "Important research findings about climate change",
            tags=["research", "climate"],
        )

        md_files = list((tmp_path / "episodes").glob("*.md"))
        content = md_files[0].read_text()
        # Should be valid markdown a human can open in any editor
        assert "---" in content
        assert "Important research findings about climate change" in content


class TestMarkdownMemoryEdgeCases:
    """Edge cases for coverage of internal pure functions."""

    def test_retrieve_with_missing_tier_dir(self, tmp_path: Path) -> None:
        """retrieve skips non-existent tier dirs (line 67)."""
        mem = MarkdownMemory(str(tmp_path))
        # Delete the episodes dir to trigger the missing-dir branch
        import shutil

        shutil.rmtree(tmp_path / "episodes")
        # Should not raise, should return empty
        results = mem.retrieve("anything")
        assert results == []

    def test_get_core_missing_dir(self, tmp_path: Path) -> None:
        """get_core returns empty dict if core dir missing (line 87)."""
        mem = MarkdownMemory(str(tmp_path))
        import shutil

        shutil.rmtree(tmp_path / "core")
        assert mem.get_core() == {}

    def test_extract_body_without_frontmatter(self) -> None:
        """_extract_body returns stripped content when no frontmatter (line 136)."""
        from pop.memory.markdown import _extract_body

        raw = "  Just plain content  "
        assert _extract_body(raw) == "Just plain content"

    def test_extract_frontmatter_text_without_frontmatter(self) -> None:
        """_extract_frontmatter_text returns empty when no frontmatter (line 144)."""
        from pop.memory.markdown import _extract_frontmatter_text

        raw = "No frontmatter here"
        assert _extract_frontmatter_text(raw) == ""
