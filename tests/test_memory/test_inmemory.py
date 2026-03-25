"""Tests for InMemoryStore — the default non-persistent memory backend."""

from pop.memory import InMemoryStore


class TestInMemoryStoreBasic:
    """Core store and retrieve functionality."""

    def test_store_and_retrieve_basic_content(self) -> None:
        store = InMemoryStore()
        store.store("The weather today is sunny and warm", tags=["weather"])
        results = store.retrieve("sunny")
        assert len(results) == 1
        assert "sunny" in results[0]

    def test_retrieve_with_keyword_matching(self) -> None:
        store = InMemoryStore()
        store.store("Python is great for data science")
        store.store("JavaScript powers the web")
        store.store("Python and JavaScript are both popular")

        results = store.retrieve("Python")
        assert len(results) == 2
        assert all("Python" in r for r in results)

    def test_store_with_tags_and_retrieve_by_tag(self) -> None:
        store = InMemoryStore()
        store.store("Meeting notes from Monday", tags=["meeting", "monday"])
        store.store("Shopping list for groceries", tags=["personal"])

        results = store.retrieve("meeting")
        assert len(results) == 1
        assert "Meeting notes" in results[0]

    def test_retrieve_returns_empty_for_no_matches(self) -> None:
        store = InMemoryStore()
        store.store("Something about cats")
        results = store.retrieve("quantum physics")
        assert results == []

    def test_top_k_limits_results(self) -> None:
        store = InMemoryStore()
        for i in range(10):
            store.store(f"Document {i} about machine learning")

        results = store.retrieve("machine learning", top_k=3)
        assert len(results) == 3

    def test_store_returns_entry_id(self) -> None:
        store = InMemoryStore()
        entry_id = store.store("Some content")
        assert isinstance(entry_id, str)
        assert len(entry_id) > 0

    def test_store_to_different_tiers(self) -> None:
        store = InMemoryStore()
        id1 = store.store("Episode content", tier="episodes")
        id2 = store.store("Knowledge content", tier="knowledge")
        assert id1 != id2


class TestInMemoryCoreMemory:
    """Core memory (always-in-context) operations."""

    def test_update_core_and_get_core(self) -> None:
        store = InMemoryStore()
        store.update_core("persona", "I am a helpful assistant")
        store.update_core("user_facts", "User prefers Python")

        core = store.get_core()
        assert core["persona"] == "I am a helpful assistant"
        assert core["user_facts"] == "User prefers Python"

    def test_update_core_overwrites_existing(self) -> None:
        store = InMemoryStore()
        store.update_core("persona", "Version 1")
        store.update_core("persona", "Version 2")

        core = store.get_core()
        assert core["persona"] == "Version 2"

    def test_get_core_returns_empty_dict_initially(self) -> None:
        store = InMemoryStore()
        assert store.get_core() == {}


class TestInMemoryConversation:
    """Conversation history operations."""

    def test_save_and_get_conversation(self) -> None:
        store = InMemoryStore()
        messages = ["Hello", "Hi there", "How are you?"]
        store.save_conversation("session-1", messages)

        result = store.get_conversation("session-1")
        assert result == messages

    def test_get_conversation_with_window_limit(self) -> None:
        store = InMemoryStore()
        messages = [f"Message {i}" for i in range(30)]
        store.save_conversation("session-2", messages)

        result = store.get_conversation("session-2", window=5)
        assert len(result) == 5
        assert result == messages[-5:]

    def test_get_conversation_missing_session_returns_empty(self) -> None:
        store = InMemoryStore()
        result = store.get_conversation("nonexistent")
        assert result == []
