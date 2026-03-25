"""Memory system with markdown-based persistence."""

from __future__ import annotations

from pop.memory.base import MemoryBackend

__all__ = ["InMemoryStore", "MarkdownMemory", "MemoryBackend"]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "InMemoryStore": ("pop.memory.inmemory", "InMemoryStore"),
    "MarkdownMemory": ("pop.memory.markdown", "MarkdownMemory"),
}


def __getattr__(name: str) -> object:
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        import importlib

        module = importlib.import_module(module_path)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'pop.memory' has no attribute {name!r}")
