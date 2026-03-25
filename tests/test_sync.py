"""Tests for the shared sync-over-async helper."""

from __future__ import annotations

from pop._sync import run_sync


async def _async_add(a: int, b: int) -> int:
    return a + b


def test_run_sync_basic():
    result = run_sync(_async_add(2, 3))
    assert result == 5


def test_run_sync_returns_correct_type():
    result = run_sync(_async_add(10, 20))
    assert isinstance(result, int)
    assert result == 30
