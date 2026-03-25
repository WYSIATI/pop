"""Shared sync-over-async helper.

Both Agent.run() and Runner.run() need to call async methods from sync code.
This module provides a single implementation to avoid duplication.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Coroutine


def run_sync(coro: Coroutine[Any, Any, Any]) -> Any:
    """Run a coroutine synchronously, handling running event loops (e.g. Jupyter)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()

    return asyncio.run(coro)
