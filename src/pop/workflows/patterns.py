"""Pre-agent workflow patterns: chain, route, parallel.

Level 0-3 patterns that compose LLM calls without a full agent loop.
All functions accept a ModelAdapter directly for testability.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from pop.types import Message

if TYPE_CHECKING:
    from collections.abc import Callable

    from pop.models.base import ModelAdapter


async def chain(
    model_adapter: ModelAdapter,
    steps: list[str],
    input_text: str,
) -> str:
    """Sequential prompt chain where each step feeds into the next.

    Each step is a prompt template with optional {input} and {prev} placeholders.
    Step 1 receives {input}=input_text, {prev}="".
    Subsequent steps receive {prev}=output of the previous step.

    Returns the final step's output.
    """
    if not steps:
        raise ValueError("Chain requires at least one step")

    prev = ""
    for step_template in steps:
        prompt = step_template.replace("{input}", input_text).replace("{prev}", prev)
        messages = [Message.user(prompt)]
        response = await model_adapter.chat(messages)
        prev = response.content

    return prev


async def route(
    model_adapter: ModelAdapter,
    input_text: str,
    routes: dict[str, Callable[[str], Any]],
) -> Any:
    """Classify input using the model, then dispatch to the matching handler.

    Asks the model to classify input_text into one of the route keys,
    then calls the matching handler with the original input_text.

    Raises ValueError if the model returns a classification not in routes.
    """
    route_names = ", ".join(sorted(routes.keys()))
    classification_prompt = (
        f"Classify the following input into exactly one of these categories: "
        f"{route_names}\n\n"
        f"Input: {input_text}\n\n"
        f"Respond with only the category name, nothing else."
    )

    # Normalize route keys to lowercase for case-insensitive matching
    normalized_routes = {k.lower(): v for k, v in routes.items()}

    messages = [Message.user(classification_prompt)]
    response = await model_adapter.chat(messages)
    category = response.content.strip().lower()

    handler = normalized_routes.get(category)
    if handler is None:
        available = ", ".join(sorted(routes.keys()))
        raise ValueError(
            f"Model returned unknown route '{category}'. Available routes: {available}"
        )

    return handler(input_text)


async def parallel(
    model_adapter: ModelAdapter,
    tasks: list[str],
    context: str = "",
) -> list[str]:
    """Run multiple prompts concurrently and return results in order.

    Each task string is a prompt template with an optional {context} placeholder.
    Returns a list of model response strings in the same order as input tasks.
    """
    if not tasks:
        return []

    async def _run_task(task_template: str) -> str:
        prompt = task_template.replace("{context}", context)
        messages = [Message.user(prompt)]
        response = await model_adapter.chat(messages)
        return response.content

    results = await asyncio.gather(*[_run_task(task) for task in tasks])
    return list(results)
