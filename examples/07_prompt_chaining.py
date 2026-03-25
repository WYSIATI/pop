"""Demo: Prompt Chaining (Workflow)

Pattern from: https://www.anthropic.com/engineering/building-effective-agents

Decompose a task into a sequence of LLM calls, where each step's output
feeds into the next. Trade latency for higher accuracy by breaking complex
work into focused subtasks.

This demo: Generate a product description → Translate to Spanish → Create
a tweet-length summary.
"""

import asyncio

from pop import chain, model

# ── Setup ──────────────────────────────────────────────────────────────

adapter = model("openai", "gpt-4o")

# ── Chain ──────────────────────────────────────────────────────────────

steps = [
    # Step 1: Generate a product description from a brief
    (
        "Write a compelling 2-sentence product description for: {input}. "
        "Focus on benefits, not features."
    ),
    # Step 2: Translate to Spanish
    (
        "Translate the following product description to Spanish. "
        "Output only the Spanish text:\n\n{prev}"
    ),
    # Step 3: Create a tweet
    (
        "Create a catchy tweet (under 280 chars) in Spanish "
        "based on this product description:\n\n{prev}"
    ),
]


async def main() -> None:
    input_text = (
        "A noise-cancelling wireless headphone with 30-hour battery life, "
        "perfect for remote workers"
    )

    print(f"Input: {input_text}\n")
    result = await chain(adapter, steps=steps, input_text=input_text)
    print(f"Final tweet:\n{result}")


if __name__ == "__main__":
    asyncio.run(main())
