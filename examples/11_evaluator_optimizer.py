"""Demo: Evaluator-Optimizer (Workflow)

Pattern from: https://www.anthropic.com/engineering/building-effective-agents

An iterative loop where one LLM generates a response and another evaluates
it, providing feedback for improvement. The loop continues until the
evaluator approves or max rounds are reached.

This is the "debate" pattern in pop — a generator and a critic refine
output collaboratively.

This demo: Write a concise product landing page headline, refined
through critique rounds.
"""

import asyncio

from pop import Agent, debate

# ── Generator: creative copywriter ─────────────────────────────────────

generator = Agent(
    model="openai:gpt-4o",
    name="copywriter",
    instructions=(
        "You are an expert copywriter. Write a compelling product landing page "
        "headline and a 2-sentence subheadline. Be creative, concise, and "
        "benefit-focused. Avoid cliches and jargon."
    ),
    max_steps=3,
)

# ── Evaluator: marketing director ──────────────────────────────────────

critic = Agent(
    model="openai:gpt-4o",
    name="marketing_director",
    instructions=(
        "You are a senior marketing director reviewing landing page copy. "
        "Evaluate for: clarity, emotional impact, specificity, and brevity. "
        "If the copy is strong, respond with APPROVED and explain why. "
        "If it needs work, respond with REJECTED and give specific, actionable feedback."
    ),
    max_steps=3,
)

# ── Run ────────────────────────────────────────────────────────────────


async def main() -> None:
    task = (
        "Write a landing page headline and subheadline for a new AI-powered "
        "email client that saves professionals 2 hours per day by automatically "
        "drafting replies, scheduling meetings, and prioritizing inbox."
    )
    print(f"Task: {task}\n")

    result = await debate(generator, critic, task, max_rounds=3)

    print(f"Final copy:\n{result.output}\n")
    print("--- Details ---")
    print(f"Rounds: {result.rounds}")
    print(f"Approved: {result.approved}")
    print(f"Debate history ({len(result.history)} exchanges):")
    for i, entry in enumerate(result.history):
        role = "Generator" if i % 2 == 0 else "Critic"
        preview = entry[:100] + "..." if len(entry) > 100 else entry
        print(f"  [{role}] {preview}")


if __name__ == "__main__":
    asyncio.run(main())
