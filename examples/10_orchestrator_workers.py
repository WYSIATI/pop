"""Demo: Orchestrator-Workers (Workflow)

Pattern from: https://www.anthropic.com/engineering/building-effective-agents

A central LLM (orchestrator) dynamically breaks down a task and delegates
subtasks to worker agents. Unlike parallelization, the subtasks are NOT
pre-defined — the orchestrator decides what work is needed.

This demo: An orchestrator plans a blog post, delegating research,
writing, and SEO optimization to specialized worker agents.
"""

import asyncio

from pop import Agent, orchestrate

# ── Worker agents ──────────────────────────────────────────────────────

researcher = Agent(
    model="openai:gpt-4o",
    name="researcher",
    instructions=(
        "You are a research specialist. When given a topic, provide 3-4 key "
        "facts or statistics with brief explanations. Be concise and factual."
    ),
    max_steps=3,
)

writer = Agent(
    model="openai:gpt-4o",
    name="writer",
    instructions=(
        "You are a blog writer. Write engaging, clear content based on the "
        "information provided. Keep paragraphs short and use active voice."
    ),
    max_steps=3,
)

seo_optimizer = Agent(
    model="openai:gpt-4o",
    name="seo_optimizer",
    instructions=(
        "You are an SEO specialist. Given content, suggest a title, meta "
        "description (under 160 chars), and 5 relevant keywords."
    ),
    max_steps=3,
)

# ── Orchestrator ───────────────────────────────────────────────────────

boss = Agent(
    model="openai:gpt-4o",
    name="boss",
    instructions=(
        "You are a content strategist orchestrating a blog post creation. "
        "You have workers available for research, writing, and SEO. "
        "Coordinate them to produce a short blog post outline with SEO metadata. "
        "Synthesize all worker outputs into a cohesive final deliverable."
    ),
    max_steps=5,
)

# ── Run ────────────────────────────────────────────────────────────────


async def main() -> None:
    task = "Create a short blog post about the benefits of remote work in 2025"
    print(f"Task: {task}\n")

    result = await orchestrate(boss, [researcher, writer, seo_optimizer], task)

    print(f"Orchestrator output:\n{result.output}\n")
    print("--- Details ---")
    print(f"Steps: {len(result.steps)}")
    print(f"Cost: ${result.cost:.6f}")


if __name__ == "__main__":
    asyncio.run(main())
