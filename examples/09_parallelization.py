"""Demo: Parallelization (Workflow)

Pattern from: https://www.anthropic.com/engineering/building-effective-agents

Two sub-patterns:
  1. Sectioning — split a task into independent parallel subtasks
  2. Voting — run the same task multiple times for consensus

This demo uses SECTIONING to analyze a business idea from three angles
simultaneously, then aggregates the results.
"""

import asyncio

from pop import model, parallel

# ── Setup ──────────────────────────────────────────────────────────────

adapter = model("openai", "gpt-4o")

# ── Parallel analysis ──────────────────────────────────────────────────

business_idea = (
    "An AI-powered meal planning app that generates grocery lists "
    "and suggests recipes based on dietary restrictions and local store sales"
)

analysis_tasks = [
    # Market perspective
    (
        "You are a market analyst. In 3-4 sentences, evaluate the market "
        "opportunity for this business idea. Consider target audience size "
        "and competition:\n\n{context}"
    ),
    # Technical feasibility
    (
        "You are a CTO. In 3-4 sentences, evaluate the technical feasibility "
        "of this business idea. Consider what APIs and data sources are "
        "needed:\n\n{context}"
    ),
    # Revenue model
    (
        "You are a business strategist. In 3-4 sentences, propose the best "
        "revenue model for this business idea. Consider pricing tiers "
        "and monetization:\n\n{context}"
    ),
]

labels = ["Market Analysis", "Technical Feasibility", "Revenue Model"]


async def main() -> None:
    print(f"Business Idea: {business_idea}\n")
    print("Running 3 analyses in parallel...\n")

    results = await parallel(adapter, tasks=analysis_tasks, context=business_idea)

    for label, result in zip(labels, results, strict=True):
        print(f"=== {label} ===")
        print(result)
        print()


if __name__ == "__main__":
    asyncio.run(main())
