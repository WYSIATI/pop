"""Demo: Routing (Workflow)

Pattern from: https://www.anthropic.com/engineering/building-effective-agents

Classify an input and direct it to a specialized handler. This allows
each route to be optimized for its specific task, rather than having one
prompt handle everything.

This demo routes customer support messages to the right department.
"""

import asyncio

from pop import model, route

# ── Setup ──────────────────────────────────────────────────────────────

adapter = model("openai", "gpt-4o")

# ── Route handlers ─────────────────────────────────────────────────────


def handle_billing(text: str) -> str:
    return (
        f"[BILLING DEPT] Processing billing request: '{text[:60]}...'\n"
        "→ Checking account status, generating invoice summary."
    )


def handle_technical(text: str) -> str:
    return (
        f"[TECH SUPPORT] Processing technical issue: '{text[:60]}...'\n"
        "→ Running diagnostics, checking known issues database."
    )


def handle_general(text: str) -> str:
    return (
        f"[GENERAL SUPPORT] Processing inquiry: '{text[:60]}...'\n"
        "→ Preparing FAQ-based response."
    )


# ── Route definitions ──────────────────────────────────────────────────

routes = {
    "billing": handle_billing,
    "technical": handle_technical,
    "general": handle_general,
}

# ── Test messages ──────────────────────────────────────────────────────

test_messages = [
    "I was charged twice for my subscription last month, can you help?",
    "My app keeps crashing when I try to upload files larger than 10MB.",
    "What are your business hours and do you have a physical office?",
]


async def main() -> None:
    for msg in test_messages:
        print(f"Customer: {msg}")
        result = await route(adapter, input_text=msg, routes=routes)
        print(f"Routed:   {result}\n")


if __name__ == "__main__":
    asyncio.run(main())
