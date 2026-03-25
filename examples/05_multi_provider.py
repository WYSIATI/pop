"""Example: Multi-Provider Support
Requires: OPENAI_API_KEY and/or ANTHROPIC_API_KEY environment variables

Demonstrates that switching LLM providers is a one-string change.
The same agent definition works across OpenAI, Anthropic, and any
registered provider. You can also pass a fallback list for resilience.
"""

from pop import Agent, tool


@tool
def summarize_data(text: str) -> str:
    """Summarize a block of text into bullet points.

    Args:
        text: The text to summarize.
    """
    return f"Summary of {len(text)} chars: [processed by tool]"


SAMPLE_TASK = "Summarize the key points of quantum computing for a beginner."

# --- Option 1: OpenAI ---
openai_agent = Agent(
    model="openai:gpt-4o",
    tools=[summarize_data],
    instructions="You are a helpful science explainer.",
)

# --- Option 2: Anthropic ---
anthropic_agent = Agent(
    model="anthropic:claude-sonnet-4-20250514",
    tools=[summarize_data],
    instructions="You are a helpful science explainer.",
)

# --- Option 3: Fallback list ---
# If the first model fails (rate limit, outage), pop tries the next one.
resilient_agent = Agent(
    model=[
        "anthropic:claude-sonnet-4-20250514",
        "openai:gpt-4o",
        "openai:gpt-4o-mini",
    ],
    tools=[summarize_data],
    instructions="You are a helpful science explainer.",
)

# Run whichever agent you prefer -- the API is identical.
result = openai_agent.run(SAMPLE_TASK)
print(f"[OpenAI] {result.output[:120]}...")
print(f"  Model used: {result.steps[0].model_used if result.steps else 'n/a'}")
print(f"  Cost: ${result.cost:.6f}")
