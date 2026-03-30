# Multi-Agent Patterns

pop provides multiple patterns for composing agents, from simple handoffs to full orchestration.

## Handoff

Route tasks between specialized agents. A triage agent decides which specialist handles each request.

```python
from pop import Agent, tool, handoff

@tool
def lookup_invoice(invoice_id: str) -> str:
    """Look up an invoice by ID."""
    return f"Invoice {invoice_id}: $299, paid 2024-01-15"

@tool
def check_logs(error_code: str) -> str:
    """Check system logs for an error code."""
    return f"Error {error_code}: timeout on upstream service"

billing = Agent(
    model="openai:gpt-4o-mini",
    tools=[lookup_invoice],
    instructions="You handle billing and payment questions.",
)
tech = Agent(
    model="openai:gpt-4o-mini",
    tools=[check_logs],
    instructions="You handle technical support and debugging.",
)

triage = Agent(
    model="openai:gpt-4o-mini",
    tools=[
        handoff(billing, when="billing or payment issues"),
        handoff(tech, when="technical problems or errors"),
    ],
)

result = triage.run("I was charged twice for my subscription")
# -> automatically routes to billing agent
```

**Shorthand: `workers` parameter.** Auto-wires handoff tools without explicit `handoff()` calls:

```python
triage = Agent(
    model="openai:gpt-4o-mini",
    workers=[billing, tech],
)
# Equivalent to tools=[handoff(billing), handoff(tech)]
```

## Pipeline

Sequential processing: each agent's output feeds into the next.

```python
from pop import Agent, pipeline

researcher = Agent(model="openai:gpt-4o", tools=[search], instructions="Research thoroughly.")
writer = Agent(model="openai:gpt-4o", instructions="Write clear, concise reports.")
editor = Agent(model="openai:gpt-4o", instructions="Review for accuracy and clarity.")

# research -> write -> edit
result = await pipeline([researcher, writer, editor], task="Report on AI agent frameworks in 2026")
```

## Debate

Generator-critic loop: one agent proposes, another critiques, repeating until approved.

```python
from pop import Agent, debate

writer = Agent(model="openai:gpt-4o", instructions="Write clear, concise reports.")
editor = Agent(model="openai:gpt-4o", instructions="Review for accuracy and clarity.")

result = await debate(writer, editor, task="Write a product announcement", max_rounds=3)
```

## Other Patterns

| Pattern | Import | Purpose |
|---------|--------|---------|
| `handoff` | `from pop import handoff` | Route to specialist agents |
| `pipeline` | `from pop import pipeline` | Sequential agent chain |
| `debate` | `from pop import debate` | Generator-critic loop |
| `orchestrate` | `from pop import orchestrate` | Coordinator dispatches to workers |
| `fan_out` | `from pop import fan_out` | Parallel execution, collect results |
