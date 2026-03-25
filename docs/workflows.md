# Workflow Patterns

pop provides progressive complexity levels. Use the simplest pattern that solves your problem.

## Level 0 -- Single LLM Call

No agent needed. Just call the model directly.

```python
from pop import chat

response = chat("openai:gpt-4o-mini", "Explain quantum computing in one paragraph.")
print(response.content)
```

## Level 1 -- Chain

Sequential prompts where each step feeds into the next.

```python
from pop import chain, model

adapter = model("openai:gpt-4o-mini")
result = await chain(
    adapter,
    steps=[
        "Summarize this topic: {input}",
        "Translate to Spanish: {prev}",
        "Create 3 quiz questions from: {prev}",
    ],
    input_text="The history of the internet",
)
```

## Level 2 -- Route

Classify input and dispatch to the right handler.

```python
from pop import route, model

adapter = model("openai:gpt-4o-mini")
result = await route(
    adapter,
    input_text="My order hasn't arrived yet",
    routes={
        "billing": handle_billing,
        "shipping": handle_shipping,
        "technical": handle_technical,
    },
)
```

## Level 3 -- Parallel

Run multiple prompts concurrently and collect results.

```python
from pop import parallel, model

adapter = model("openai:gpt-4o-mini")
results = await parallel(
    adapter,
    tasks=[
        "List pros of {context}",
        "List cons of {context}",
        "Suggest alternatives to {context}",
    ],
    context="migrating from PostgreSQL to DynamoDB",
)
```

## Level 4 -- Agent with Tools

The full ReAct loop. The agent reasons, picks tools, executes, and loops until done.

```python
from pop import Agent, tool

@tool
def calculate(expression: str) -> float:
    """Evaluate a math expression."""
    return eval(expression)  # use a sandbox in production

agent = Agent(model="openai:gpt-4o", tools=[calculate], max_steps=10)
result = agent.run("What is the compound interest on $10,000 at 5% for 10 years?")
```

## Level 5 -- Multi-Agent Orchestration

Compose agents for complex workflows. See [Multi-Agent Patterns](multi-agent.md) for details on `pipeline`, `orchestrate`, `debate`, and `fan_out`.
