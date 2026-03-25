# Skills Guide

How to build AI agents with pop. This guide is written for coding agents (Claude, Cursor, Copilot) and humans alike.

## 5 Core Concepts

| Concept | What it does |
|---------|-------------|
| `Agent` | ReAct loop: thinks, calls tools, repeats until done |
| `@tool` | Turns a Python function into a tool the LLM can call |
| `Runner` | Streams events from an agent run |
| Memory | Pluggable storage (in-memory or markdown files) |
| Patterns | Multi-agent composition: `handoff`, `pipeline`, `debate`, `orchestrate`, `fan_out` |

## Install

```bash
uv add pop-framework
# or
pip install pop-framework
```

## 1. Minimal Agent

```python
from pop import Agent, tool

@tool
def add(a: float, b: float) -> float:
    """Add two numbers.

    Args:
        a: First number.
        b: Second number.
    """
    return a + b

agent = Agent(model="openai:gpt-4o-mini", tools=[add])
result = agent.run("What is 42 + 17?")
print(result.output)
```

**Key points:**
- `@tool` reads the function signature and docstring to generate JSON Schema automatically
- Use Google-style docstrings with `Args:` section for parameter descriptions
- `model` is a string in `provider:model-name` format

## 2. Defining Tools

```python
from pop import tool

# Simple tool
@tool
def search(query: str) -> str:
    """Search the web."""
    return f"Results for {query}"

# Tool with optional params
@tool
def fetch_page(url: str, max_length: int = 5000) -> str:
    """Fetch a web page.

    Args:
        url: The URL to fetch.
        max_length: Max characters to return.
    """
    return httpx.get(url).text[:max_length]

# Tool with Pydantic input
from pydantic import BaseModel, Field

class Contact(BaseModel):
    name: str = Field(description="Full name")
    email: str = Field(description="Email address")

@tool
def save_contact(contact: Contact) -> str:
    """Save a contact to the database.

    Args:
        contact: Contact information.
    """
    return f"Saved {contact.name}"
```

**Supported types:** `str`, `int`, `float`, `bool`, `list[T]`, `dict`, Pydantic models, `Optional[T]`.

## 3. Agent Configuration

```python
agent = Agent(
    model="openai:gpt-4o",          # or "anthropic:claude-sonnet-4-20250514"
    name="researcher",                # name for multi-agent identification
    tools=[search, fetch_page],       # list of @tool-decorated functions
    instructions="You are a research assistant. Cite sources.",
    max_steps=10,                     # max ReAct loop iterations (default: 10)
    max_cost=0.50,                    # USD budget cap (optional)
    max_retries=3,                    # retries on transient errors
    reflect_on_failure=False,         # enable Reflexion loop on errors
    output_guardrails=[is_safe],      # validators for final answer
)
```

**Model fallback:** pass a list to try models in order.

```python
agent = Agent(
    model=["anthropic:claude-sonnet-4-20250514", "openai:gpt-4o", "openai:gpt-4o-mini"],
    tools=[search],
)
```

## 4. Running Agents

### Sync

```python
result = agent.run("What happened in AI today?")
print(result.output)        # final answer string
print(result.cost)           # total USD cost
print(result.token_usage)    # TokenUsage(input_tokens=..., output_tokens=...)
print(result.steps)          # list[Step] -- full execution trace
```

### Async

```python
result = await agent.arun("What happened in AI today?")
```

### Streaming

```python
import asyncio
from pop import Agent, Runner, ToolCallEvent, ToolResultEvent, TextDeltaEvent, DoneEvent, tool

runner = Runner(agent)

async for event in runner.stream("What should I wear in SF?"):
    match event:
        case ToolCallEvent(name=name, args=args):
            print(f"Calling {name}({args})")
        case ToolResultEvent(name=name, output=output):
            print(f"{name} -> {output}")
        case TextDeltaEvent(delta=text):
            print(text, end="")
        case DoneEvent(result=result):
            print(f"\nCost: ${result.cost:.6f}")
```

## 5. Multi-Agent Patterns

### Handoff

Route to specialist agents based on the task.

```python
from pop import Agent, handoff, tool

billing = Agent(model="openai:gpt-4o-mini", tools=[lookup_invoice],
                instructions="Handle billing questions.")
tech = Agent(model="openai:gpt-4o-mini", tools=[check_logs],
             instructions="Handle technical issues.")

triage = Agent(
    model="openai:gpt-4o-mini",
    tools=[
        handoff(billing, when="billing or payment issues"),
        handoff(tech, when="technical problems or errors"),
    ],
)
result = triage.run("I was charged twice")
```

### Pipeline

Sequential: each agent's output feeds into the next.

```python
from pop import Agent, pipeline

result = await pipeline(
    [researcher, writer, editor],
    task="Report on AI agent frameworks",
)
```

### Debate

Generator proposes, critic reviews, loop until approved.

```python
from pop import Agent, debate

result = await debate(writer, editor, task="Write a product announcement", max_rounds=3)
```

### Orchestrate

A boss agent dynamically delegates to workers.

```python
from pop import Agent, orchestrate

boss = Agent(model="openai:gpt-4o", name="boss",
             instructions="Coordinate research, writing, and SEO.")
result = await orchestrate(boss, [researcher, writer, seo_optimizer], task="Write a blog post")
```

### Fan Out

Run multiple agents in parallel, aggregate results.

```python
from pop import Agent, fan_out

result = await fan_out(
    [analyst_a, analyst_b, analyst_c],
    task="Analyze Q4 earnings",
    strategy="majority",  # or "first", "weighted_vote"
)
```

## 6. Workflows (No Agent Loop)

For simpler LLM patterns that don't need tool-calling or a ReAct loop.

```python
from pop import chain, route, parallel, model

m = model("openai:gpt-4o")

# Chain: sequential prompts, {prev} is replaced with the previous output
result = await chain(m, steps=[
    "Summarize: {input}",
    "Translate to French: {prev}",
], input_text="...")

# Route: classify and dispatch
result = await route(m, "refund request", {
    "refund": handle_refund,
    "support": handle_support,
})

# Parallel: concurrent LLM calls
results = await parallel(m, [
    "Summarize: {context}",
    "Extract keywords: {context}",
], context="...")
```

## 7. Memory

### In-Memory (Default)

No configuration needed. State is lost when the process exits.

### Markdown Memory (Persistent)

```python
from pop.memory import MarkdownMemory

# Default: stores in ~/.pop/memory
memory = MarkdownMemory()

# Or specify a custom directory
memory = MarkdownMemory(base_dir="./agent_memory")

agent = Agent(
    model="openai:gpt-4o",
    tools=[search],
    memory=memory,
    core_memory={
        "user_name": "Chester",
        "preferences": "Concise answers. No fluff.",
    },
)
```

Default directory: `~/.pop/memory`. Override with `POP_MEMORY_DIR` env var or the `base_dir` argument.

Core memory (`dict[str, str]`) is always included in context. Episodes and conversations persist as markdown files on disk.

## 8. Hooks

```python
from pop.hooks import ConsoleHook, CostHook, FileLogHook

runner = Runner(agent, hooks=[
    ConsoleHook(),                       # print steps to stdout
    CostHook(budget=1.00),               # track cumulative cost
    FileLogHook(path="./agent.log"),     # append steps to JSON file
])
```

## 9. Providers

7 built-in: OpenAI, Anthropic, Gemini, DeepSeek, Kimi, MiniMax, GLM.

```python
# Switch by changing the string
Agent(model="openai:gpt-4o")
Agent(model="anthropic:claude-sonnet-4-20250514")
Agent(model="gemini:gemini-2.0-flash")
Agent(model="deepseek:deepseek-chat")
```

Custom providers implement `ModelAdapter`:

```python
from pop.models import ModelAdapter, register_provider

class MyAdapter(ModelAdapter):
    async def chat(self, messages, tools=None):
        ...  # return ModelResponse

    async def chat_stream(self, messages, tools=None):
        ...  # yield StreamChunk

register_provider("my_provider", MyAdapter)
Agent(model="my_provider:my-model")
```

## 10. Inspecting Results

Every run returns `AgentResult` with a full trace:

```python
result = agent.run("...")

# Walk the step trace
for step in result.steps:
    print(f"Step {step.index}: {step.tool_name or 'final answer'}")
    if step.tool_name:
        print(f"  Args: {step.tool_args}")
        print(f"  Result: {step.tool_result}")
    print(f"  Tokens: {step.token_usage.total}")
    print(f"  Cost: ${step.cost_usd:.6f}")

# Totals
print(f"Total cost: ${result.cost:.6f}")
print(f"Total tokens: {result.token_usage.total}")
print(f"Run ID: {result.run_id}")
```

## Quick Reference

```python
from pop import (
    # Core
    Agent, tool, Runner, run,
    # Multi-agent
    handoff, pipeline, orchestrate, debate, fan_out,
    # Workflows
    chain, route, parallel,
    # Models
    chat, model, register_provider,
    # Types
    AgentResult, Step, TokenUsage, Message,
    # Stream events
    ThinkEvent, ToolCallEvent, ToolResultEvent, TextDeltaEvent, DoneEvent,
)

from pop.memory import MarkdownMemory, InMemoryStore
from pop.hooks import ConsoleHook, CostHook, FileLogHook
```
