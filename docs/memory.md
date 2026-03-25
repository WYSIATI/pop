# Memory

pop ships two memory backends: in-memory (default, zero-config) and markdown-based (persistent, human-readable files on disk).

## Markdown Memory

Persistent memory stored as plain markdown files. Human-readable and version-control friendly.

```python
from pop import Agent, tool
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

## Directory Structure

Memory persists across sessions as plain markdown files:

```
~/.pop/memory/          (default location)
    core/               -- always in context
    conversations/      -- session transcripts
    episodes/           -- past experiences
    knowledge/          -- domain knowledge
```

## Configuration

| Method | Priority | Example |
|--------|----------|---------|
| `base_dir` argument | Highest | `MarkdownMemory(base_dir="./my_memory")` |
| `POP_MEMORY_DIR` env var | Medium | `export POP_MEMORY_DIR=/data/agent_memory` |
| Default | Lowest | `~/.pop/memory` |

## In-Memory (Default)

If you don't pass a `memory` parameter, the agent uses an in-memory backend. State is lost when the process exits. This is the right choice for stateless tasks or when you manage conversation history yourself.

## Core Memory

The `core_memory` parameter accepts a `dict[str, str]` of key-value pairs that are always included in the agent's context. Use it for user preferences, identity, or configuration that should persist across turns.
