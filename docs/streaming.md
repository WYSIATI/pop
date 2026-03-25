# Streaming

pop supports real-time event streaming with structural pattern matching. Use it to build live UIs, progress indicators, or logging pipelines.

## Basic Streaming

```python
import asyncio
from pop import Agent, Runner, tool
from pop import ToolCallEvent, ToolResultEvent, TextDeltaEvent, DoneEvent

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"72F, sunny in {city}"

agent = Agent(model="openai:gpt-4o-mini", tools=[get_weather])
runner = Runner(agent)

async def main():
    async for event in runner.stream("What's the weather in NYC?"):
        match event:
            case ToolCallEvent(name=name, args=args):
                print(f"[tool call] {name}({args})")
            case ToolResultEvent(name=name, output=output):
                print(f"[tool result] {name} -> {output}")
            case TextDeltaEvent(delta=text):
                print(text, end="", flush=True)
            case DoneEvent(result=result):
                print(f"\n[done] Cost: ${result.cost:.6f}")

asyncio.run(main())
```

## Event Types

| Event | Fields | When it fires |
|-------|--------|---------------|
| `ToolCallEvent` | `name`, `args` | Agent decides to call a tool |
| `ToolResultEvent` | `name`, `output` | Tool returns a result |
| `TextDeltaEvent` | `delta` | LLM streams a text chunk |
| `DoneEvent` | `result` | Agent finishes (contains full `RunResult`) |

## Pattern Matching

Python 3.10+ structural pattern matching makes event handling clean and type-safe. Each event type can be destructured directly in the `match` statement.
