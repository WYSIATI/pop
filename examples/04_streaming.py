"""Example: Streaming Events
Requires: OPENAI_API_KEY environment variable (or appropriate provider key)

Demonstrates streaming agent execution with event matching.
Events are yielded as they happen, letting you build real-time UIs,
progress indicators, or logging pipelines.
"""

import asyncio

from pop import (
    Agent,
    DoneEvent,
    Runner,
    TextDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
    tool,
)


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The city name to look up.
    """
    weather_data = {
        "San Francisco": "62F, foggy",
        "New York": "75F, sunny",
        "London": "58F, overcast",
    }
    return weather_data.get(city, f"No data for {city}")


@tool
def get_recommendation(temperature_f: int) -> str:
    """Get a clothing recommendation based on temperature.

    Args:
        temperature_f: Temperature in Fahrenheit.
    """
    if temperature_f < 50:
        return "Wear a warm coat and layers."
    if temperature_f < 65:
        return "A light jacket should be fine."
    return "T-shirt weather!"


async def main() -> None:
    agent = Agent(
        model="openai:gpt-4o-mini",
        tools=[get_weather, get_recommendation],
        instructions="Help users plan what to wear based on the weather.",
    )

    runner = Runner(agent)

    # Stream events as they happen
    async for event in runner.stream("What should I wear in San Francisco today?"):
        match event:
            case ToolCallEvent(name=name, args=args):
                print(f"[tool call] {name}({args})")

            case ToolResultEvent(name=name, output=output):
                print(f"[tool result] {name} -> {output}")

            case TextDeltaEvent(delta=text):
                print(f"[response] {text}")

            case DoneEvent(result=result):
                print(f"\n[done] Cost: ${result.cost:.6f}")


if __name__ == "__main__":
    asyncio.run(main())
