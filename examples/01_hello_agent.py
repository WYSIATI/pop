"""Example: Hello Agent
Requires: OPENAI_API_KEY environment variable (or appropriate provider key)

The simplest possible pop agent -- define a tool, create an agent, run a task.
This demonstrates the core API in 5 lines of actual code.
"""

from pop import Agent, tool


# 1. Define a tool with the @tool decorator
@tool
def add(a: float, b: float) -> float:
    """Add two numbers together.

    Args:
        a: The first number.
        b: The second number.
    """
    return a + b


# 2. Create an agent with a model and tools
agent = Agent(
    model="openai:gpt-4o-mini",
    tools=[add],
)

# 3. Run a task and get the result
result = agent.run("What is 42 + 17?")

# 4. Print the output
print(result.output)

# 5. Inspect cost and steps
print(f"Cost: ${result.cost:.6f}")
print(f"Steps: {len(result.steps)}")
