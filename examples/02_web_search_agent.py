"""Example: Web Search Agent
Requires: OPENAI_API_KEY environment variable (or appropriate provider key)

An agent with a mock web search tool that demonstrates:
- @tool decorator with Google-style docstrings and type hints
- Agent instructions for controlling behavior
- Inspecting the full step trace after a run
"""

from pop import Agent, tool


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web and return relevant results.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return.
    """
    # In production, this would call a real search API.
    # The LLM sees the tool schema and calls it with proper arguments.
    return (
        f"Search results for '{query}' (top {max_results}):\n"
        "1. Python 3.12 Release Notes - python.org\n"
        "2. What's New in Python 3.12 - Real Python\n"
        "3. Python 3.12 Performance Improvements - blog.python.org"
    )


@tool
def read_url(url: str) -> str:
    """Fetch and return the text content of a URL.

    Args:
        url: The URL to fetch.
    """
    return f"[Content from {url}]: Python 3.12 introduces several performance improvements..."


agent = Agent(
    model="openai:gpt-4o",
    tools=[web_search, read_url],
    instructions=(
        "You are a research assistant. Search the web to answer questions. "
        "Always cite your sources with URLs."
    ),
    max_steps=5,
)

result = agent.run("What are the main performance improvements in Python 3.12?")

print(result.output)
print("\n--- Run details ---")
print(f"Steps taken: {len(result.steps)}")
print(f"Total tokens: {result.token_usage.total}")
print(f"Cost: ${result.cost:.6f}")

# Walk through the step trace
for step in result.steps:
    if step.tool_name:
        print(f"  Step {step.index}: called {step.tool_name}({step.tool_args})")
    else:
        print(f"  Step {step.index}: final answer")
