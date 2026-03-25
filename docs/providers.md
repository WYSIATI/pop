# LLM Providers

pop supports multiple LLM providers through a unified interface. Switching providers is a one-string change -- the rest of your code stays identical.

## Supported Providers

| Provider | Model string | Install extra |
|----------|-------------|---------------|
| OpenAI | `openai:gpt-4o` | `pop-framework[openai]` |
| Anthropic | `anthropic:claude-sonnet-4-20250514` | `pop-framework[anthropic]` |
| Google Gemini | `gemini:gemini-2.0-flash` | `pop-framework[gemini]` |
| DeepSeek | `deepseek:deepseek-chat` | *(uses OpenAI-compatible API)* |
| Kimi | `kimi:moonshot-v1-auto` | *(uses OpenAI-compatible API)* |

## Switching Providers

```python
from pop import Agent

# Just change the model string
agent = Agent(model="openai:gpt-4o", tools=[search])
agent = Agent(model="anthropic:claude-sonnet-4-20250514", tools=[search])
agent = Agent(model="deepseek:deepseek-chat", tools=[search])
agent = Agent(model="gemini:gemini-2.0-flash", tools=[search])
agent = Agent(model="kimi:moonshot-v1-auto", tools=[search])
```

## Automatic Failover

Pass a list of models. If the first model hits a rate limit or outage, pop tries the next one.

```python
agent = Agent(
    model=[
        "anthropic:claude-sonnet-4-20250514",
        "openai:gpt-4o",
        "openai:gpt-4o-mini",
    ],
    tools=[search],
)
```

## Installing Provider Extras

```bash
# Single provider
uv add "pop-framework[openai]"
uv add "pop-framework[anthropic]"
uv add "pop-framework[gemini]"

# All providers
uv add "pop-framework[all]"

# Or with pip
pip install "pop-framework[openai]"
```

## Using the Low-Level Model Adapter

For direct model access without the agent loop:

```python
from pop import chat, model

# One-shot call
response = chat("openai:gpt-4o-mini", "Explain quantum computing in one paragraph.")
print(response.content)

# Reusable adapter
adapter = model("openai:gpt-4o-mini")
```
