# LLM Providers

pop supports 8 LLM providers through a unified interface. All providers are included in the base install — no extras needed. Switching providers is a one-string change.

## Supported Providers

| Provider | Model string |
|----------|-------------|
| OpenAI | `openai:gpt-4o` |
| Anthropic | `anthropic:claude-sonnet-4-20250514` |
| Google Gemini | `gemini:gemini-2.0-flash` |
| DeepSeek | `deepseek:deepseek-chat` |
| Grok (xAI) | `grok:grok-3` |
| Kimi | `kimi:moonshot-v1-auto` |
| MiniMax | `minimax:MiniMax-Text-01` |
| GLM | `glm:glm-4-flash` |

## Switching Providers

```python
from pop import Agent

# Just change the model string
agent = Agent(model="openai:gpt-4o", tools=[search])
agent = Agent(model="anthropic:claude-sonnet-4-20250514", tools=[search])
agent = Agent(model="deepseek:deepseek-chat", tools=[search])
agent = Agent(model="grok:grok-3", tools=[search])
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
