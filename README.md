<p align="center">
  <img src="assets/logo.svg" alt="pop" width="300">
</p>

<p align="center">
  <em>Fast, lean AI agents. 5 lines to production.</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/pop-framework/"><img src="https://img.shields.io/pypi/v/pop-framework?v=1.2.0" alt="PyPI"></a>
  <a href="https://github.com/WYSIATI/pop/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="License"></a>
</p>

---

> **Coding agents:** See **[SKILLS.md](SKILLS.md)** for the complete API guide — tools, agents, multi-agent patterns, streaming, memory, and all imports in one file.

---

**pop** is a lightweight Python framework for building AI agents. It supports multiple LLM providers, has 5 core concepts, and gets you from install to a working agent in under 2 minutes.

## Why pop?

- **5 lines to a working agent** -- define a tool, create an agent, call `run`.
- **8 LLM providers built-in** -- OpenAI, Anthropic, Gemini, DeepSeek, Grok, Kimi, MiniMax, GLM. Switch by changing one string.
- **~2,500 lines of code** -- read the entire framework in an afternoon.
- **2 runtime dependencies** -- `httpx` and `pydantic`. Import time under 1ms (lazy imports).
- **Zero commercial dependencies** -- no forced telemetry, no vendor lock-in.

## Install

```bash
uv add pop-framework
# or
pip install pop-framework
```

All 8 providers (OpenAI, Anthropic, Gemini, DeepSeek, Grok, Kimi, MiniMax, GLM) are included — no extras needed.

## Quick Start

```python
from pop import Agent, tool

@tool
def search(query: str) -> str:
    """Search the web for current information."""
    return web_search(query)  # your implementation

agent = Agent(model="openai:gpt-4o", tools=[search])
result = agent.run("What happened in AI today?")
print(result.output)
```

That's it. No `StateGraph`, no `RunnableSequence`, no `ChannelWrite`.

## Docs

| Guide | What it covers |
|-------|---------------|
| **[Skills](SKILLS.md)** | **Complete API guide for building agents** |
| [Providers](docs/providers.md) | Switching LLMs, failover, model adapters |
| [Streaming](docs/streaming.md) | Real-time events, pattern matching |
| [Workflows](docs/workflows.md) | Chain, route, parallel, agent, orchestration |
| [Multi-Agent](docs/multi-agent.md) | Handoff, pipeline, debate, fan_out |
| [Memory](docs/memory.md) | In-memory and markdown-based persistence |
| [Benchmarks](docs/benchmarks.md) | Performance numbers, framework comparison |

## Benchmarks

<p align="center">
  <img src="assets/bench-import-time.svg" alt="Import Time: pop vs smolagents vs LangChain" width="700">
</p>

<p align="center">
  <img src="assets/bench-overhead.svg" alt="Framework Overhead: pop vs LangChain" width="700">
</p>

<p align="center">
  <img src="assets/bench-deps.svg" alt="Dependencies: pop vs smolagents vs LangChain" width="700">
</p>

<p align="center">
  <img src="assets/bench-dx.svg" alt="Developer Experience: lines of code per task" width="700">
</p>

> Reproduce: `python benchmarks/bench_startup.py && python benchmarks/bench_dx.py && python benchmarks/generate_charts.py`

Details: [docs/benchmarks.md](docs/benchmarks.md)

## License

MIT
