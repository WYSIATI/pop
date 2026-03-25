<p align="center">
  <img src="assets/logo.svg" alt="pop" width="300">
</p>

<p align="center">
  <em>Fast, lean AI agents. 5 lines to production.</em>
</p>

<p align="center">
  <a href="https://github.com/WYSIATI/pop/actions"><img src="https://img.shields.io/github/actions/workflow/status/WYSIATI/pop/ci.yml?label=CI" alt="CI"></a>
  <a href="https://github.com/WYSIATI/pop"><img src="https://img.shields.io/badge/coverage-99%25-brightgreen" alt="Coverage"></a>
  <a href="https://pypi.org/project/pop-framework/"><img src="https://img.shields.io/pypi/v/pop-framework" alt="PyPI"></a>
  <a href="https://pypi.org/project/pop-framework/"><img src="https://img.shields.io/pypi/pyversions/pop-framework" alt="Python"></a>
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="License">
</p>

<p align="center">
  <a href="docs/">Documentation</a> |
  <a href="https://github.com/WYSIATI/pop">Source Code</a> |
  <a href="https://discord.gg/pop">Discord</a>
</p>

---

**pop** is a lightweight Python framework for building AI agents. It supports multiple LLM providers, has 5 core concepts, and gets you from install to a working agent in under 2 minutes.

## Why pop?

- **5 lines to a working agent** -- define a tool, create an agent, call `run`.
- **7 LLM providers built-in** -- OpenAI, Anthropic, Gemini, DeepSeek, Kimi, MiniMax, GLM. Switch by changing one string.
- **~2,500 lines of code** -- read the entire framework in an afternoon.
- **2 runtime dependencies** -- `httpx` and `pydantic`. Import time under 1ms (lazy imports).
- **Zero commercial dependencies** -- no forced telemetry, no vendor lock-in.

## Install

```bash
# Recommended
uv add pop-framework

# With a provider extra
uv add "pop-framework[openai]"
uv add "pop-framework[anthropic]"
uv add "pop-framework[all]"
```

Or with pip:

```bash
pip install pop-framework
pip install "pop-framework[openai]"
```

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
| [Providers](docs/providers.md) | Switching LLMs, failover, model adapters |
| [Streaming](docs/streaming.md) | Real-time events, pattern matching |
| [Workflows](docs/workflows.md) | Chain, route, parallel, agent, orchestration |
| [Multi-Agent](docs/multi-agent.md) | Handoff, pipeline, debate, fan_out |
| [Memory](docs/memory.md) | In-memory and markdown-based persistence |
| [Benchmarks](docs/benchmarks.md) | Performance numbers, framework comparison |

## Benchmarks

| Metric | pop | LangChain + LangGraph | Delta |
|--------|-----|----------------------|-------|
| Task success rate | ~85% | ~83% | parity |
| Cost per successful task | $0.012 | $0.018 | 33% lower |
| Framework overhead | ~0.15ms | ~45ms | ~300x faster |
| Import time | ~0.17ms | ~1,200ms | ~7,000x faster |
| Lines of code (avg task) | ~12 | ~42 | 71% less |
| Dependencies | 2 | 20+ | 90% fewer |

Methodology: [EVAL_STRATEGY.md](EVAL_STRATEGY.md) | Details: [docs/benchmarks.md](docs/benchmarks.md)

## Community

- [Discord](https://discord.gg/pop) -- questions, help, showcase
- [GitHub Issues](https://github.com/WYSIATI/pop/issues) -- bug reports
- [GitHub Discussions](https://github.com/WYSIATI/pop/discussions) -- feature ideas, Q&A

## Contributing

```bash
git clone https://github.com/WYSIATI/pop.git
cd pop
uv sync --group dev
uv run pytest
```

## License

MIT
