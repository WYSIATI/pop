# Benchmarks

All benchmarks are measured, reproducible, and run from real processes. No fabricated numbers.

## Summary

| Metric | pop | smolagents | LangChain + LangGraph |
|--------|-----|------------|----------------------|
| Import time | ~0.28ms | ~740ms | ~1,200ms |
| Framework overhead per step | ~0.14ms | ~2.8ms | ~45ms |
| Runtime dependencies | 2 | 6 | 20+ |
| Lines of code (avg task) | 6 | 9 | 19 |

> Task accuracy and cost are determined by the LLM model, not the framework.
> All frameworks call the same model — accuracy is identical.

## Running Benchmarks

```bash
# Startup performance (import time, agent creation, framework overhead)
python benchmarks/bench_startup.py

# Developer experience comparison (lines of code)
python benchmarks/bench_dx.py

# Generate SVG charts
python benchmarks/generate_charts.py
```

Results are written to `benchmarks/results/latest.json` and a human-readable summary to
`benchmarks/results/latest_report.md`. Charts are written to `assets/`.

## Developer Experience: Lines of Code

Hand-counted from official docs. Docstrings excluded.

| Task | pop | smolagents | LangChain |
|------|-----|------------|-----------|
| Hello World (agent + 1 custom tool) | 10 | 11 | 11 |
| Web Search (built-in tool) | 4 | 4 | 12 |
| Multi-Agent Handoff | 5 | 12 | 35 |

pop uses `pop.tools.WebSearch()` (built-in) and `Agent(workers=[...])` shorthand.

## pop vs LangChain + LangGraph

| Capability | pop | LangChain + LangGraph |
|-----------|-----|----------------------|
| Agent loop | `Agent.run()` | StateGraph + nodes + edges + compile |
| Tool definition | `@tool` decorator | `BaseTool` subclass or `@tool` + schema |
| Built-in tools | `WebSearch()`, `ReadURL()`, `Calculator()` | TavilySearch, WikipediaQuery, ... |
| Streaming | `async for event in runner.stream()` | Callbacks + `astream_events` |
| Multi-agent | `handoff()`, `pipeline()`, `debate()`, `workers=` | LangGraph subgraphs + channels |
| Provider switching | Change one string | Swap class imports + init params |
| Memory | Built-in, 2 backends | LangChain Memory / LangGraph checkpointer |
| Error recovery | Built-in Reflexion loop | Manual retry logic |
| Model fallback | Pass a list of models | Custom `with_fallbacks` chain |
| Dependencies | 2 | 20+ |
| Learning curve | 5 concepts | StateGraph, Runnable, Chain, Agent, Tool, Memory, Callback, Channel, ... |
| Commercial deps | None | LangSmith (telemetry opt-out required) |
| License | MIT | MIT |
