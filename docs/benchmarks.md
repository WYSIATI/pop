# Benchmarks

All benchmarks are measured, reproducible, and run from real processes. No fabricated numbers.

## Summary

| Metric | pop | LangChain + LangGraph | Delta |
|--------|-----|----------------------|-------|
| Framework overhead per step | ~0.15ms | ~45ms | ~300x faster |
| Import time | ~0.17ms | ~1,200ms | ~7,000x faster |
| Lines of code (avg task) | 12 | 42 | 71% less code |
| Dependencies | 2 | 20+ | 90% fewer |
| Core source | ~2,600 lines | ~188,000 lines | 1/72nd the size |

> Task accuracy and cost are determined by the LLM model, not the framework.
> Both frameworks call the same model — accuracy is identical.

## Running Benchmarks

```bash
# Developer experience comparison (lines of code)
python benchmarks/bench_dx.py

# Startup performance (import time, agent creation, framework overhead)
python benchmarks/bench_startup.py
```

Results are written to `benchmarks/results/latest.json` and a human-readable summary to
`benchmarks/results/latest_report.md`.

## pop vs LangChain + LangGraph

| Capability | pop | LangChain + LangGraph |
|-----------|-----|----------------------|
| Agent loop | `Agent.run()` | StateGraph + nodes + edges + compile |
| Tool definition | `@tool` decorator | `BaseTool` subclass or `@tool` + schema |
| Streaming | `async for event in runner.stream()` | Callbacks + `astream_events` |
| Multi-agent | `handoff()`, `pipeline()`, `debate()` | LangGraph subgraphs + channels |
| Provider switching | Change one string | Swap class imports + init params |
| Memory | Built-in, 2 backends | LangChain Memory / LangGraph checkpointer |
| Error recovery | Built-in Reflexion loop | Manual retry logic |
| Model fallback | Pass a list of models | Custom `with_fallbacks` chain |
| Dependencies | 2 | 20+ |
| Learning curve | 5 concepts | StateGraph, Runnable, Chain, Agent, Tool, Memory, Callback, Channel, ... |
| Commercial deps | None | LangSmith (telemetry opt-out required) |
| License | MIT | MIT |
