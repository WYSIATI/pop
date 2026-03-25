# pop Framework — Benchmark Report

> Run `uv run python benchmarks/bench_startup.py` and `uv run python benchmarks/bench_dx.py` to reproduce.

---

## The Headline

> **Millisecond startup. Near-zero overhead. One-third the code.**

pop is designed to get out of the way. The framework adds almost no overhead —
you pay for the LLM, not the scaffolding.

---

## Import Time

| | pop | LangChain | Speedup |
|---|---|---|---|
| Cold import | 0.24ms | ~1200ms | ~5,088x faster |

Measured: mean **0.24ms**, min 0.20ms, max 0.47ms \
(10 cold-process runs)

**Why it's fast:** pop uses lazy imports. `import pop` loads only the package index (~50 lines).
All framework code (Agent, tools, providers) loads on first use. LangChain eagerly imports its
entire dependency tree at import time (~1,200ms on a cold interpreter).

---

## Framework Overhead Per Step

| | pop | LangChain | Speedup |
|---|---|---|---|
| Per-step overhead | 0.12ms | ~45ms | ~374x faster |

Measured: mean 0.13ms, median **0.12ms** \
(100 iterations, instant mock LLM)

**Method:** The LLM is replaced with a mock that returns instantly. Remaining time = pure
framework overhead: message assembly, hook dispatch, state updates. LangChain/LangGraph's
Runnable and callback machinery adds ~45ms per step.

---

## Agent Creation

Instantiating `Agent(model=..., tools=[])`: mean 0.5µs, \
median 0.5µs (1,000 runs).

Creating an agent is effectively free — no network calls, no schema validation on init.

---

## Developer Experience: Lines of Code

Hand-counted from working implementations of the same task in each framework.

| Task | pop | LangChain | Reduction |
|------|-----|-----------|-----------|
| Hello World Agent | 5 | 35 | 85% less |
| Web Search Agent | 12 | 45 | 73% less |
| Structured Output | 15 | 40 | 62% less |
| Streaming | 15 | 50 | 70% less |
| Multi Provider | 20 | 60 | 66% less |
| Tool Definition | 6 | 15 | 60% less |
| Multi Agent Handoff | 15 | 55 | 72% less |
| Agent With Memory | 10 | 35 | 71% less |

pop requires **71% fewer lines** on average (12 vs 41 lines).

---

## Package Footprint

| | pop | LangChain |
|---|---|---|
| Core source | 2,683 lines | ~188,000 lines |
| Runtime deps | 2 | 20+ |
| Dep names | httpx>=0.27, pydantic>=2.0 | langchain, langchain-core, langsmith, openai, tiktoken, ... |

---

## Reproducing These Results

```bash
uv add pop-framework
uv run python benchmarks/bench_startup.py
uv run python benchmarks/bench_dx.py
```

Raw data: `benchmarks/results/latest.json`, `benchmarks/results/dx_comparison.json`
