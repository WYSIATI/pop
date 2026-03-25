# Evaluation Strategy & Benchmark Plan

> Eval-driven design: define how we measure excellence FIRST, then build to meet those criteria.

---

## 1. Philosophy: Eval-First Design

Most agent frameworks ship first, benchmark later. We flip this:

1. **Define eval criteria** before writing framework code
2. **Build the eval harness** as the first deliverable
3. **Every PR must pass evals** — regressions are caught automatically
4. **Public, reproducible benchmarks** — anyone can run `make bench` and verify our claims

This means our eval harness is **not marketing** — it's the test suite.

---

## 2. The 8 Evaluation Dimensions

### Overview

| # | Dimension | What it measures | Who cares |
|---|-----------|-----------------|-----------|
| 1 | **Task Accuracy** | Does the agent complete the task correctly? | Everyone |
| 2 | **Cost Efficiency** | How many tokens/dollars per successful task? | Production users |
| 3 | **Latency** | How fast, end-to-end and framework overhead? | Production users |
| 4 | **Reliability** | How consistent across runs? Error recovery? | Production users |
| 5 | **Developer Experience** | Lines of code, time to build, debuggability | Developers (adoption) |
| 6 | **Tool Calling Accuracy** | Right tool, right args, right time? | Framework designers |
| 7 | **Multi-Agent Efficiency** | Coordination overhead, scaling behavior | Advanced users |
| 8 | **Resource Footprint** | Memory, import time, dependency count | DevOps, edge deploy |

---

## 3. Concrete Metrics & Targets

### Dimension 1: Task Accuracy

| Metric | Definition | Target | How we measure |
|--------|-----------|--------|---------------|
| Pass@1 | Success rate on first attempt | ≥ LangChain baseline | Standardized task suite, 100 tasks |
| Pass@3 | Success rate given 3 attempts | ≥95% on Level 1 tasks | Same suite, 3 runs per task |
| Partial credit | Fraction of subtasks completed | Higher than competitors | Multi-step task decomposition scoring |

**Task Suite Categories** (100 tasks total):

| Category | Count | Example | Ground truth type |
|----------|-------|---------|-------------------|
| Information Retrieval | 20 | "What is the market cap of Apple?" | Exact match (±5%) |
| Data Analysis | 15 | "Read sales.csv, what month had highest revenue?" | Exact match |
| Multi-step Reasoning | 20 | "Find 3 papers on X, compare their approaches" | LLM-as-judge (rubric) |
| API Orchestration | 15 | "Get weather in NYC, convert F→C, save to file" | Functional test |
| Code Generation | 15 | "Write a function that validates emails + tests" | Test suite pass |
| Conversational | 15 | "Help user book a flight with constraints" | Rubric + constraint check |

### Dimension 2: Cost Efficiency

| Metric | Definition | Target | How we measure |
|--------|-----------|--------|---------------|
| Tokens per task | Total input+output tokens | 20-30% fewer than LangChain | Instrument all LLM calls |
| LLM calls per task | Number of model invocations | 15-25% fewer than LangChain | Count calls |
| Cost per success | $ spent / success rate | Lowest among frameworks | Published pricing × tokens |
| Token overhead ratio | Framework tokens / minimal tokens | <1.1x (10% overhead max) | Compare vs bare-bones loop |

**Why we should win here:**
- LangChain injects verbose system prompts via its Runnable/callback machinery
- LangGraph's state serialization adds token overhead in checkpoints
- Our thin wrapper means the LLM sees almost exactly what the developer intended
- No hidden retry loops or redundant validation calls

### Dimension 3: Latency & Performance

| Metric | Definition | Target | How we measure |
|--------|-----------|--------|---------------|
| Import time | `python -c "import agynt"` | <50ms | `time` command, 10 runs |
| Agent creation | `Agent(model=..., tools=[...])` | <1ms | `timeit`, 10K iterations |
| Framework overhead | Total time minus LLM API time | <5ms per step | Mock LLM (instant response) |
| Time-to-first-token | Start → first streaming token | <50ms added over raw API | Streaming benchmark |
| Memory footprint | Peak RSS during agent run | <50MB for simple agent | `tracemalloc` |
| Concurrent throughput | Agents/sec with 100 parallel | >50 agents/sec | `asyncio` load test |

**How we isolate framework overhead:**
```
Real task time = LLM time + tool execution time + framework overhead
                                                   ^^^^^^^^^^^^^^^^
                                                   This is what we benchmark

Method: Replace LLM with MockModel (returns instantly).
        Replace tools with MockTools (return instantly).
        Remaining time = pure framework overhead.
```

### Dimension 4: Reliability

| Metric | Definition | Target | How we measure |
|--------|-----------|--------|---------------|
| Run-to-run variance | Std dev of success rate | <5% across 10 runs | Repeat each task 10× |
| Error recovery rate | % of injected failures recovered | >80% | Fault injection tests |
| Graceful degradation | Partial result when budget exceeded | Always return partial | Test with max_steps=1 |
| Infinite loop rate | % of runs hitting max iterations | <5% | Monitor step counts |
| Hallucination rate | Agent fabricates vs. calls tools | <2% | Compare actions vs. tool list |

**Fault injection scenarios:**
1. Tool returns error (HTTP 500) → agent should retry or use alternative
2. Tool times out → agent should skip and explain
3. Malformed LLM response → framework should recover gracefully
4. Rate limit hit → framework should backoff and retry
5. Missing API key → clear error message, not cryptic stack trace

### Dimension 5: Developer Experience (DX)

| Metric | Definition | Target | How we measure |
|--------|-----------|--------|---------------|
| Lines of code | For 10 reference tasks | 40-60% fewer than LangChain | Side-by-side code count |
| Time to hello world | Install → working agent | <2 minutes | Timed user test |
| Concepts to learn | Core abstractions count | 5 (Agent, Tool, Model, Memory, Runner) | Documentation audit |
| Stack trace depth | Frames in error trace | <5 framework frames | Intentional error test |
| Error message quality | Is the error actionable? | Specific fix suggestion | Rubric scoring |
| Type coverage | IDE autocomplete works? | 100% typed | mypy strict |
| Boilerplate ratio | Framework code / business logic | <0.3 (30% boilerplate) | Code analysis |

**The 10 Reference Tasks for DX benchmarking:**

| # | Task | Complexity |
|---|------|-----------|
| 1 | Hello world agent (one tool) | Trivial |
| 2 | Agent with 3 tools | Simple |
| 3 | Streaming agent | Simple |
| 4 | Structured output (Pydantic) | Medium |
| 5 | Agent with memory | Medium |
| 6 | Multi-provider (swap models) | Medium |
| 7 | Human-in-the-loop | Medium |
| 8 | Two-agent handoff | Complex |
| 9 | Orchestrator + 3 workers | Complex |
| 10 | Full research agent (search + read + write) | Complex |

For each: implement in our framework AND in LangChain/LangGraph, count lines, measure time.

### Dimension 6: Tool Calling Accuracy

| Metric | Definition | Target | How we measure |
|--------|-----------|--------|---------------|
| Tool selection accuracy | Correct tool chosen | ≥ baseline (model-dependent) | Known-optimal task suite |
| Argument accuracy | Correct params passed | ≥ baseline | Schema validation |
| Unnecessary call rate | Redundant tool calls | <5% | Compare vs optimal sequence |
| Schema generation quality | Auto-generated JSON Schema | 100% valid | Schema validator |

**Why framework matters here:**
- Tool description quality affects model's tool selection
- Our `@tool` decorator auto-generates descriptions from docstrings + type hints
- Better schema → better tool calls → fewer wasted LLM rounds

### Dimension 7: Multi-Agent Efficiency

| Metric | Definition | Target | How we measure |
|--------|-----------|--------|---------------|
| Overhead vs single agent | Extra cost for multi-agent | <30% overhead | Compare 1-agent vs N-agent |
| Message efficiency | Inter-agent messages sent | Minimal (no chatter) | Count messages |
| Scaling behavior | Perf change with N agents | Sub-linear degradation | 1, 2, 4, 8 agents |
| Deadlock rate | Agents stuck waiting | 0% | Timeout detection |

### Dimension 8: Resource Footprint

| Metric | Definition | Target | How we measure |
|--------|-----------|--------|---------------|
| Package size | Installed size on disk | <5MB | `du -sh` after install |
| Dependency count | Transitive deps | <5 | `pip show` + tree |
| Import time | Cold import | <50ms | `python -X importtime` |
| Min Python version | Oldest supported | 3.10+ | CI matrix |

---

## 4. Eval Harness Architecture

```
eval/
├── README.md                    # How to run benchmarks
├── Makefile                     # make bench, make bench-quick, make bench-full
│
├── tasks/                       # Task definitions (YAML)
│   ├── schema.json              # Task definition schema
│   ├── information_retrieval/
│   │   ├── task_001.yaml        # { prompt, tools_available, ground_truth, scoring }
│   │   ├── task_002.yaml
│   │   └── ...
│   ├── data_analysis/
│   ├── multi_step_reasoning/
│   ├── api_orchestration/
│   ├── code_generation/
│   └── conversational/
│
├── adapters/                    # Framework wrappers (common interface)
│   ├── base.py                  # Abstract FrameworkAdapter
│   ├── ours.py                  # Our framework adapter
│   ├── langchain_adapter.py     # LangChain/LangGraph adapter
│   ├── crewai_adapter.py        # CrewAI adapter
│   ├── smolagents_adapter.py    # smolagents adapter
│   ├── pydantic_ai_adapter.py   # Pydantic-AI adapter
│   └── bare_adapter.py          # Bare-bones (raw API) baseline
│
├── tools/                       # Shared tool implementations
│   ├── web_search.py            # Mock + real web search
│   ├── file_ops.py              # Read/write files
│   ├── calculator.py            # Math operations
│   ├── code_executor.py         # Sandboxed code execution
│   └── mock_tools.py            # Instant-return mocks for overhead testing
│
├── scorers/                     # Scoring functions
│   ├── exact_match.py           # Deterministic comparison
│   ├── fuzzy_match.py           # Approximate string matching
│   ├── llm_judge.py             # GPT-4 as judge with rubric
│   ├── code_test.py             # Run test suite on generated code
│   └── composite.py             # Weighted multi-metric scorer
│
├── metrics/                     # Metric collectors
│   ├── accuracy.py              # Task success metrics
│   ├── cost.py                  # Token counting + dollar cost
│   ├── latency.py               # Timing + overhead isolation
│   ├── reliability.py           # Variance + fault injection
│   ├── dx.py                    # Lines of code + complexity
│   └── resources.py             # Memory + import time
│
├── runners/
│   ├── run_benchmark.py         # Main entry: runs all tasks × all frameworks
│   ├── run_dx_benchmark.py      # Developer experience comparison
│   ├── run_overhead_benchmark.py # Framework overhead (mocked LLM)
│   └── run_fault_injection.py   # Reliability testing
│
├── analysis/
│   ├── generate_report.py       # Markdown report with tables
│   ├── generate_charts.py       # Matplotlib/Plotly comparison charts
│   └── templates/
│       └── report.md.jinja      # Report template
│
├── results/                     # Generated outputs (gitignored)
│   ├── raw/                     # Per-task JSON results
│   ├── reports/                 # Generated markdown reports
│   └── charts/                  # Generated comparison charts
│
└── dx_comparison/               # Side-by-side code for DX benchmarking
    ├── task_01_hello_world/
    │   ├── ours.py
    │   ├── langchain.py
    │   ├── crewai.py
    │   └── comparison.md
    ├── task_02_three_tools/
    │   ├── ...
    └── task_10_research_agent/
        ├── ...
```

### Common Adapter Interface

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class TaskResult:
    answer: str                     # Agent's final output
    success: bool | None            # None if needs scoring
    total_tokens: int               # Input + output tokens
    llm_calls: int                  # Number of LLM invocations
    wall_clock_ms: float            # Total time
    framework_overhead_ms: float    # Time minus LLM + tool time
    peak_memory_bytes: int          # Peak RSS
    steps: list[dict]               # Full trajectory
    error: str | None               # If agent failed

class FrameworkAdapter(ABC):
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def version(self) -> str: ...

    @abstractmethod
    def create_agent(
        self,
        model: str,
        tools: list[callable],
        system_prompt: str,
    ) -> object: ...

    @abstractmethod
    def run(self, agent: object, task: str) -> TaskResult: ...
```

### Task Definition Format

```yaml
# eval/tasks/information_retrieval/task_001.yaml
id: ir_001
category: information_retrieval
difficulty: easy
description: "Simple factual lookup requiring one search"

prompt: "What is the current population of Tokyo?"
tools_required: [web_search]

ground_truth:
  type: fuzzy_match
  answer: "14 million"
  tolerance: 0.1  # ±10%

scoring:
  - metric: exact_match
    weight: 0.7
  - metric: llm_judge
    weight: 0.3
    rubric: |
      Does the answer correctly state Tokyo's population?
      Is a source cited or implied?

metadata:
  optimal_tool_calls: 1
  optimal_tokens_estimate: 500
```

---

## 5. The Headline Benchmarks (for README + launch)

These are the 7 numbers we'll put front-and-center:

### Benchmark Table (Target)

```
┌─────────────────────────┬──────────┬───────────────┬─────────┐
│ Metric                  │ [ours]   │ LangChain+LG  │ vs.     │
├─────────────────────────┼──────────┼───────────────┼─────────┤
│ Task success rate       │ ~85%     │ ~83%          │ ≥ parity│
│ Cost per success ($)    │ $0.012   │ $0.018        │ 33% ↓   │
│ LLM calls per task      │ 3.2      │ 4.1           │ 22% ↓   │
│ Framework overhead      │ <5ms     │ ~45ms         │ 9x ↓    │
│ Import time             │ ~40ms    │ ~1,200ms      │ 30x ↓   │
│ Lines of code (avg)     │ 12       │ 35            │ 66% ↓   │
│ Dependencies            │ 3        │ 20+           │ 85% ↓   │
└─────────────────────────┴──────────┴───────────────┴─────────┘
```

**Key insight:** We will likely match or slightly beat LangChain on task accuracy (same model underneath), but **dominate on cost, speed, and DX**. That's the story:

> "Same accuracy, 33% cheaper, 30x faster to start, 66% less code."

### The Charts We'll Generate

1. **Bar chart: Framework overhead** (mocked LLM) — us vs LangChain vs CrewAI vs bare-bones
2. **Bar chart: Cost per successful task** — stacked (input tokens + output tokens)
3. **Bar chart: Lines of code** — 10 reference tasks side by side
4. **Scatter plot: Accuracy vs Cost** — each framework as a dot (we want bottom-right)
5. **Table: Import time + memory** — cold numbers
6. **Radar chart: All 8 dimensions** — normalized 0-1 scale

---

## 6. How Evals Drive the Architecture

Every architectural decision must justify itself against eval metrics:

| Architecture Decision | Eval Metric It Optimizes |
|----------------------|--------------------------|
| Thin provider wrappers (no Runnable) | Framework overhead <5ms |
| @tool auto-schema from type hints | Tool calling accuracy (better descriptions) |
| No hidden system prompt injection | Cost efficiency (fewer tokens) |
| Built-in Reflexion loop | Error recovery rate >80% |
| Async-first with sync wrapper | Concurrent throughput >50/sec |
| Immutable state snapshots | Reliability (deterministic replay) |
| Single package, 3 deps | Import time <50ms, package size <5MB |
| Functions, not classes | DX: 66% fewer lines of code |
| Model fallback chain | Reliability (graceful degradation) |
| Step budget (max_steps) | Prevent infinite loops (<5% rate) |

---

## 7. Eval-Driven Development Workflow

```
For every PR:

1. Unit tests pass                     (pytest)
2. Type check passes                   (mypy --strict)
3. Lint passes                         (ruff)
4. Quick eval passes                   (make bench-quick)
   - 10 tasks, mocked LLM
   - Checks: framework overhead, import time, memory
   - Must not regress by >5%

For every release:

5. Full eval suite                     (make bench-full)
   - 100 tasks, real LLM (GPT-4o)
   - All 8 dimensions measured
   - Comparison against LangChain + 3 others
   - Results published in release notes

Monthly:

6. Cross-framework benchmark update    (make bench-compare)
   - Re-run against latest versions of all competitors
   - Update README numbers if improved
   - Blog post with analysis
```

---

## 8. Eval Timeline

| Phase | When | What |
|-------|------|------|
| **Phase 0** | Before v0.1 | Build eval harness skeleton + 20 tasks + bare-bones adapter |
| **Phase 1** | v0.1 launch | 50 tasks, our adapter + LangChain adapter. Quick eval in CI. |
| **Phase 2** | v0.2 | 100 tasks, add CrewAI + smolagents adapters. Full eval suite. |
| **Phase 3** | v0.3 | DX benchmark (10 reference tasks, side-by-side code). Publish results. |
| **Phase 4** | v0.4 | Fault injection tests. Multi-agent eval. Reliability metrics. |
| **Phase 5** | v0.5 | Public benchmark site. Automated monthly comparisons. |

---

## 9. What Makes Our Evals Credible

1. **Open-source harness** — anyone can run `make bench` and verify
2. **Adapter-based** — same tasks, same tools, same model across frameworks
3. **Statistical rigor** — mean ± std dev, 5+ runs per task, significance tests
4. **Honest reporting** — we show where we lose, not just where we win
5. **Reproducible** — pinned framework versions, seeded randomness, Docker option
6. **Not just microbenchmarks** — real end-to-end tasks that developers care about
7. **Community-contributed tasks** — task suite grows over time via PRs

---

## 10. Competitive Advantages We Must Prove

| Claim | Eval that proves it | Compelling if... |
|-------|-------------------|-----------------|
| "Faster" | Import time, framework overhead | 10x+ difference |
| "Cheaper" | Cost per successful task | 20%+ cheaper |
| "Simpler" | Lines of code comparison | 40%+ fewer lines |
| "Same accuracy" | Task success rate | Within ±3% |
| "More reliable" | Error recovery, run variance | 80%+ recovery |
| "Lighter" | Dependencies, memory, package size | 5x+ fewer deps |
| "Better errors" | Stack trace depth, message quality | Concrete examples |

The story we tell:

> **"Equal accuracy. One-third the cost. One-thirtieth the startup time. One-third the code. Zero commercial dependencies."**

This is provable, reproducible, and compelling. It's not marketing — it's measurement.
