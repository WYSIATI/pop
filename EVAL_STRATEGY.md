# Evaluation Strategy & Benchmark Plan

> Eval-driven design: define how we measure excellence FIRST, then build to meet those criteria.

---

## Current Status: What's Shipped vs. Planned

| Phase | Status | What |
|-------|--------|------|
| **Phase 0 (current)** | ✅ Shipped | Startup + DX benchmarks. Human-readable reports. |
| Phase 1 | 🗓 Planned | 50-task eval suite + LangChain adapter. Quick eval in CI. |
| Phase 2 | 🗓 Planned | 100 tasks, add CrewAI + smolagents adapters. Full eval suite. |
| Phase 3 | 🗓 Planned | Fault injection + reliability metrics. |
| Phase 4 | 🗓 Planned | Public benchmark site. Automated monthly comparisons. |

**What exists today** (`benchmarks/`):

```
benchmarks/
├── bench_startup.py          # Import time, agent creation, per-step overhead
├── bench_dx.py               # Lines-of-code comparison (pop vs LangChain, 8 tasks)
└── results/
    ├── latest.json           # Latest startup benchmark data
    ├── dx_comparison.json    # Latest DX benchmark data
    └── latest_report.md      # Auto-generated human-readable summary
```

Run both benchmarks:
```bash
python benchmarks/bench_startup.py
python benchmarks/bench_dx.py
```

**What is planned but not yet built:** task accuracy suite, cost efficiency measurement,
reliability/fault-injection tests, multi-agent efficiency benchmarks, and the full
eval harness architecture described in Section 4.

---

## 1. Philosophy: Eval-First Design

Most agent frameworks ship first, benchmark later. We flip this:

1. **Define eval criteria** before writing framework code
2. **Build the eval harness** as the first deliverable
3. **Every PR must pass evals** — regressions are caught automatically
4. **Public, reproducible benchmarks** — anyone can run `python benchmarks/bench_startup.py` and verify

This means our eval harness is **not marketing** — it's the test suite.

---

## 2. The 8 Evaluation Dimensions

| # | Dimension | What it measures | Who cares | Status |
|---|-----------|-----------------|-----------|--------|
| 1 | **Task Accuracy** | Does the agent complete the task correctly? | Everyone | 🗓 Planned |
| 2 | **Cost Efficiency** | How many tokens/dollars per successful task? | Production users | 🗓 Planned |
| 3 | **Latency** | Framework overhead and end-to-end speed | Production users | ✅ Partially shipped |
| 4 | **Reliability** | Consistency across runs, error recovery | Production users | 🗓 Planned |
| 5 | **Developer Experience** | Lines of code, time to build, debuggability | Developers | ✅ Partially shipped |
| 6 | **Tool Calling Accuracy** | Right tool, right args, right time | Framework designers | 🗓 Planned |
| 7 | **Multi-Agent Efficiency** | Coordination overhead, scaling behavior | Advanced users | 🗓 Planned |
| 8 | **Resource Footprint** | Memory, import time, dependency count | DevOps, edge deploy | ✅ Partially shipped |

---

## 3. Concrete Metrics & Targets

### Dimension 1: Task Accuracy (Planned)

| Metric | Definition | Target | How we measure |
|--------|-----------|--------|---------------|
| Pass@1 | Success rate on first attempt | >= LangChain baseline | Standardized task suite, 100 tasks |
| Pass@3 | Success rate given 3 attempts | >=95% on Level 1 tasks | Same suite, 3 runs per task |
| Partial credit | Fraction of subtasks completed | Higher than competitors | Multi-step task decomposition scoring |

**Task Suite Categories** (100 tasks total, planned):

| Category | Count | Example | Ground truth type |
|----------|-------|---------|-------------------|
| Information Retrieval | 20 | "What is the market cap of Apple?" | Exact match (+-5%) |
| Data Analysis | 15 | "Read sales.csv, what month had highest revenue?" | Exact match |
| Multi-step Reasoning | 20 | "Find 3 papers on X, compare their approaches" | LLM-as-judge (rubric) |
| API Orchestration | 15 | "Get weather in NYC, convert F to C, save to file" | Functional test |
| Code Generation | 15 | "Write a function that validates emails + tests" | Test suite pass |
| Conversational | 15 | "Help user book a flight with constraints" | Rubric + constraint check |

### Dimension 2: Cost Efficiency (Planned)

| Metric | Definition | Target | How we measure |
|--------|-----------|--------|---------------|
| Tokens per task | Total input+output tokens | 20-30% fewer than LangChain | Instrument all LLM calls |
| LLM calls per task | Number of model invocations | 15-25% fewer than LangChain | Count calls |
| Cost per success | $ spent / success rate | Lowest among frameworks | Published pricing x tokens |

**Why we should win:** LangChain injects verbose system prompts via Runnable/callback machinery.
LangGraph's state serialization adds checkpoint token overhead. Our thin wrapper sends almost
exactly what the developer intended — no hidden injections.

### Dimension 3: Latency & Performance (Partially Shipped)

| Metric | Definition | Target | Status | Current result |
|--------|-----------|--------|--------|----------------|
| Import time | python -c "import pop" | <50ms | Measured | ~0.16ms (lazy imports) |
| Agent creation | Agent(model=..., tools=[...]) | <1ms | Measured | ~0.001ms |
| Framework overhead | Time minus LLM + tool time | <5ms per step | Measured | ~0.15ms |
| Time-to-first-token | Start to first streaming token | <50ms added | Planned | — |
| Memory footprint | Peak RSS during agent run | <50MB | Planned | — |
| Concurrent throughput | Agents/sec with 100 parallel | >50 agents/sec | Planned | — |

**Isolation method:**
```
Real task time = LLM time + tool time + framework overhead
                                         ^^^^^^^^^^^^^^^^
                                         This is what we measure

Method: Replace LLM with MockModel (returns instantly).
        Replace tools with MockTools (return instantly).
        Remaining time = pure framework overhead.
```

### Dimension 4: Reliability (Planned)

| Metric | Definition | Target | How we measure |
|--------|-----------|--------|---------------|
| Run-to-run variance | Std dev of success rate | <5% across 10 runs | Repeat each task 10x |
| Error recovery rate | % of injected failures recovered | >80% | Fault injection tests |
| Graceful degradation | Partial result when budget exceeded | Always | Test with max_steps=1 |
| Infinite loop rate | % of runs hitting max iterations | <5% | Monitor step counts |

**Fault injection scenarios (planned):** tool HTTP 500, tool timeout, malformed LLM response,
rate limit hit, missing API key.

### Dimension 5: Developer Experience (Partially Shipped)

| Metric | Definition | Target | Status | Current result |
|--------|-----------|--------|--------|----------------|
| Lines of code | For 8 reference tasks | 40-60% fewer than LangChain | Measured | **71% fewer** |
| Time to hello world | Install to working agent | <2 minutes | Planned | — |
| Concepts to learn | Core abstractions count | 5 | By design | 5 (Agent, Tool, Model, Memory, Runner) |
| Type coverage | IDE autocomplete works? | 100% typed | mypy strict | 100% |
| Stack trace depth | Frames in error trace | <5 framework frames | Planned | — |

**The 8 measured DX tasks:**

| Task | pop | LangChain | Reduction |
|------|-----|-----------|-----------|
| Hello world agent | 5 | 35 | 86% |
| Web search agent | 12 | 45 | 73% |
| Streaming agent | 15 | 50 | 70% |
| Structured output (Pydantic) | 15 | 40 | 62% |
| Agent with memory | 10 | 35 | 71% |
| Multi-provider (swap models) | 20 | 60 | 67% |
| Tool definition | 6 | 15 | 60% |
| Two-agent handoff | 15 | 55 | 73% |
| **Average** | **~12** | **~42** | **71%** |

### Dimensions 6-8 (Planned)

**Tool Calling Accuracy:** tool selection accuracy, argument accuracy, schema generation quality.

**Multi-Agent Efficiency:** overhead vs single agent, message efficiency, scaling behavior.

**Resource Footprint:** package size, import time (measured: ~0.16ms), memory footprint.

---

## 4. Eval Harness Architecture (Planned)

```
eval/
├── tasks/                       # Task definitions (YAML)
│   ├── information_retrieval/
│   ├── data_analysis/
│   ├── multi_step_reasoning/
│   ├── api_orchestration/
│   ├── code_generation/
│   └── conversational/
│
├── adapters/                    # Framework wrappers (common interface)
│   ├── base.py                  # Abstract FrameworkAdapter
│   ├── pop_adapter.py           # pop adapter
│   ├── langchain_adapter.py     # LangChain/LangGraph adapter
│   └── bare_adapter.py          # Raw API baseline
│
├── scorers/                     # Scoring functions
│   ├── exact_match.py
│   ├── fuzzy_match.py
│   ├── llm_judge.py             # LLM-as-judge with rubric
│   └── composite.py
│
└── runners/
    ├── run_benchmark.py         # All tasks x all frameworks
    ├── run_dx_benchmark.py      # DX comparison
    └── run_overhead_benchmark.py # Overhead (mocked LLM)
```

**Common Adapter Interface:**
```python
@dataclass
class TaskResult:
    answer: str
    success: bool | None
    total_tokens: int
    llm_calls: int
    wall_clock_ms: float
    framework_overhead_ms: float
    peak_memory_bytes: int
    steps: list[dict]
    error: str | None

class FrameworkAdapter(ABC):
    def name(self) -> str: ...
    def create_agent(self, model: str, tools: list, system_prompt: str) -> object: ...
    def run(self, agent: object, task: str) -> TaskResult: ...
```

---

## 5. The Headline Benchmarks

### Measured and verified today

```
+---------------------------------+-----------+---------------+--------------+
| Metric                          | pop       | LangChain+LG  | vs.          |
+---------------------------------+-----------+---------------+--------------+
| Import time                     | ~0.16ms   | ~1,200ms      | ~7,000x down |
| Framework overhead per step     | ~0.15ms   | ~45ms         | ~300x down   |
| Lines of code (avg task)        | ~12       | ~42           | 71% fewer    |
| Dependencies                    | 2         | 20+           | 90% fewer    |
| Core source                     | 2,658     | ~188,000      | 1/71st size  |
+---------------------------------+-----------+---------------+--------------+
```

See `benchmarks/results/latest_report.md` for the full auto-generated report.

### Projected (pending task-suite implementation)

```
+---------------------------------+----------+---------------+---------+
| Metric                          | pop      | LangChain+LG  | vs.     |
+---------------------------------+----------+---------------+---------+
| Task success rate               | ~85%     | ~83%          | parity  |
| Cost per success ($)            | $0.012   | $0.018        | 33% down|
| LLM calls per task              | 3.2      | 4.1           | 22% down|
+---------------------------------+----------+---------------+---------+
```

These figures are architectural expectations — same model underneath means accuracy should be
at parity; fewer hidden injections means lower token overhead. Not yet confirmed by a task suite.

---

## 6. How Evals Drive Architecture

| Architecture Decision | Eval Metric It Optimizes |
|----------------------|--------------------------|
| Thin provider wrappers (no Runnable) | Framework overhead <5ms |
| @tool auto-schema from type hints | Tool calling accuracy |
| No hidden system prompt injection | Cost efficiency |
| Built-in Reflexion loop | Error recovery rate >80% |
| Async-first with sync wrapper | Concurrent throughput |
| Immutable state snapshots | Reliability (deterministic replay) |
| Single package, 2 deps | Import time, package size |
| Functions, not classes | DX: 71% fewer lines |
| Model fallback chain | Reliability (graceful degradation) |
| Step budget (max_steps) | Prevent infinite loops |

---

## 7. Development Workflow

```
For every PR:
1. Unit tests pass          (pytest)
2. Type check passes        (mypy --strict)
3. Lint passes              (ruff)
4. Benchmark does not regress (python benchmarks/bench_startup.py)

For every release (planned):
5. Full eval suite -- 100 tasks, real LLM, all 8 dimensions
6. Comparison published in release notes

Monthly (planned):
7. Cross-framework update -- re-run against latest competitor versions
```

---

## 8. What Makes Our Evals Credible

1. **Open-source** — anyone can run `python benchmarks/bench_startup.py` and verify
2. **No fabricated numbers** — results saved to `benchmarks/results/`, run from real processes
3. **Honest reporting** — this document distinguishes measured results from projections
4. **Methodology visible** — mock adapter approach for overhead isolation is in benchmark source
5. **Statistical** — multiple iterations, mean + min/max/median reported

---

## 9. Claims vs. Evidence

| Claim | Evidence | Status |
|-------|----------|--------|
| "Faster startup" | Import time benchmark | ~7,000x faster |
| "Less overhead" | Per-step overhead benchmark | ~300x less |
| "Simpler" | Lines of code comparison | 71% fewer lines |
| "Lighter" | Dependency count | 2 vs 20+ |
| "Same accuracy" | Task success rate | Planned |
| "Cheaper" | Cost per successful task | Planned |
| "More reliable" | Error recovery benchmark | Planned |

> **"~7,000x faster import. ~300x less per-step overhead. 71% less code. 2 dependencies."**
>
> Proven by measurement today. Task accuracy and cost data coming in Phase 1.
