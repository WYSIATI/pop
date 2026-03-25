# AgentLite → Final Name TBD: Complete Open-Source Project Plan

> Master plan for building, naming, launching, and growing a lightweight Python agent framework that aims to become the "uv of AI agents."

---

## Table of Contents

1. [Name & Branding](#1-name--branding)
2. [Positioning & Tagline](#2-positioning--tagline)
3. [Repository Structure](#3-repository-structure)
4. [Architecture Recap & Refinements](#4-architecture-recap--refinements)
5. [README Blueprint](#5-readme-blueprint)
6. [Documentation Plan](#6-documentation-plan)
7. [Examples & Demo Strategy](#7-examples--demo-strategy)
8. [Visual & Diagram Assets](#8-visual--diagram-assets)
9. [CI/CD & Quality](#9-cicd--quality)
10. [Community Infrastructure](#10-community-infrastructure)
11. [Launch Strategy](#11-launch-strategy)
12. [Growth & Sustainability](#12-growth--sustainability)
13. [Milestone Roadmap](#13-milestone-roadmap)
14. [Risk Analysis](#14-risk-analysis)

---

## 1. Name & Branding

### Name Candidates (Ranked)

After exhaustive PyPI + GitHub availability checks:

| Rank | Name | Chars | Available | Rationale |
|------|------|-------|-----------|-----------|
| 1 | **agynt** | 5 | PyPI ✅ GitHub ✅ | "Agent" respelled — short, punchy, immediately recognizable. Same energy as `ruff`, `uv`, `rye`. |
| 2 | **nimbl** | 5 | PyPI ✅ | "Nimble" without the e — evokes speed + agility. Ruff-style vowel dropping. |
| 3 | **lyte** | 4 | PyPI ✅ | "Light/Lite" respelled — ultra-short, signals lightweight. |
| 4 | **agntiq** | 6 | PyPI ✅ GitHub ✅ | "Agentic" respelled — trendy "iq" suffix implies intelligence. |
| 5 | **swif** | 4 | PyPI ✅ | "Swift" truncated — speed connotation, ultra-short. |
| 6 | **litegent** | 8 | PyPI ✅ GitHub ✅ | "Lite" + "agent" — most self-descriptive option. |
| 7 | **dispatr** | 7 | PyPI ✅ GitHub ✅ | "Dispatcher" — evokes routing/orchestration. |
| 8 | **agentlite** | 9 | PyPI ✅ | Direct compound — clear but longer. |

### Recommendation: **agynt**

**Why:**
- 5 characters — fast to type, impossible to forget
- Obviously means "agent" — zero explanation needed
- Same naming philosophy as the tools we admire: `uv` (2), `ruff` (4), `rye` (3), `nox` (3)
- Unique in search engines (no pollution from common English words)
- Works as both package name and brand: `pip install agynt`, `from agynt import Agent`
- Domain `agynt.dev` likely available

### Brand Identity

```
Logo concept:    Bold "pop" wordmark with coral ripple mark — flat, no gradients
Colors:          Coral (#FF6154) + dark slate (#1a1a2e) + muted gray (#94a3b8)
Font:            Inter for brand, monospace for code
Tone:            Technical, confident, no-BS. Like uv/ruff docs.
Mascot:          None. Clean and professional > cute.
```

### Package Naming

```
PyPI:            agynt
Import:          from agynt import Agent, tool
CLI (future):    agynt run my_agent.py
Docs URL:        agynt.dev or docs.agynt.dev
GitHub:          github.com/<org>/agynt
Discord:         discord.gg/agynt
```

---

## 2. Positioning & Tagline

### One-Liner (for GitHub description, PyPI, social)

> **agynt** — Fast, lean AI agents. 5 lines to production.

### Elevator Pitch (for README intro)

> agynt is a lightweight Python framework for building AI agents. It supports 7+ LLM providers, has 5 core concepts, and gets you from `pip install` to a working agent in under 2 minutes. It's what LangChain would look like if built today — without the 188K lines of abstraction.

### Positioning Statement

```
For:             Python developers building LLM-powered agents
Who need:        A production-ready agent framework that doesn't fight them
agynt is:        A fast, minimal agent framework
That:            Lets you build agents in 5 lines with any LLM provider
Unlike:          LangChain (over-engineered) or raw API calls (too low-level)
Our approach:    Functions, not classes. Loops, not graphs. Explicit, not magic.
```

### Anti-Positioning (what we're NOT)

- We're NOT a LangChain wrapper or plugin
- We're NOT trying to do everything (no built-in vector DB, no RAG pipeline, no UI)
- We're NOT opinionated about which LLM you use
- We're NOT a low-code/no-code tool — we're for developers who want control

### Competitive Anchoring

People will ask "how is this different from X?" Here are our answers:

| Question | Answer |
|----------|--------|
| vs LangChain | "LangChain is 188K lines with mandatory commercial deps. agynt is <4K lines with zero. It's uv to LangChain's Anaconda." |
| vs CrewAI | "CrewAI is opinionated about multi-agent roles. agynt gives you primitives — use them for multi-agent, single-agent, or simple workflows." |
| vs Pydantic-AI | "Similar philosophy! Pydantic-AI is tightly coupled to Pydantic. agynt is standalone with optional Pydantic support." |
| vs OpenAI Agents SDK | "OpenAI SDK locks you into OpenAI. agynt works with any provider — OpenAI, Anthropic, Gemini, DeepSeek, Kimi, MiniMax, GLM." |
| vs raw API calls | "agynt adds the agent loop, tool calling, memory, multi-agent, streaming, and error recovery — things that take 500+ lines to build yourself." |

---

## 3. Repository Structure

```
agynt/
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.yml              # Structured bug report form
│   │   └── feature_request.yml         # Feature request form
│   ├── PULL_REQUEST_TEMPLATE.md
│   ├── workflows/
│   │   ├── ci.yml                      # Test + lint on every PR
│   │   ├── release.yml                 # Auto-publish to PyPI on tag
│   │   └── docs.yml                    # Build + deploy docs
│   ├── FUNDING.yml                     # GitHub Sponsors
│   └── dependabot.yml
│
├── docs/
│   ├── index.md                        # Landing page
│   ├── getting-started/
│   │   ├── installation.md
│   │   ├── quickstart.md               # 2-minute quickstart
│   │   └── first-agent.md              # Tutorial: your first agent
│   ├── concepts/
│   │   ├── agents.md                   # What is an Agent
│   │   ├── tools.md                    # How tools work
│   │   ├── models.md                   # Multi-provider LLM layer
│   │   ├── memory.md                   # Memory system
│   │   └── multi-agent.md              # Multi-agent patterns
│   ├── guides/
│   │   ├── tool-definition.md          # Deep dive: defining tools
│   │   ├── structured-output.md        # Getting typed outputs
│   │   ├── streaming.md                # Streaming responses
│   │   ├── error-handling.md           # Error recovery patterns
│   │   ├── human-in-the-loop.md        # Confirmation flows
│   │   └── custom-providers.md         # Adding your own LLM provider
│   ├── patterns/
│   │   ├── workflows.md                # chain, route, parallel
│   │   ├── handoff.md                  # Agent handoff (Swarm-style)
│   │   ├── orchestrator.md             # Boss-worker pattern
│   │   ├── debate.md                   # Generator-critic pattern
│   │   └── fan-out.md                  # Parallel voting
│   ├── reference/
│   │   └── api.md                      # Auto-generated API docs
│   ├── comparison.md                   # vs LangChain, CrewAI, etc.
│   └── changelog.md
│
├── examples/
│   ├── 01_hello_agent.py               # Simplest possible agent (5 lines)
│   ├── 02_web_search_agent.py          # Agent with a search tool
│   ├── 03_structured_output.py         # Agent returning Pydantic models
│   ├── 04_streaming.py                 # Streaming agent responses
│   ├── 05_multi_provider.py            # Same agent, different providers
│   ├── 06_memory_agent.py              # Agent with persistent memory
│   ├── 07_customer_support.py          # Multi-agent handoff demo
│   ├── 08_research_agent.py            # Research + report writing
│   ├── 09_code_agent.py                # Coding assistant agent
│   ├── 10_data_pipeline.py             # Orchestrator-workers pattern
│   ├── 11_debate.py                    # Generator-critic pattern
│   ├── 12_human_in_the_loop.py         # Human confirmation flow
│   ├── 13_custom_provider.py           # Registering a custom LLM provider
│   └── 14_mcp_tools.py                 # Using MCP server tools
│
├── benchmarks/
│   ├── bench_startup.py                # Import + agent creation time
│   ├── bench_tool_call.py              # Tool calling overhead
│   ├── bench_streaming.py              # Time-to-first-token
│   └── bench_vs_langchain.py           # Head-to-head comparison
│
├── src/agynt/
│   ├── __init__.py                     # Agent, tool, run (3 exports)
│   ├── agent.py                        # Agent class (~200 lines)
│   ├── tool.py                         # @tool decorator (~100 lines)
│   ├── runner.py                       # Execution engine (~300 lines)
│   ├── state.py                        # AgentState, Step, Result (~150 lines)
│   ├── types.py                        # Shared types (~50 lines)
│   ├── models/
│   │   ├── __init__.py                 # model(), chat(), register_provider()
│   │   ├── router.py                   # Provider routing (~100 lines)
│   │   ├── base.py                     # ModelAdapter protocol (~50 lines)
│   │   ├── openai.py                   # OpenAI adapter (~150 lines)
│   │   ├── anthropic.py                # Anthropic adapter (~150 lines)
│   │   ├── gemini.py                   # Google Gemini adapter (~150 lines)
│   │   ├── deepseek.py                 # DeepSeek (OpenAI-compat, ~80 lines)
│   │   ├── kimi.py                     # Kimi/Moonshot (~80 lines)
│   │   ├── minimax.py                  # MiniMax (~100 lines)
│   │   └── glm.py                      # GLM/Zhipu (~80 lines)
│   ├── memory/
│   │   ├── __init__.py                 # MemoryBackend protocol
│   │   ├── inmemory.py                 # Default in-memory store
│   │   ├── sqlite.py                   # SQLite persistent store
│   │   └── base.py                     # Base protocol + types
│   ├── multi/
│   │   ├── __init__.py                 # pipeline, orchestrate, debate, fan_out
│   │   ├── handoff.py                  # Agent handoff mechanism
│   │   └── patterns.py                 # Composition patterns
│   ├── workflows/
│   │   ├── __init__.py                 # chain, route, parallel
│   │   └── patterns.py                 # Pre-agent patterns
│   └── hooks/
│       ├── __init__.py                 # Hook protocol
│       └── console.py                  # Console logger
│
├── tests/
│   ├── conftest.py                     # Shared fixtures, mock LLM
│   ├── test_agent.py                   # Agent unit tests
│   ├── test_tool.py                    # Tool decorator tests
│   ├── test_runner.py                  # Runner tests
│   ├── test_state.py                   # State management tests
│   ├── test_models/
│   │   ├── test_router.py
│   │   ├── test_openai.py
│   │   └── test_anthropic.py
│   ├── test_memory/
│   │   ├── test_inmemory.py
│   │   └── test_sqlite.py
│   ├── test_multi/
│   │   ├── test_handoff.py
│   │   └── test_patterns.py
│   └── test_workflows/
│       └── test_patterns.py
│
├── assets/
│   ├── logo.svg                        # Main logo
│   ├── logo-dark.svg                   # Logo for dark mode
│   ├── demo.gif                        # Terminal demo GIF
│   ├── architecture.png                # Architecture diagram
│   └── benchmark.png                   # Benchmark comparison chart
│
├── README.md                           # The sales page
├── LICENSE                             # MIT
├── CONTRIBUTING.md                     # How to contribute
├── CODE_OF_CONDUCT.md                  # Contributor Covenant
├── CHANGELOG.md                        # Keep a Changelog format
├── pyproject.toml                      # Modern Python packaging
├── mkdocs.yml                          # MkDocs Material config
└── .pre-commit-config.yaml             # Pre-commit hooks (ruff, mypy)
```

---

## 4. Architecture Recap & Refinements

### Core Architecture (5 concepts, ~3,000 lines)

```
┌─────────────────────────────────────────────────────────────────────┐
│                           USER CODE                                 │
│                                                                     │
│   from agynt import Agent, tool                                     │
│   agent = Agent(model="openai:gpt-4o", tools=[...])                │
│   result = agent.run("...")                                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────┐  ┌────────┐  ┌─────────┐  ┌────────┐  ┌──────────┐ │
│   │  Agent   │  │  Tool  │  │  Model  │  │ Memory │  │  Runner  │ │
│   │  (loop)  │  │(@tool) │  │(router) │  │ (opt)  │  │(execute) │ │
│   └────┬────┘  └───┬────┘  └────┬────┘  └───┬────┘  └────┬─────┘ │
│        └────────────┴───────────┴────────────┴────────────┘       │
│                              │                                     │
│   ┌──────────────────────────┴──────────────────────────────────┐  │
│   │                    Provider Adapters                         │  │
│   │  OpenAI │ Anthropic │ Gemini │ DeepSeek │ Kimi │ MM │ GLM  │  │
│   └─────────────────────────────────────────────────────────────┘  │
│                              │                                     │
│   ┌──────────────────────────┴──────────────────────────────────┐  │
│   │                     Hooks (opt-in)                           │  │
│   │     Tracing  │  Guardrails  │  Cost  │  Custom              │  │
│   └─────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Sync/Async | Async-first, sync wrapper | Modern Python, no method duplication |
| Serialization | None in core | User handles persistence if needed |
| Observability | Hook-based, opt-in | Zero overhead when not used |
| Dependency injection | `Context` object | Type-safe, testable, no global state |
| Output parsing | Pydantic (optional) | Industry standard, but not required |
| Tool schema | Auto from type hints | Zero boilerplate |
| State | Immutable snapshots | Debuggable, forkable |
| Error recovery | Reflexion built-in | Based on latest research |
| Multi-agent | Composition functions | pipeline/orchestrate/debate/fan_out |
| Provider compat | `provider:model` string | Zero-config, discoverable |

### What's NOT in Core (and why)

| Feature | Why excluded | Where it lives |
|---------|-------------|----------------|
| Vector DB | Not every agent needs RAG | `agynt[vectors]` extra |
| Web UI | Framework, not product | Separate repo |
| LangSmith/tracing | Must not be mandatory | `agynt[otel]` extra |
| Redis memory | Advanced use case | `agynt[redis]` extra |
| Graph execution | Most agents don't need it | Not planned (YAGNI) |
| Prompt templates | Python f-strings work | Not planned |

---

## 5. README Blueprint

The README is the single most important file. Here's the exact blueprint:

```markdown
<p align="center">
  <img src="assets/logo.svg" alt="agynt" width="300">
</p>

<p align="center">
  <em>Fast, lean AI agents. 5 lines to production.</em>
</p>

<p align="center">
  <a href="..."><img src="..." alt="CI"></a>
  <a href="..."><img src="..." alt="Coverage"></a>
  <a href="..."><img src="..." alt="PyPI"></a>
  <a href="..."><img src="..." alt="Python"></a>
  <a href="..."><img src="..." alt="License: MIT"></a>
  <a href="..."><img src="..." alt="Discord"></a>
</p>

<p align="center">
  <a href="https://agynt.dev">Documentation</a> •
  <a href="https://github.com/org/agynt">Source Code</a> •
  <a href="https://discord.gg/agynt">Discord</a>
</p>

---

**agynt** is a lightweight Python framework for building AI agents. It supports
7+ LLM providers, has 5 core concepts, and gets you from install to working
agent in under 2 minutes.

## Why agynt?

- **5 lines to a working agent** — not 50. No boilerplate, no base classes.
- **Any LLM provider** — OpenAI, Anthropic, Gemini, DeepSeek, Kimi, MiniMax, GLM.
  Switch with one string.
- **~3,000 lines of code** — you can read the entire framework in an afternoon.
  (LangChain is 188,000 lines.)
- **Zero mandatory commercial dependencies** — no forced telemetry, no langsmith.
- **Errors in YOUR code** — not 6,000 lines of framework stack trace.
- **Async-first** with sync wrappers — no duplicate `invoke`/`ainvoke` methods.

## Install

```bash
pip install agynt
```

## Quick Start

```python
from agynt import Agent, tool

@tool
def search(query: str) -> str:
    """Search the web."""
    return web_search(query)

agent = Agent(model="openai:gpt-4o", tools=[search])
result = agent.run("What happened in AI today?")
print(result.output)
```

That's it. No `StateGraph`, no `RunnableSequence`, no `ChannelWrite`.

## Switch Providers in One Line

```python
agent = Agent(model="anthropic:claude-sonnet-4-20250514", tools=[search])   # Anthropic
agent = Agent(model="deepseek:deepseek-chat", tools=[search])         # DeepSeek
agent = Agent(model="gemini:gemini-2.0-flash", tools=[search])        # Google
agent = Agent(model="kimi:moonshot-v1-auto", tools=[search])          # Kimi
```

## Streaming

```python
async for event in agent.stream("Analyze Q3 earnings"):
    match event:
        case TextDelta(text=t): print(t, end="")
        case ToolCall(name=n):  print(f"\n🔧 {n}")
```

## Multi-Agent (Handoff)

```python
billing = Agent(model="openai:gpt-4o-mini", tools=[lookup_invoice])
tech = Agent(model="openai:gpt-4o-mini", tools=[check_logs])

triage = Agent(
    model="openai:gpt-4o-mini",
    tools=[
        handoff(billing, when="billing issues"),
        handoff(tech, when="tech issues"),
    ],
)
result = triage.run("I was charged twice")
# → automatically routes to billing agent
```

## Benchmarks

| Metric | agynt | LangChain + LangGraph |
|--------|-------|----------------------|
| Import time | ~50ms | ~1,200ms |
| Agent creation | ~1ms | ~15ms |
| Lines for basic agent | 5 | 35+ |
| Core code size | ~3K lines | ~188K lines |
| Dependencies | 3 | 20+ |

[→ Full benchmark methodology](./benchmarks/)

## Documentation

📖 **[agynt.dev](https://agynt.dev)** — Getting started, guides, API reference, examples.

## Community

- 💬 [Discord](https://discord.gg/agynt)
- 🐛 [Issues](https://github.com/org/agynt/issues)
- 💡 [Discussions](https://github.com/org/agynt/discussions)

## License

MIT
```

---

## 6. Documentation Plan

### Stack

- **MkDocs + Material for MkDocs** (same as FastAPI, uv, Pydantic)
- Hosted on GitHub Pages via CI
- Custom domain: `agynt.dev`

### Structure (Diátaxis Framework)

| Type | Purpose | Examples |
|------|---------|---------|
| **Tutorials** | Learning-oriented | "Your First Agent", "Build a Research Bot" |
| **How-to Guides** | Task-oriented | "How to Define Tools", "How to Use Streaming" |
| **Reference** | Information-oriented | API docs (auto-generated from docstrings) |
| **Explanation** | Understanding-oriented | "Why Loops, Not Graphs", "How Memory Works" |

### Launch-Day Docs (Minimum Viable)

1. Installation
2. Quickstart (2 minutes)
3. Core Concepts (Agent, Tool, Model, Memory, Runner)
4. 5 examples with explanations
5. API reference
6. Comparison with alternatives

### Post-Launch Docs (Month 1)

7. Migration guide from LangChain
8. Each multi-agent pattern explained
9. Custom provider guide
10. Cookbook / recipes
11. FAQ

---

## 7. Examples & Demo Strategy

### The 14 Examples (Progressive Complexity)

| # | Example | What it teaches | Lines |
|---|---------|----------------|-------|
| 01 | Hello Agent | Minimal agent, one tool | 5 |
| 02 | Web Search Agent | Real tool integration | 12 |
| 03 | Structured Output | Pydantic output types | 15 |
| 04 | Streaming | Real-time token streaming | 15 |
| 05 | Multi-Provider | Same agent, 5 providers | 20 |
| 06 | Memory Agent | Persistent conversation memory | 18 |
| 07 | Customer Support | Multi-agent handoff | 30 |
| 08 | Research Agent | Tool use + self-reflection | 20 |
| 09 | Code Agent | File ops + shell execution | 25 |
| 10 | Data Pipeline | Orchestrator-workers | 35 |
| 11 | Debate | Generator-critic loop | 25 |
| 12 | Human-in-the-Loop | Confirmation flows | 20 |
| 13 | Custom Provider | Registering new LLM provider | 25 |
| 14 | MCP Tools | Using MCP servers | 15 |

### Demo GIF Strategy

Create 3 terminal recordings using `vhs` (Charm):

1. **Hero GIF** (for README top): Install → create agent → run → see result. 15 seconds.
2. **Provider switching GIF**: Same agent running on 4 different providers. 20 seconds.
3. **vs LangChain GIF**: Side-by-side LOC comparison. 10 seconds.

### Killer Demo App

**"Research Assistant in 10 Lines"** — A complete agent that:
1. Takes a research question
2. Searches the web
3. Reads relevant articles
4. Synthesizes a report with citations

This becomes the centerpiece of every launch post and talk.

---

## 8. Visual & Diagram Assets

### Required Assets

| Asset | Format | Purpose | Where used |
|-------|--------|---------|------------|
| Logo | SVG | Brand identity | README, docs, social |
| Logo (dark) | SVG | Dark mode | README, docs |
| Architecture diagram | PNG/SVG | System overview | README, docs, talks |
| Agent loop diagram | PNG/SVG | Core concept | Docs, blog |
| Provider diagram | PNG/SVG | Multi-LLM support | README, docs |
| Memory tiers diagram | PNG/SVG | Memory architecture | Docs |
| Multi-agent patterns | PNG/SVG | Pattern catalog | Docs |
| Workflow levels | PNG/SVG | Progressive complexity | Docs, blog |
| Benchmark chart | PNG/SVG | Performance comparison | README, blog |
| Demo GIF | GIF | Quick demo | README |

### Diagram Style Guide

- Use **Excalidraw** (hand-drawn style) for architecture diagrams — it's trendy and approachable
- Use **Mermaid** in docs for inline diagrams — renders natively in GitHub and MkDocs
- Colors: Match brand palette (coral #FF6154, dark slate #1a1a2e, muted gray #94a3b8)
- Keep diagrams simple — max 10 elements per diagram
- Every diagram should be understandable in 5 seconds

### Full Diagram Set (from FRAMEWORK_ARCHITECTURE.md)

All 11 Mermaid diagrams from the architecture doc will be:
1. Rendered as high-res PNGs for README/social
2. Kept as Mermaid source in docs for editability
3. Styled consistently with brand colors

---

## 9. CI/CD & Quality

### GitHub Actions Workflows

**`ci.yml`** — Runs on every PR:
```yaml
- Python 3.10, 3.11, 3.12, 3.13
- ruff check + ruff format
- mypy (strict)
- pytest with coverage
- Coverage gate: >80%
```

**`release.yml`** — Runs on version tags:
```yaml
- Build wheel
- Publish to PyPI (trusted publisher)
- Create GitHub Release with changelog
- Deploy docs
```

**`docs.yml`** — Runs on main merge:
```yaml
- Build MkDocs
- Deploy to GitHub Pages
```

### Quality Standards

| Metric | Target |
|--------|--------|
| Test coverage | >90% |
| Type coverage (mypy) | 100% strict |
| Linting (ruff) | Zero warnings |
| Import time | <100ms |
| Core code size | <5,000 lines |
| Public API surface | <30 functions/classes |

### Testing Strategy

```
tests/
├── unit/              # Fast, mock LLM responses, test framework logic
├── integration/       # Slow, real API calls (run in CI with secrets)
└── examples/          # Verify all examples still work
```

Use a `MockModel` fixture that returns predictable responses for unit tests. Real provider tests run in a separate CI job with API keys.

---

## 10. Community Infrastructure

### Day 1 Setup

| Platform | Purpose | Setup |
|----------|---------|-------|
| **Discord** | Real-time community | Channels: #announcements, #help, #showcase, #dev, #off-topic |
| **GitHub Discussions** | Q&A, ideas, RFCs | Categories: Questions, Ideas, Show & Tell, RFCs |
| **Twitter/X** | Announcements, engagement | Official account: @agynt_dev |

### Contributor Experience

**CONTRIBUTING.md** covers:
1. Development setup (single `uv sync` command)
2. Running tests
3. Code style (enforced by ruff)
4. PR process
5. Where to find "good first issue" labels

**Good First Issues** (seed 5-10 before launch):
- "Add timeout parameter to @tool decorator"
- "Improve error message when API key is missing"
- "Add example: agent with calculator tool"
- "Support response_format in OpenAI adapter"
- "Add token counting to AgentResult"

### Contributor Recognition

- All-contributors bot in README
- Discord "Contributor" role
- Shoutouts in release notes

---

## 11. Launch Strategy

### Timeline

```
Week -4:  Start building in public (Twitter threads about design decisions)
Week -3:  Alpha release to 15-20 early testers
Week -2:  Incorporate feedback, polish README, record demos
Week -1:  Final polish, prepare all launch content
Day 0:    LAUNCH
Week +1:  Respond, fix, iterate
Week +2:  Tutorials, blog posts, newsletter submissions
```

### Launch Day Sequence

| Time (EST) | Action | Platform |
|------------|--------|----------|
| 8:00 AM | Publish blog post | Personal blog / Medium |
| 8:15 AM | "Show HN" post | Hacker News |
| 8:30 AM | Launch thread (8 tweets) | Twitter/X |
| 9:00 AM | Post to r/Python | Reddit |
| 10:00 AM | Post to r/MachineLearning | Reddit |
| 11:00 AM | Share in AI Discord servers | Discord |
| 12:00 PM | LinkedIn announcement | LinkedIn |
| 2:00 PM | Post to r/LocalLLaMA | Reddit |
| All day | Respond to every comment | All platforms |

### Show HN Post (Draft)

```
Title: Show HN: agynt – Fast, lean AI agents in Python. 5 lines, 7 providers, 3K lines of code

Body:
Hi HN, I built agynt because I got tired of fighting LangChain's 188K lines
of abstraction to do simple things.

agynt is a Python agent framework where:
- A tool is a decorated function (not a 1,586-line BaseTool class)
- An agent is a loop (not a graph compilation pipeline)
- Provider switching is one string: "openai:gpt-4o" → "anthropic:claude-sonnet-4-20250514"
- The entire framework is ~3,000 lines you can read in one sitting

It supports OpenAI, Anthropic, Gemini, DeepSeek, Kimi, MiniMax, and GLM.

I know the "yet another AI framework" eye-roll is deserved. But I genuinely
think the space needs a minimal, production-quality option — the way uv
brought sanity to Python packaging.

GitHub: [link]
Docs: [link]
PyPI: pip install agynt

Happy to discuss the architecture, tradeoffs, or why I think graphs are
overkill for 90% of agent use cases.
```

### Twitter Launch Thread (Draft)

```
🧵 1/8: I just open-sourced agynt: a fast, lean AI agent framework for Python.

Build a working agent in 5 lines. Support 7+ LLM providers. The entire
framework is 3,000 lines of code.

Here's why I built it and how it works: 👇

[Hero GIF]

---

2/8: The problem with current agent frameworks:

• LangChain: 188K lines, 685 classes, mandatory commercial deps
• Most alternatives: still too complex for what agents actually need
• Raw API calls: too low-level, you rebuild the same loop every time

---

3/8: agynt's philosophy:

Functions, not classes.
Loops, not graphs.
Explicit, not magic.
5 concepts: Agent, Tool, Model, Memory, Runner.

[Code screenshot: 5-line agent]

---

4/8: Switch providers in one line:

model="openai:gpt-4o"
model="anthropic:claude-sonnet-4-20250514"
model="deepseek:deepseek-chat"
model="kimi:moonshot-v1-auto"

No separate packages. No version matrix. One string.

---

5/8: Multi-agent in 10 lines:

[Code screenshot: handoff example]

Built-in patterns: handoff, pipeline, orchestrate, debate, fan-out.

---

6/8: What's NOT in the framework (by design):

❌ No mandatory tracing/telemetry
❌ No vendor lock-in
❌ No graph compilation
❌ No 254-method base class

Just agents that work.

---

7/8: Benchmarks:

Import time: 50ms (vs 1,200ms)
Agent creation: 1ms (vs 15ms)
Dependencies: 3 (vs 20+)
Code size: 3K lines (vs 188K)

---

8/8: Try it:

pip install agynt
GitHub: [link]
Docs: agynt.dev

MIT licensed. Built for developers who want control.

If this is useful, a ⭐ on GitHub helps others find it.
```

---

## 12. Growth & Sustainability

### Content Calendar (First 3 Months)

| Week | Content | Platform |
|------|---------|----------|
| 1 | Launch posts | HN, Reddit, Twitter, LinkedIn |
| 2 | "What I learned from launch" | Blog, Twitter |
| 3 | "Build a Research Agent with agynt" | Blog, Dev.to |
| 4 | "agynt vs LangChain: Honest Comparison" | Blog (high SEO value) |
| 5 | v0.2 release + announcement | GitHub, Twitter |
| 6 | "Multi-Agent Patterns in Python" | Blog, r/Python |
| 7 | Video tutorial: "agynt in 10 minutes" | YouTube, Twitter |
| 8 | "How agynt's Provider System Works" | Blog (technical deep-dive) |
| 9 | v0.3 release + announcement | GitHub, Twitter |
| 10 | Conference talk submission | PyCon, local meetup |
| 11 | "Production Agent Architecture with agynt" | Blog |
| 12 | "Month 3: Lessons from Building an OSS Framework" | Blog, HN |

### Revenue / Sustainability Options (Future)

| Option | Model | When |
|--------|-------|------|
| GitHub Sponsors | Donations | From day 1 |
| Hosted docs (premium) | SaaS | 6+ months |
| Enterprise support | Consulting | When companies adopt |
| Hosted agent runtime | SaaS | 12+ months |

### Star Growth Targets

| Milestone | Target Date | Stars |
|-----------|-------------|-------|
| Launch day | Day 1 | 100+ |
| Week 1 | Day 7 | 500+ |
| Month 1 | Day 30 | 2,000+ |
| Month 3 | Day 90 | 5,000+ |
| Month 6 | Day 180 | 10,000+ |
| Year 1 | Day 365 | 20,000+ |

These targets assume successful HN + Twitter launch + consistent content.

---

## 13. Milestone Roadmap

### v0.1.0 — "Hello World" (Launch)

- [ ] Core agent loop (ReAct)
- [ ] @tool decorator with auto-schema
- [ ] OpenAI + Anthropic + DeepSeek adapters
- [ ] Sync + async execution
- [ ] Basic streaming
- [ ] In-memory state
- [ ] 5 examples
- [ ] README + quickstart docs
- [ ] PyPI package
- [ ] CI/CD pipeline

### v0.2.0 — "Multi-Provider" (Week 3-4)

- [ ] Gemini + Kimi + MiniMax + GLM adapters
- [ ] Model fallback chain
- [ ] Structured output (Pydantic)
- [ ] Conversation memory (sliding window + summary)
- [ ] Console hook (pretty logging)
- [ ] 10 examples
- [ ] Full docs site

### v0.3.0 — "Multi-Agent" (Month 2)

- [ ] Agent handoff (Swarm-style)
- [ ] pipeline() pattern
- [ ] orchestrate() pattern
- [ ] debate() pattern
- [ ] fan_out() pattern
- [ ] Human-in-the-loop (confirm_before)
- [ ] 14 examples

### v0.4.0 — "Memory & Recovery" (Month 3)

- [ ] SQLite persistent memory
- [ ] Core memory (always-in-context)
- [ ] Episodic memory (recall/memorize tools)
- [ ] Reflexion (reflect_on_failure)
- [ ] Checkpoint + resume
- [ ] Budget limits (cost, tokens, steps)

### v0.5.0 — "Ecosystem" (Month 4)

- [ ] MCP tool support
- [ ] OpenTelemetry hook
- [ ] Custom provider registration
- [ ] Migration guide from LangChain
- [ ] Benchmark suite
- [ ] Plugin system for community extensions

### v1.0.0 — "Production Ready" (Month 6)

- [ ] Stable API (no breaking changes after this)
- [ ] 100% type coverage
- [ ] >90% test coverage
- [ ] Battle-tested by community
- [ ] Enterprise documentation
- [ ] Security audit

---

## 14. Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Name "agynt" already claimed before we launch | Low | High | Register PyPI + domain immediately. Have backup names ready. |
| LangChain simplifies and kills our positioning | Medium | High | Move fast. Build community. Our "3K lines" advantage is structural. |
| Anthropic/OpenAI release a perfect framework | Medium | High | We're provider-agnostic. Their SDKs lock you in. |
| "Yet another framework" fatigue | High | Medium | Lead with benchmarks + real differentiation, not just claims. |
| Initial launch flops | Medium | Medium | Build in public pre-launch. Have multiple launch vectors. |
| Key provider API changes break adapters | High | Low | Thin adapters are easy to update. Community can contribute. |
| Contributors don't come | Medium | Medium | Make contributing frictionless. Seed good-first-issues. |
| Framework too minimal for real use cases | Low | High | Progressive complexity. Examples prove real-world usage. |

---

## Appendix: The "10-Second Test"

Every piece of our public presence must pass the 10-second test:

> A developer lands on the repo/docs/tweet. In 10 seconds, they should understand:
> 1. What this is (an AI agent framework)
> 2. Why they should care (faster, simpler than LangChain)
> 3. How to try it (pip install agynt)

If any of those takes more than 10 seconds, we need to simplify.
