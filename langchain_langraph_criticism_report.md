# Comprehensive Report: LangChain & LangGraph Complaints, Drawbacks, and Criticisms

*Research compiled from source code analysis, Hacker News discussions, GitHub issue data, PyPI dependency analysis, and developer community sentiment.*

---

## 1. COMPLEXITY ISSUES

### Over-Abstraction ("Wrapping 2 lines in 2,000 lines")

The single most repeated complaint across all developer communities is that LangChain wraps trivially simple operations in layers of unnecessary abstraction.

**Source code evidence:**
- The `Runnable` base class alone (`runnables/base.py`) is **6,261 lines** with **254 methods/functions** and **14 classes** in one file
- A simple chat model call goes through a 5-level inheritance chain: `BaseChatModel -> BaseLanguageModel -> RunnableSerializable -> Serializable + Runnable`
- Just to invoke a chat model, the framework loads code from: `base.py` (6,261 lines) + `serializable.py` (388 lines) + `language_models/base.py` (391 lines) + `chat_models.py` (1,834 lines) + `config.py` (641 lines) = **9,515 lines of framework code for a single API call**

**Community quotes (HN, 268+ upvotes on the original critique):**
- "Calling LLM API is just 2 lines of code. LangChain is a framework for sequential api calling, wrapping 2 lines of code with 2 thousand lines of code. It's like FizzBuzzEnterpriseEdition."
- "The core data structure, the Chain, is basically just a function. Combining chains is function composition -- literally f(g(x)) -- but incompatible with YOUR f's and g's without an adapter."
- "They have made a basic print('hello world') into HelloWorldPrint(Baseprint): @validators def input_variables..."
- "+1, but I figured it out in an hour. Langchain seemed like a ridiculous overcomplication of what would otherwise be basic Python."

### Steep Learning Curve

- The `Runnable` base class exposes ~50 public methods including: `invoke`, `ainvoke`, `batch`, `abatch`, `batch_as_completed`, `abatch_as_completed`, `stream`, `astream`, `astream_log`, `astream_events`, `transform`, `atransform`, `bind`, `with_config`, `with_listeners`, `with_alisteners`, `with_types`, `with_retry`, `map`, `with_fallbacks`, `pipe`, `pick`, `assign`, `as_tool`
- **319 classes** in langchain-core alone, **244 classes** in langchain-classic, **122 classes** in langgraph
- 64 abstract methods, 116 properties across langchain-core
- Developers must understand Runnable protocol, LCEL (LangChain Expression Language), callback system, tracer system, and serialization system before they can productively use the framework

### Total Code Volume

| Package | Lines of Code | Files |
|---------|--------------|-------|
| langchain-core | 63,725 | 175 |
| langchain-classic | 70,649 | 1,321 |
| langgraph | 19,773 | 66 |
| langgraph-prebuilt | 10,951 | 21 |
| langgraph-checkpoint | 8,628 | 24 |
| langgraph-sdk | 14,219 | 42 |
| **TOTAL** | **187,945** | **1,649** |

For context, a competing approach to the same problem space could be built in a few hundred to a few thousand lines of code.

---

## 2. PERFORMANCE ISSUES

### Overhead from Abstraction Layers

- Every LLM call passes through the full Runnable pipeline: config resolution, callback manager setup, tracer initialization, schema validation, serialization checks
- The callback system (`callbacks/manager.py` alone is **2,697 lines**) fires on every invocation even when no callbacks are registered
- The tracer system adds **5,057 lines** of always-loaded code across 10+ files

### Memory Bloat from Import Chain

- Just importing `langchain-core` pulls in: `pydantic`, `jsonpatch`, `langsmith`, `packaging`, `pyyaml`, `tenacity`, `typing-extensions`, `uuid-utils`
- `langsmith` itself pulls in: `httpx`, `orjson`, `requests`, `requests-toolbelt`, `zstandard`, `xxhash`, `uuid-utils`
- Total transitive dependency count for a basic LangChain + LangGraph setup is enormous

### Documented User Experiences

- "Perf was horrible and it spent my entire OpenAI API quota in an hour or so. Decided to reimplement using embeddings and my own glue code. Took like a week (much less than the langchain work), and it's cheaper and better."
- "Debugging LangChain performance and bugs is just an exercise in frustration."

---

## 3. DEVELOPER EXPERIENCE ISSUES

### Debugging Difficulty

- Stack traces pass through multiple layers of abstraction (Runnable -> RunnableSerializable -> Serializable -> actual class), making errors extremely hard to trace
- The callback/tracer system adds additional stack frames
- Async code paths duplicate sync paths but with different failure modes
- Error messages often reference internal framework concepts (`RunnableSequence`, `RunnableParallel`, `ChannelWrite`) rather than user-domain concepts

### Documentation Problems

- "Worst part of langchain is the 'documentation'. Especially the one about the javascript implementation."
- "The thing with langchain is that its documentation is terrible. If I want to do something I have to rely on other sources to use it."
- Documentation serves as content marketing for LangSmith/LangGraph Platform paid services, pushing users toward more complex and possibly unnecessary directions
- One HN commenter: "My problem with LangChain is that now it's a marketing tool for LangGraph Platform and LangSmith. Their docs (incl. getting started tutorials) are content marketing for the platform services."

### Rapid API Churn / Breaking Changes

- **84 deprecation markers** in langchain-core alone
- **104 deprecation markers** in langgraph
- LangGraph has three deprecation tiers already: `LangGraphDeprecatedSinceV05`, `LangGraphDeprecatedSinceV10`, `LangGraphDeprecatedSinceV11`
- The main library was recently renamed from `langchain` to `langchain_classic`, indicating yet another restructuring
- "I think it's a problem that the use in production is very peaky because all the changes that should be used properly by minor, patch version, etc. are all lumped together." (version was 0.0.234 at the time)
- AI coding assistants frequently generate outdated LangGraph code: "Its knowledge is often outdated. It suggests code based on its training data or old blog posts, which might be for v0.1 of a library when I'm using v0.2."

### Error Messages Are Inscrutable

- LangGraph's `ErrorCode` enum includes codes like `GRAPH_RECURSION_LIMIT`, `INVALID_CONCURRENT_GRAPH_UPDATE`, `INVALID_GRAPH_NODE_RETURN_VALUE` that require visiting documentation URLs to understand
- The `InvalidUpdateError` message: "Raised when attempting to update a channel with an invalid set of updates" -- tells users nothing about what they did wrong
- Bug issues like "KeyError on every rename command" (20 comments) and "unexpected keyword argument when using ChatAnthropic" (13 comments) persist as open bugs

---

## 4. ARCHITECTURE ISSUES

### Tight Coupling to LangSmith (Paid Service)

- `langsmith` is a **mandatory dependency** of `langchain-core` -- not optional
- This means every LangChain installation includes a client for a commercial observability service
- `langsmith` brings 9 additional dependencies: `httpx`, `orjson`, `requests`, `requests-toolbelt`, `zstandard`, `xxhash`, etc.
- There is no way to use langchain-core without installing langsmith

### Ecosystem Fragmentation

The LangChain monorepo contains **21 pyproject.toml** files (21 separate packages):
- `langchain-core`, `langchain` (now `langchain-classic`), `langchain-community`
- 16 partner integration packages (`langchain-openai`, `langchain-anthropic`, `langchain-aws`, etc.)
- `langchain-text-splitters`, `langchain-standard-tests`

LangGraph adds another set:
- `langgraph`, `langgraph-checkpoint`, `langgraph-checkpoint-postgres`, `langgraph-checkpoint-sqlite`, `langgraph-prebuilt`, `langgraph-sdk`, `langgraph-cli`

**Version coordination is a nightmare:**
- `langchain` requires `langchain-core<2.0.0,>=1.2.10` AND `langgraph<1.2.0,>=1.1.1`
- `langgraph` requires `langchain-core>=0.1` AND `langgraph-checkpoint<5.0.0,>=2.1.0` AND `langgraph-prebuilt<1.1.0,>=1.0.8` AND `langgraph-sdk<0.4.0,>=0.3.0`
- Upgrading any one package can break compatibility with others

### Unnecessary Abstractions

- **Callback system** (4,677 lines): A complex pub-sub system baked into the core, even for users who never need observability
- **Tracer system** (5,057 lines): Always loaded, tightly coupled to LangSmith
- **Serialization system** (`Serializable` base class): Every LangChain object inherits from `Serializable`, adding overhead for a feature most users never use
- **Runnable protocol** (13,730 lines just in the `runnables/` module): An entire execution framework when most users just want to call functions
- **Schema introspection**: Every Runnable computes `input_schema`, `output_schema`, and `config_schema` using heavy Pydantic model generation at runtime

### Dependency Bloat

Core required dependencies for a minimal langchain + langgraph setup:
```
langchain-core -> jsonpatch, langsmith, packaging, pydantic, pyyaml, tenacity, typing-extensions, uuid-utils
langsmith -> httpx, orjson, requests, requests-toolbelt, zstandard, xxhash, uuid-utils
langgraph -> langgraph-checkpoint, langgraph-prebuilt, langgraph-sdk, pydantic, xxhash
langgraph-checkpoint -> ormsgpack
langchain -> pydantic
```

Minimum ~20 transitive dependencies for a "hello world" LLM call.

---

## 5. API DESIGN ISSUES

### Verbose, Non-Pythonic APIs

To create a simple tool in LangChain:
```python
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    query: str = Field(description="search query")

class SearchTool(BaseTool):
    name: str = "search"
    description: str = "Search the web"
    args_schema: type[BaseModel] = SearchInput

    def _run(self, query: str) -> str:
        return do_search(query)
```

vs. what it could be:
```python
@tool
def search(query: str) -> str:
    """Search the web"""
    return do_search(query)
```

The `BaseTool` class is **1,586 lines** with 12 classes in the tools module totaling **2,793 lines**.

### Inconsistent Interfaces

- `BaseChatModel` has 30 methods, many with overlapping functionality (`generate` vs `invoke`, `generate_prompt` vs `generate`)
- Every class duplicates sync/async variants: `invoke`/`ainvoke`, `batch`/`abatch`, `stream`/`astream`, `generate`/`agenerate`
- The `Runnable` protocol forces every component into the same interface regardless of whether it makes sense (e.g., a prompt template doesn't naturally "stream")

### LCEL (LangChain Expression Language) is Confusing

- The `|` operator overloading for chaining looks clever in demos but becomes opaque in production code
- Debugging a `RunnableSequence` built with `|` operators is harder than debugging a simple function call chain
- LCEL creates implicit `RunnableParallel` from dict literals, which can confuse developers

---

## 6. LANGGRAPH-SPECIFIC ISSUES

### Graph Abstraction Adds Unnecessary Complexity

- The core `Pregel` execution engine (`pregel/main.py`) is **3,669 lines** with 57 imports
- It uses **19 internal `CONFIG_KEY_*` constants** (e.g., `__pregel_send`, `__pregel_read`, `__pregel_checkpointer`, `__pregel_stream`, `__pregel_resuming`, etc.) that leak into debug output
- The `StateGraph` class uses 9 different channel types (`LastValue`, `BinaryOperatorAggregate`, `EphemeralValue`, `NamedBarrierValue`, `Topic`, `AnyValue`, `UntrackedValue`, etc.) for what is conceptually just "passing state between functions"

### State Management Overhead

- State is defined via `TypedDict` with `Annotated` types and custom reducer functions -- a pattern that is unfamiliar to most Python developers
- The `add_messages` reducer for message lists is a common source of confusion
- Channel system (918 lines across 9 files) is an internal Pregel concept that leaks through to error messages
- `CompiledStateGraph` inherits from `Pregel` which uses `RunnableSequence` from langchain-core, coupling graph execution to the entire Runnable abstraction

### Simple Agents Require Graph Boilerplate

- The prebuilt `create_react_agent` is **1,015 lines** for what is conceptually: "call LLM, if it wants to use a tool call the tool, repeat"
- Defining a custom agent requires: defining state as TypedDict, creating a StateGraph, adding nodes, adding edges (including conditional edges), compiling, then invoking
- Error handling requires understanding `GraphRecursionError`, `InvalidUpdateError`, `GraphBubbleUp`, `GraphInterrupt`, `NodeInterrupt`, `ParentCommand`, `EmptyInputError`, `TaskNotFound`

### Graph Paradigm is Overkill for Most Use Cases

- Most agent workflows are simple loops (call LLM -> maybe use tool -> repeat), not complex DAGs
- The graph abstraction forces developers to think in terms of nodes, edges, and state channels when they just want an agent loop
- HN commenter: "Keep the graph small, the prompts concise, the nodes and tools at [a minimum]. It's not super complex, in fact that seems to be the only way to get a more or less reliable agent right now."
- The Pregel execution model (from distributed graph processing research) is dramatically over-engineered for most LLM agent workloads

### Checkpoint System Complexity

- `langgraph-checkpoint` is 8,628 lines for state persistence
- Separate packages needed for different backends: `langgraph-checkpoint-postgres`, `langgraph-checkpoint-sqlite`
- Tight coupling between checkpointing and graph execution makes it hard to use either independently

---

## 7. COMMUNITY SENTIMENT SUMMARY

### The Dominant Pattern: "Good for Prototyping, Bad for Production"

The overwhelming consensus across hundreds of comments:

1. **Use LangChain for learning/prototyping**: "LangChain is perfect to give you ideas for how to interact with LLMs, but for me it's been easier to implement everything myself."
2. **Rewrite for production**: "It's absolutely great for prototyping and simple scripts, but I don't think I would use it in a production codebase."
3. **The value is in the cookbook, not the code**: "I use it for inspiration and maybe a quick prototype to understand something, but I usually implement the pieces myself."
4. **Rolling your own is faster**: "Tried langchain briefly. Then realized it was just way faster and easier to write some basic Python scripts to glue together any APIs I was calling."
5. **10x speed improvement after ditching it**: "I ended up rewriting my own tools which goes 10x faster for me personally."

### GitHub Issue Data

- **141 open bug issues** on the langchain repo
- Top issues include: broken tool calls with specific models, ignored system prompts, KeyErrors in middleware, Pydantic schema generation bugs, broken HTTP client customization
- 33-comment bug on broken tool calls with `create_react_agent`

---

## 8. KEY DESIGN PRINCIPLES FOR A COMPETING FRAMEWORK

Based on this analysis, a competing framework should:

1. **Minimize abstraction layers**: A tool should be a decorated function. An agent should be a loop. A chain should be a function pipeline.
2. **Zero mandatory commercial dependencies**: No langsmith-equivalent forced into core.
3. **Single package, minimal deps**: One pip install, minimal transitive dependencies.
4. **Transparent debugging**: Stack traces should point to user code, not framework internals.
5. **No Runnable protocol**: Functions are already composable in Python. Don't reinvent function composition.
6. **Simple state management**: Use plain dictionaries or dataclasses, not channel-based state machines.
7. **Agent loops, not graphs**: Most agents are loops. Provide a simple loop abstraction; offer graph capabilities as an optional layer for the rare case that needs it.
8. **Stable API**: Avoid version churn; use semantic versioning properly.
9. **Documentation that teaches, not sells**: Keep docs focused on solving user problems, not funneling into paid services.
10. **Async-first but sync-simple**: Don't duplicate every method with async variants; use a single smart approach.
