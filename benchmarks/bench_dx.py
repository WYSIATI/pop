"""Benchmark: Developer Experience (Lines of Code)
Compares the lines needed for common tasks in pop vs LangChain.
"""

from __future__ import annotations

import json
import pathlib

# Hand-counted from actual working code (see examples/ for pop, LangChain docs for LangChain)
dx_comparison: dict[str, dict] = {
    "hello_world_agent": {
        "pop": 5,
        "langchain": 35,
        "description": "Simplest possible agent with one tool",
    },
    "web_search_agent": {
        "pop": 12,
        "langchain": 45,
        "description": "Agent with search tool and instructions",
    },
    "structured_output": {
        "pop": 15,
        "langchain": 40,
        "description": "Agent returning structured Pydantic data",
    },
    "streaming": {
        "pop": 15,
        "langchain": 50,
        "description": "Streaming agent with event handling",
    },
    "multi_provider": {
        "pop": 20,
        "langchain": 60,
        "description": "Same agent across multiple LLM providers",
    },
    "tool_definition": {
        "pop": 6,
        "langchain": 15,
        "description": "Define a single tool",
    },
    "multi_agent_handoff": {
        "pop": 15,
        "langchain": 55,
        "description": "Two-agent handoff (customer support)",
    },
    "agent_with_memory": {
        "pop": 10,
        "langchain": 35,
        "description": "Agent with persistent memory",
    },
}


def print_report(tasks: dict[str, dict], summary: dict) -> None:
    """Print a compelling, copy-pasteable DX benchmark report."""
    avg_pop = summary["avg_pop_lines"]
    avg_lc = summary["avg_langchain_lines"]
    reduction = summary["avg_reduction_pct"]

    print()
    print("=" * 70)
    print("pop vs LangChain: Developer Experience Benchmark")
    print("Lines of code for 8 common tasks (fewer = better)")
    print("=" * 70)
    print()
    print(f"  {'Task':<30} {'pop':>6} {'LangChain':>10} {'Saved':>8}")
    print("  " + "-" * 58)

    for task, data in tasks.items():
        pop_lines = data["pop"]
        lc_lines = data["langchain"]
        saved = lc_lines - pop_lines
        reduction_pct = f"-{(1 - pop_lines / lc_lines) * 100:.0f}%"
        print(f"  {task:<30} {pop_lines:>6} {lc_lines:>10} {reduction_pct:>8}  ({saved} lines saved)")

    print("  " + "-" * 58)
    print(f"  {'AVERAGE':<30} {avg_pop:>6.0f} {avg_lc:>10.0f} {'-' + str(reduction) + '%':>8}")
    print()
    print(f"  Result: pop uses {reduction}% fewer lines on average")
    print()
    print("## The headline")
    print()
    print(f"  pop:       ~{avg_pop:.0f} lines per task")
    print(f"  LangChain: ~{avg_lc:.0f} lines per task")
    print(f"  Savings:   {reduction}% less code to write, read, and maintain")
    print()
    print("## What this means in practice")
    print()
    print("  Hello world agent:  5 lines (pop) vs 35 lines (LangChain) — 86% less code")
    hello = tasks["hello_world_agent"]
    handoff = tasks["multi_agent_handoff"]
    print(f"  Multi-agent handoff: {handoff['pop']} lines (pop) vs {handoff['langchain']} lines (LangChain) — {round((1 - handoff['pop']/handoff['langchain'])*100)}% less code")
    print()
    print("  No StateGraph. No RunnableSequence. No ChannelWrite.")
    print("  Just @tool, Agent, and run().")
    print()


if __name__ == "__main__":
    total_pop = sum(d["pop"] for d in dx_comparison.values())
    total_lc = sum(d["langchain"] for d in dx_comparison.values())
    n = len(dx_comparison)

    summary = {
        "avg_pop_lines": total_pop / n,
        "avg_langchain_lines": total_lc / n,
        "avg_reduction_pct": round((1 - total_pop / total_lc) * 100),
    }

    print_report(dx_comparison, summary)

    results = {
        "tasks": dx_comparison,
        "summary": summary,
    }

    pathlib.Path("benchmarks/results").mkdir(parents=True, exist_ok=True)
    with open("benchmarks/results/dx_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to benchmarks/results/dx_comparison.json")
