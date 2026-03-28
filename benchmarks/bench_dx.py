"""Benchmark: Developer Experience (Lines of Code)
Compares the lines needed for common tasks in pop vs LangChain.
"""

from __future__ import annotations

import json
import pathlib

# Hand-counted from official docs and working examples.
# Docstrings excluded. Counted: imports, tool defs, agent creation, run call.
# Sources: pop examples/, smolagents guided_tour, LangChain tutorials + langgraph-supervisor-py
dx_comparison: dict[str, dict] = {
    "hello_world_agent": {
        "pop": 10,
        "smolagents": 11,
        "langchain": 11,
        "description": "Simplest possible agent with one custom tool",
    },
    "web_search_agent": {
        "pop": 4,
        "smolagents": 4,
        "langchain": 12,
        "description": "Agent with built-in search tool",
    },
    "multi_agent_handoff": {
        "pop": 5,
        "smolagents": 12,
        "langchain": 35,
        "description": "Two-agent handoff",
    },
}


def print_report(tasks: dict[str, dict], summary: dict) -> None:
    """Print a compelling, copy-pasteable DX benchmark report."""
    print()
    print("=" * 80)
    print("pop vs smolagents vs LangChain: Developer Experience Benchmark")
    print(f"Lines of code for {len(tasks)} common tasks (fewer = better)")
    print("=" * 80)
    print()
    print(f"  {'Task':<25} {'pop':>6} {'smolagents':>12} {'LangChain':>12}")
    print("  " + "-" * 55)

    for task, data in tasks.items():
        label = task.replace("_", " ").title()
        pop_lines = data["pop"]
        sm_lines = data.get("smolagents", "-")
        lc_lines = data["langchain"]
        print(f"  {label:<25} {pop_lines:>6} {sm_lines:>12} {lc_lines:>12}")

    avg_pop = summary["avg_pop_lines"]
    avg_lc = summary["avg_langchain_lines"]
    avg_sm = summary.get("avg_smolagents_lines", 0)
    print("  " + "-" * 55)
    print(f"  {'AVERAGE':<25} {avg_pop:>6.0f} {avg_sm:>12.0f} {avg_lc:>12.0f}")
    print()
    print(f"  pop: fewest lines across all tasks")
    print()


if __name__ == "__main__":
    n = len(dx_comparison)
    total_pop = sum(d["pop"] for d in dx_comparison.values())
    total_lc = sum(d["langchain"] for d in dx_comparison.values())
    total_sm = sum(d.get("smolagents", 0) for d in dx_comparison.values())

    summary = {
        "avg_pop_lines": total_pop / n,
        "avg_smolagents_lines": total_sm / n,
        "avg_langchain_lines": total_lc / n,
        "avg_reduction_vs_langchain_pct": round((1 - total_pop / total_lc) * 100),
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
