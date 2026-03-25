"""Demo: Augmented LLM (Building Block)

Pattern from: https://www.anthropic.com/engineering/building-effective-agents

The most basic building block — an LLM enhanced with tools. The agent
reasons about the task, decides which tool to call, observes the result,
and continues until it has a final answer.

This demo builds a personal finance assistant with calculator and
expense-tracking tools.
"""

from datetime import datetime, timezone

from pop import Agent, tool

# ── Tools ──────────────────────────────────────────────────────────────

_expenses: list[dict] = []


@tool
def add_expense(category: str, amount: float, description: str) -> str:
    """Record a new expense.

    Args:
        category: Expense category (food, transport, entertainment, etc.).
        amount: Amount spent in dollars.
        description: Brief description of the expense.
    """
    entry = {
        "category": category,
        "amount": amount,
        "description": description,
        "date": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"),
    }
    _expenses.append(entry)
    return f"Recorded ${amount:.2f} for '{description}' under {category}."


@tool
def list_expenses(category: str = "") -> str:
    """List all recorded expenses, optionally filtered by category.

    Args:
        category: Optional category to filter by. Leave empty for all.
    """
    filtered = _expenses if not category else [e for e in _expenses if e["category"] == category]
    if not filtered:
        return "No expenses recorded yet."
    lines = [f"  - ${e['amount']:.2f} | {e['category']} | {e['description']}" for e in filtered]
    total = sum(e["amount"] for e in filtered)
    return "\n".join(lines) + f"\n  Total: ${total:.2f}"


@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression safely.

    Args:
        expression: A math expression like '100 + 200 * 0.15'.
    """
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return "Error: only numeric expressions are allowed."
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


# ── Agent ──────────────────────────────────────────────────────────────

agent = Agent(
    model="openai:gpt-4o",
    tools=[add_expense, list_expenses, calculate],
    instructions=(
        "You are a personal finance assistant. Help the user track expenses "
        "and do calculations. Always confirm after recording an expense."
    ),
    max_steps=10,
)

# ── Run ────────────────────────────────────────────────────────────────

task = (
    "I spent $12.50 on lunch, $45 on a taxi, and $8.99 on a coffee. "
    "Record all of these, then tell me my total spending today."
)

print(f"Task: {task}\n")
result = agent.run(task)

print(f"Agent: {result.output}\n")
print("--- Details ---")
print(f"Steps: {len(result.steps)}")
print(f"Tokens: {result.token_usage.total}")
print(f"Cost: ${result.cost:.6f}")

for step in result.steps:
    if step.tool_name:
        print(f"  [{step.index}] {step.tool_name}({step.tool_args})")
