"""Built-in safe calculator tool."""

from __future__ import annotations

import ast
import operator
from typing import Any

from pop.types import ToolDefinition

_SAFE_OPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> float:
    """Recursively evaluate an AST node using only safe math operations."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp):
        op_fn = _SAFE_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op_fn(_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op_fn = _SAFE_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op_fn(_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def Calculator() -> ToolDefinition:
    """Create a safe calculator tool that evaluates math expressions.

    Only supports basic arithmetic: +, -, *, /, //, %, **
    No variable access, function calls, or imports — fully sandboxed.
    """

    def _calculate(expression: str) -> str:
        """Evaluate a mathematical expression and return the result.

        Args:
            expression: A math expression like '2 + 3 * 4' or '(10 - 2) ** 3'.
        """
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"Invalid expression: {expression}") from exc
        result = _safe_eval(tree)
        # Return clean string: drop .0 for integers
        if result == int(result):
            return str(int(result))
        return str(result)

    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "A math expression to evaluate, e.g. '2 + 3 * 4'",
            },
        },
        "required": ["expression"],
    }

    return ToolDefinition(
        name="calculator",
        description="Evaluate a mathematical expression safely. Supports +, -, *, /, //, %, **.",
        parameters=parameters,
        function=_calculate,
        is_async=False,
    )
