"""The @tool decorator and JSON Schema compiler for pop agent tools.

Inspects function signatures, parses Google-style docstrings, and generates
JSON Schema from Python type hints. Returns a ToolDefinition wrapping the
original callable.
"""

from __future__ import annotations

import contextlib
import inspect
import re
from typing import TYPE_CHECKING, Any, Union, get_args, get_origin

from pop.types import ToolDefinition

if TYPE_CHECKING:
    from collections.abc import Callable

_PYTHON_TYPE_TO_JSON: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    dict: "object",
}


def _is_optional(annotation: Any) -> tuple[bool, Any]:
    """Check if a type annotation is Optional[X] and return (True, X) if so."""
    import types

    origin = get_origin(annotation)
    if origin is Union or isinstance(annotation, types.UnionType):
        args = get_args(annotation)
        non_none = tuple(a for a in args if a is not type(None))
        if len(non_none) == 1 and len(args) == 2:
            return True, non_none[0]
    return False, annotation


def _is_pydantic_model(annotation: Any) -> bool:
    """Check if annotation is a Pydantic BaseModel subclass."""
    try:
        from pydantic import BaseModel

        return isinstance(annotation, type) and issubclass(annotation, BaseModel)
    except ImportError:
        return False


def _type_to_json_schema(annotation: Any) -> dict[str, Any]:
    """Convert a Python type annotation to a JSON Schema dict."""
    is_opt, inner = _is_optional(annotation)
    if is_opt:
        return _type_to_json_schema(inner)

    if _is_pydantic_model(annotation):
        schema = dict(annotation.model_json_schema())
        # Remove the top-level title/description that pydantic adds
        return {k: v for k, v in schema.items() if k != "title"}

    if annotation in _PYTHON_TYPE_TO_JSON:
        return {"type": _PYTHON_TYPE_TO_JSON[annotation]}

    origin = get_origin(annotation)
    if origin is list:
        args = get_args(annotation)
        schema: dict[str, Any] = {"type": "array"}
        if args:
            schema = {**schema, "items": _type_to_json_schema(args[0])}
        return schema

    if origin is dict:
        return {"type": "object"}

    return {"type": "object"}


def _parse_docstring(docstring: str | None) -> tuple[str, dict[str, str]]:
    """Parse a Google-style docstring into summary and arg descriptions.

    Returns:
        A tuple of (summary, {param_name: description}).
    """
    if not docstring:
        return "", {}

    lines = docstring.strip().splitlines()
    summary_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("args:"):
            break
        if stripped == "" and summary_lines:
            break
        if stripped:
            summary_lines.append(stripped)

    summary = " ".join(summary_lines)

    arg_descriptions: dict[str, str] = {}
    arg_pattern = re.compile(r"^\s+(\w+)\s*(?:\(.*?\))?\s*:\s*(.+)$")
    in_args = False
    current_name: str | None = None

    _section_headers = frozenset(
        {"returns:", "raises:", "yields:", "examples:", "notes:", "references:", "attributes:"}
    )

    for line in lines:
        stripped = line.strip()
        if stripped.lower() == "args:":
            in_args = True
            continue
        if in_args:
            if stripped.lower() in _section_headers:
                break
            if stripped == "" or (not line.startswith(" ") and stripped != ""):
                if not line.startswith(" ") and stripped and stripped.lower() != "args:":
                    break
                continue
            match = arg_pattern.match(line)
            if match:
                current_name = match.group(1)
                arg_descriptions[current_name] = match.group(2).strip()
            elif current_name and line.startswith("            "):
                arg_descriptions[current_name] += " " + stripped

    return summary, arg_descriptions


def _build_parameters_schema(
    func: Callable[..., Any],
    arg_descriptions: dict[str, str],
) -> dict[str, Any]:
    """Build a JSON Schema parameters dict from function signature."""
    sig = inspect.signature(func)
    hints = {}
    with contextlib.suppress(Exception):
        hints = {
            k: v for k, v in inspect.get_annotations(func, eval_str=True).items() if k != "return"
        }

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue

        annotation = hints.get(param_name, str)
        is_opt, _ = _is_optional(annotation)
        has_default = param.default is not inspect.Parameter.empty

        prop = dict(_type_to_json_schema(annotation))
        if param_name in arg_descriptions:
            prop = {**prop, "description": arg_descriptions[param_name]}

        properties[param_name] = prop

        if not has_default and not is_opt:
            required.append(param_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
) -> ToolDefinition | Callable[..., ToolDefinition]:
    """Decorator that converts a function into a ToolDefinition.

    Supports both @tool and @tool(name="custom") forms.
    """

    def _wrap(fn: Callable[..., Any]) -> ToolDefinition:
        tool_name = name if name is not None else fn.__name__
        summary, arg_descriptions = _parse_docstring(fn.__doc__)
        description = summary or fn.__name__
        is_async = inspect.iscoroutinefunction(fn)
        parameters = _build_parameters_schema(fn, arg_descriptions)

        return ToolDefinition(
            name=tool_name,
            description=description,
            parameters=parameters,
            function=fn,
            is_async=is_async,
        )

    if func is not None:
        return _wrap(func)

    return _wrap
