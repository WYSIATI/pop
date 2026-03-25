"""Tests for the @tool decorator and schema compiler."""

from __future__ import annotations

import asyncio

from pydantic import BaseModel

from pop.tool import tool
from pop.types import ToolDefinition


class SearchParams(BaseModel):
    """Search parameters."""

    query: str
    max_results: int = 10


def test_basic_tool_decorator() -> None:
    """@tool on a simple function returns a ToolDefinition."""

    @tool
    def greet(name: str) -> str:
        """Say hello to someone.

        Args:
            name: The person's name.
        """
        return f"Hello, {name}!"

    assert isinstance(greet, ToolDefinition)
    assert greet.name == "greet"
    assert greet.description == "Say hello to someone."
    assert greet.is_async is False


def test_schema_string_param() -> None:
    """str hint produces {"type": "string"}."""

    @tool
    def echo(text: str) -> str:
        """Echo text.

        Args:
            text: The text to echo.
        """
        return text

    props = echo.parameters["properties"]
    assert props["text"] == {"type": "string", "description": "The text to echo."}
    assert "text" in echo.parameters["required"]


def test_schema_int_param() -> None:
    """int hint produces {"type": "integer"}."""

    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers.

        Args:
            a: First number.
            b: Second number.
        """
        return a + b

    props = add.parameters["properties"]
    assert props["a"]["type"] == "integer"
    assert props["b"]["type"] == "integer"
    assert set(add.parameters["required"]) == {"a", "b"}


def test_schema_float_param() -> None:
    """float hint produces {"type": "number"}."""

    @tool
    def scale(factor: float) -> float:
        """Scale by factor.

        Args:
            factor: The scale factor.
        """
        return factor

    assert scale.parameters["properties"]["factor"]["type"] == "number"


def test_schema_bool_param() -> None:
    """bool hint produces {"type": "boolean"}."""

    @tool
    def toggle(enabled: bool) -> bool:
        """Toggle a flag.

        Args:
            enabled: Whether enabled.
        """
        return enabled

    assert toggle.parameters["properties"]["enabled"]["type"] == "boolean"


def test_schema_list_param() -> None:
    """list[str] produces {"type": "array", "items": {"type": "string"}}."""

    @tool
    def process(items: list[str]) -> int:
        """Process items.

        Args:
            items: List of items.
        """
        return len(items)

    prop = process.parameters["properties"]["items"]
    assert prop["type"] == "array"
    assert prop["items"] == {"type": "string"}


def test_schema_dict_param() -> None:
    """dict produces {"type": "object"}."""

    @tool
    def store(data: dict) -> str:
        """Store data.

        Args:
            data: The data to store.
        """
        return "ok"

    assert store.parameters["properties"]["data"]["type"] == "object"


def test_schema_optional_param() -> None:
    """Optional[X] is not required, and uses type of X."""

    @tool
    def search(query: str, limit: int | None = None) -> str:
        """Search for items.

        Args:
            query: Search query.
            limit: Max results.
        """
        return query

    assert "query" in search.parameters["required"]
    assert "limit" not in search.parameters["required"]
    assert search.parameters["properties"]["limit"]["type"] == "integer"


def test_default_values_not_required() -> None:
    """Params with default values should not be in required."""

    @tool
    def fetch(url: str, timeout: int = 30) -> str:
        """Fetch a URL.

        Args:
            url: The URL.
            timeout: Request timeout.
        """
        return url

    assert "url" in fetch.parameters["required"]
    assert "timeout" not in fetch.parameters["required"]


def test_docstring_parsing_extracts_descriptions() -> None:
    """Google-style docstring Args section provides param descriptions."""

    @tool
    def compute(x: int, y: int) -> int:
        """Compute something.

        Args:
            x: The x coordinate.
            y: The y coordinate.
        """
        return x + y

    props = compute.parameters["properties"]
    assert props["x"]["description"] == "The x coordinate."
    assert props["y"]["description"] == "The y coordinate."


def test_pydantic_basemodel_param() -> None:
    """Pydantic BaseModel subclass uses model_json_schema()."""

    @tool
    def search(params: SearchParams) -> str:
        """Search with params.

        Args:
            params: Search parameters.
        """
        return params.query

    prop = search.parameters["properties"]["params"]
    # Should contain the schema from the Pydantic model
    assert "properties" in prop
    assert "query" in prop["properties"]
    assert "max_results" in prop["properties"]


def test_tool_with_custom_name() -> None:
    """@tool(name="custom") sets a custom tool name."""

    @tool(name="my_custom_tool")
    def do_stuff(x: int) -> int:
        """Do stuff.

        Args:
            x: Input value.
        """
        return x

    assert isinstance(do_stuff, ToolDefinition)
    assert do_stuff.name == "my_custom_tool"


def test_async_function_detection() -> None:
    """Async functions set is_async=True."""

    @tool
    async def fetch_data(url: str) -> str:
        """Fetch data from URL.

        Args:
            url: The URL to fetch.
        """
        return url

    assert isinstance(fetch_data, ToolDefinition)
    assert fetch_data.is_async is True


def test_function_remains_callable() -> None:
    """Decorated function can still be called via .function attribute."""

    @tool
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers.

        Args:
            a: First number.
            b: Second number.
        """
        return a * b

    result = multiply.function(3, 4)
    assert result == 12


def test_async_function_remains_callable() -> None:
    """Decorated async function can still be called via .function attribute."""

    @tool
    async def async_add(a: int, b: int) -> int:
        """Add two numbers async.

        Args:
            a: First.
            b: Second.
        """
        return a + b

    result = asyncio.run(async_add.function(2, 3))
    assert result == 5


def test_no_docstring_uses_function_name() -> None:
    """Function with no docstring uses the function name as description."""

    @tool
    def simple_task(x: int) -> int:
        return x * 2

    assert simple_task.name == "simple_task"
    assert simple_task.description == "simple_task"


def test_no_args_in_docstring() -> None:
    """Function with docstring but no Args section still works."""

    @tool
    def hello(name: str) -> str:
        """Say hello."""
        return f"Hi {name}"

    assert hello.description == "Say hello."
    assert hello.parameters["properties"]["name"]["type"] == "string"
    # No description extracted since no Args section
    assert "description" not in hello.parameters["properties"]["name"]


def test_return_annotation_excluded_from_params() -> None:
    """Return type annotation should not appear in parameters."""

    @tool
    def identity(x: str) -> str:
        """Return input.

        Args:
            x: The input.
        """
        return x

    assert "return" not in identity.parameters.get("properties", {})


def test_parameters_schema_structure() -> None:
    """Parameters dict has correct JSON Schema structure."""

    @tool
    def example(a: str, b: int) -> str:
        """Example function.

        Args:
            a: A string.
            b: An integer.
        """
        return a

    assert example.parameters["type"] == "object"
    assert "properties" in example.parameters
    assert "required" in example.parameters


def test_list_without_type_args() -> None:
    """Bare list produces {"type": "object"} in Python 3.10 (no generic origin)."""
    # Bare `list` in Python 3.10: get_origin returns None, falls to default
    # In Python 3.12+: get_origin returns list. Test the actual function directly.

    from pop.tool import _type_to_json_schema

    schema = _type_to_json_schema(list)
    # In 3.10 bare list maps to "object" since get_origin is None
    # In 3.12+ it may be "array"
    assert schema["type"] in ("array", "object")


def test_list_with_generic_args() -> None:
    """list[int] produces {"type": "array", "items": {"type": "integer"}}."""
    from pop.tool import _type_to_json_schema

    schema = _type_to_json_schema(list[int])
    assert schema["type"] == "array"
    assert schema["items"] == {"type": "integer"}


def test_dict_generic_origin() -> None:
    """dict[str, int] produces {"type": "object"}."""
    from pop.tool import _type_to_json_schema

    schema = _type_to_json_schema(dict[str, int])
    assert schema["type"] == "object"


def test_unknown_type_defaults_to_object() -> None:
    """Unknown type annotations fall back to {"type": "object"}."""
    from pop.tool import _type_to_json_schema

    class CustomClass:
        pass

    schema = _type_to_json_schema(CustomClass)
    assert schema["type"] == "object"


def test_docstring_multiline_arg_description() -> None:
    """Multi-line arg descriptions in docstring are concatenated."""

    @tool
    def search(query: str) -> str:
        """Search for items.

        Args:
            query: The search query that can span
                multiple lines.
        """
        return query

    desc = search.parameters["properties"]["query"].get("description", "")
    assert "search query" in desc


def test_docstring_with_non_args_section_after_args() -> None:
    """Docstring parsing stops arg section at next non-indented section."""

    @tool
    def compute(x: int) -> int:
        """Compute something.

                Args:
                    x: The input.
        Returns:
                    The result.
        """
        return x

    props = compute.parameters["properties"]
    assert "x" in props
    assert props["x"].get("description") == "The input."


def test_function_with_no_type_hints() -> None:
    """Function with no type annotations defaults to string type."""

    @tool
    def legacy(value) -> str:
        """Legacy function.

        Args:
            value: Some value.
        """
        return str(value)

    # Without type hint, should default to str
    assert legacy.parameters["properties"]["value"]["type"] == "string"
