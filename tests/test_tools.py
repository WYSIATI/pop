"""Tests for the pop.tools built-in tools module."""

from __future__ import annotations

import pytest

from pop.tools.calculator import Calculator
from pop.tools.read_url import ReadURL
from pop.tools.web_search import WebSearch
from pop.types import ToolDefinition

# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------


class TestCalculator:
    def test_returns_tool_definition(self) -> None:
        tool = Calculator()
        assert isinstance(tool, ToolDefinition)
        assert tool.name == "calculator"
        assert tool.is_async is False

    def test_basic_arithmetic(self) -> None:
        tool = Calculator()
        assert tool.function(expression="2 + 3") == "5"
        assert tool.function(expression="10 - 4") == "6"
        assert tool.function(expression="3 * 7") == "21"
        assert tool.function(expression="15 / 3") == "5"

    def test_complex_expressions(self) -> None:
        tool = Calculator()
        assert tool.function(expression="2 ** 10") == "1024"
        assert tool.function(expression="(10 + 5) * 2") == "30"
        assert tool.function(expression="17 % 5") == "2"
        assert tool.function(expression="17 // 5") == "3"

    def test_float_result(self) -> None:
        tool = Calculator()
        assert tool.function(expression="7 / 2") == "3.5"

    def test_negative_numbers(self) -> None:
        tool = Calculator()
        assert tool.function(expression="-5 + 3") == "-2"

    def test_invalid_expression_raises(self) -> None:
        tool = Calculator()
        with pytest.raises(ValueError, match=r"Invalid|Unsupported"):
            tool.function(expression="not math")

    def test_no_variable_access(self) -> None:
        tool = Calculator()
        with pytest.raises(ValueError, match="Unsupported"):
            tool.function(expression="x + 1")

    def test_no_function_calls(self) -> None:
        tool = Calculator()
        with pytest.raises(ValueError, match=r"Unsupported|Invalid"):
            tool.function(expression="__import__('os')")

    def test_parameters_schema(self) -> None:
        tool = Calculator()
        assert "expression" in tool.parameters["properties"]
        assert tool.parameters["required"] == ["expression"]


# ---------------------------------------------------------------------------
# ReadURL
# ---------------------------------------------------------------------------


class TestReadURL:
    def test_returns_tool_definition(self) -> None:
        tool = ReadURL()
        assert isinstance(tool, ToolDefinition)
        assert tool.name == "read_url"
        assert tool.is_async is False

    def test_custom_timeout(self) -> None:
        tool = ReadURL(timeout=30)
        assert isinstance(tool, ToolDefinition)

    def test_parameters_schema(self) -> None:
        tool = ReadURL()
        assert "url" in tool.parameters["properties"]
        assert tool.parameters["required"] == ["url"]


# ---------------------------------------------------------------------------
# WebSearch
# ---------------------------------------------------------------------------


class TestWebSearch:
    def test_returns_tool_definition(self) -> None:
        tool = WebSearch()
        assert isinstance(tool, ToolDefinition)
        assert tool.name == "web_search"
        assert tool.is_async is False

    def test_custom_max_results(self) -> None:
        tool = WebSearch(max_results=3)
        assert isinstance(tool, ToolDefinition)

    def test_parameters_schema(self) -> None:
        tool = WebSearch()
        assert "query" in tool.parameters["properties"]
        assert tool.parameters["required"] == ["query"]

    def test_import_error_without_duckduckgo(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """WebSearch raises ImportError with helpful message when deps missing."""
        import builtins

        original_import = builtins.__import__

        def mock_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
            if name == "duckduckgo_search":
                raise ImportError("No module named 'duckduckgo_search'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        tool = WebSearch()
        with pytest.raises(ImportError, match="pop-framework\\[tools\\]"):
            tool.function(query="test")


# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------


class TestToolsModule:
    def test_all_exports(self) -> None:
        from pop.tools import Calculator, ReadURL, WebSearch

        assert callable(Calculator)
        assert callable(ReadURL)
        assert callable(WebSearch)
