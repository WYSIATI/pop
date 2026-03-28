"""Built-in tools for pop agents.

Optional batteries-included tools that reduce boilerplate for common tasks.
Install with: pip install pop-framework[tools]
"""

from __future__ import annotations

from pop.tools.calculator import Calculator
from pop.tools.read_url import ReadURL
from pop.tools.web_search import WebSearch

__all__ = ["Calculator", "ReadURL", "WebSearch"]
