"""Generate SVG benchmark charts from JSON data.

Reads benchmarks/results/latest.json and dx_comparison.json,
outputs four SVG charts to assets/.

Usage:
    python benchmarks/generate_charts.py
"""

from __future__ import annotations

import json
import pathlib

ASSETS_DIR = pathlib.Path("assets")
RESULTS_DIR = pathlib.Path("benchmarks/results")

# Design tokens (matching logo.svg)
CORAL = "#FF6154"
SKY = "#38bdf8"
GRAY = "#cbd5e1"
NAVY = "#1a1a2e"
TEXT_GRAY = "#64748b"
LIGHT_GRAY = "#94a3b8"
BG = "#fafbfc"
BORDER = "#e2e8f0"
GRID = "#f1f5f9"
SKY_DARK = "#0284c7"

FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif"


def _svg_header(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">\n'
        f"  <style>text {{ font-family: {FONT}; }}</style>\n"
        f'  <rect width="{width}" height="{height}" rx="12" fill="{BG}" stroke="{BORDER}" stroke-width="1"/>\n'
    )


def _bar(x: int, y: int, w: int, h: int, color: str, rx: int = 2) -> str:
    w = max(w, 3)  # minimum visible width
    return f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{color}"/>\n'


def _text(
    x: int,
    y: int,
    content: str,
    *,
    size: int = 13,
    weight: int = 500,
    color: str = TEXT_GRAY,
    anchor: str = "",
) -> str:
    anchor_attr = f' text-anchor="{anchor}"' if anchor else ""
    return (
        f'  <text x="{x}" y="{y}" font-size="{size}" '
        f'font-weight="{weight}" fill="{color}"{anchor_attr}>{content}</text>\n'
    )


def _badge(x: int, y: int, label: str, color: str = CORAL) -> str:
    w = len(label) * 7 + 16
    return (
        f'  <rect x="{x}" y="{y}" width="{w}" height="18" rx="9" '
        f'fill="{color}" opacity="0.12"/>\n'
        + _text(x + 8, y + 13, label, size=11, weight=700, color=color)
    )


def generate_import_time(data: dict) -> str:
    """Generate import time comparison chart."""
    pop_ms = data["import_time"]["mean_ms"]
    smolagents_ms = 740.0  # measured
    langchain_ms = 1200.0

    max_ms = langchain_ms
    bar_max = 520
    bar_x = 180

    def bar_w(ms: float) -> int:
        return max(int(ms / max_ms * bar_max), 3)

    svg = _svg_header(760, 250)
    svg += _text(40, 40, "Import Time", size=18, weight=700, color=NAVY)
    svg += _text(160, 40, "cold start, 10 fresh subprocess runs", size=18, weight=400, color=LIGHT_GRAY)

    # pop
    svg += _text(40, 90, "pop", size=13, weight=700, color=CORAL)
    svg += _bar(bar_x, 76, bar_w(pop_ms), 24, CORAL)
    svg += _text(bar_x + bar_w(pop_ms) + 8, 93, f"{pop_ms:.2f} ms", weight=700, color=CORAL)

    # smolagents
    svg += _text(40, 140, "smolagents", size=13, weight=600, color=SKY_DARK)
    svg += _bar(bar_x, 126, bar_w(smolagents_ms), 24, SKY)
    svg += _text(bar_x + bar_w(smolagents_ms) + 8, 143, f"~{smolagents_ms:.0f} ms", weight=600, color=SKY_DARK)

    # LangChain
    svg += _text(40, 190, "LangChain", size=13, weight=600, color=TEXT_GRAY)
    svg += _bar(bar_x, 176, bar_w(langchain_ms), 24, GRAY)
    svg += _text(bar_x + bar_w(langchain_ms) + 8, 193, f"~{langchain_ms:.0f} ms", weight=600, color=TEXT_GRAY)

    svg += _text(40, 235, "Measured in fresh subprocesses. pop uses lazy imports — only 2 deps load at import time.", size=11, color=LIGHT_GRAY)

    svg += "</svg>\n"
    return svg


def generate_overhead(data: dict) -> str:
    """Generate framework overhead comparison chart."""
    pop_ms = data["framework_overhead"]["median_ms"]
    langchain_ms = 45.0

    max_ms = langchain_ms
    bar_max = 520
    bar_x = 180

    def bar_w(ms: float) -> int:
        return max(int(ms / max_ms * bar_max), 3)

    ratio = int(langchain_ms / pop_ms) if pop_ms > 0 else 0

    svg = _svg_header(760, 200)
    svg += _text(40, 40, "Framework Overhead", size=18, weight=700, color=NAVY)
    svg += _text(260, 40, "per step, mock LLM, 100 runs", size=18, weight=400, color=LIGHT_GRAY)

    # pop
    svg += _text(40, 90, "pop", size=13, weight=700, color=CORAL)
    svg += _bar(bar_x, 76, bar_w(pop_ms), 24, CORAL)
    svg += _text(bar_x + bar_w(pop_ms) + 8, 93, f"{pop_ms:.2f} ms", weight=700, color=CORAL)

    # LangChain
    svg += _text(40, 140, "LangChain", size=13, weight=600, color=TEXT_GRAY)
    svg += _bar(bar_x, 126, bar_w(langchain_ms), 24, GRAY)
    svg += _text(bar_x + bar_w(langchain_ms) + 8, 143, f"~{langchain_ms:.0f} ms", weight=600, color=TEXT_GRAY)

    svg += _badge(220, 130, f"~{ratio}x faster")

    svg += _text(40, 185, "LLM replaced with instant mock. Remaining time = message assembly, hooks, state updates.", size=11, color=LIGHT_GRAY)

    svg += "</svg>\n"
    return svg


def generate_deps() -> str:
    """Generate runtime dependencies comparison chart."""
    frameworks = [
        ("pop", 2, CORAL, "httpx, pydantic"),
        ("smolagents", 6, SKY, ""),
        ("LangChain", 20, GRAY, ""),
    ]
    max_deps = 20
    bar_max = 460
    bar_x = 200

    svg = _svg_header(760, 250)
    svg += _text(40, 40, "Framework Footprint", size=18, weight=700, color=NAVY)
    svg += _text(260, 40, "runtime dependencies", size=18, weight=400, color=LIGHT_GRAY)

    y = 74
    for name, deps, color, note in frameworks:
        text_color = CORAL if color == CORAL else (SKY_DARK if color == SKY else TEXT_GRAY)
        name_weight = 700 if color == CORAL else 600
        bar_w = max(int(deps / max_deps * bar_max), 30)

        svg += _text(40, y + 18, name, size=14, weight=name_weight, color=text_color)
        svg += _bar(bar_x, y, bar_w, 30, color, rx=6)

        label = f"{deps}" if deps < 20 else "20+"
        svg += _text(bar_x + bar_w + 8, y + 20, label, weight=600, color=text_color)
        if note:
            svg += _text(bar_x + bar_w + 30, y + 20, note, size=11, weight=400, color=LIGHT_GRAY)

        y += 55

    svg += _text(40, y + 10, "Direct runtime dependencies from each framework's pyproject.toml", size=11, color=LIGHT_GRAY)

    svg += "</svg>\n"
    return svg


def generate_dx(dx_data: dict) -> str:
    """Generate developer experience (LOC) comparison chart."""
    tasks = [
        ("Hello World", "agent + one tool", {"pop": 10, "smolagents": 11, "langchain": 11}),
        ("Web Search", "with built-in tools", {"pop": 4, "smolagents": 4, "langchain": 12}),
        ("Multi-Agent", "two-agent handoff", {"pop": 5, "smolagents": 12, "langchain": 35}),
    ]

    max_loc = 35
    bar_max = 440
    bar_x = 200

    def bar_w(loc: int) -> int:
        return max(int(loc / max_loc * bar_max), 8)

    svg = _svg_header(760, 350)
    svg += _text(40, 40, "Developer Experience", size=18, weight=700, color=NAVY)
    svg += _text(258, 40, "lines of code per task (fewer = better)", size=18, weight=400, color=LIGHT_GRAY)

    # Legend
    items = [("pop", CORAL, 40), ("smolagents", SKY, 100), ("LangChain", GRAY, 210)]
    for label, color, lx in items:
        svg += f'  <rect x="{lx}" y="62" width="12" height="12" rx="3" fill="{color}"/>\n'
        text_color = CORAL if color == CORAL else TEXT_GRAY
        text_weight = 600 if color == CORAL else 500
        svg += _text(lx + 18, 73, label, size=11, weight=text_weight, color=text_color)

    # Grid lines
    for loc_val in range(0, 40, 10):
        gx = bar_x + int(loc_val / max_loc * bar_max)
        svg += f'  <line x1="{gx}" y1="95" x2="{gx}" y2="300" stroke="{GRID}" stroke-width="1"/>\n'
        svg += _text(gx, 315, str(loc_val), size=9, color=LIGHT_GRAY, anchor="middle")

    y_start = 110
    for task_name, subtitle, locs in tasks:
        svg += _text(40, y_start + 10, task_name, weight=600, color=NAVY)
        svg += _text(40, y_start + 23, subtitle, size=10, weight=400, color=LIGHT_GRAY)

        bar_y = y_start - 4
        # pop
        w = bar_w(locs["pop"])
        svg += _bar(bar_x, bar_y, w, 13, CORAL, rx=4)
        svg += _text(bar_x + w + 6, bar_y + 12, str(locs["pop"]), size=10, weight=700, color=CORAL)

        # smolagents
        bar_y += 16
        w = bar_w(locs["smolagents"])
        svg += _bar(bar_x, bar_y, w, 13, SKY, rx=4)
        svg += _text(bar_x + w + 6, bar_y + 12, str(locs["smolagents"]), size=10, weight=600, color=SKY_DARK)

        # LangChain
        bar_y += 16
        w = bar_w(locs["langchain"])
        svg += _bar(bar_x, bar_y, w, 13, GRAY, rx=4)
        svg += _text(bar_x + w + 6, bar_y + 12, str(locs["langchain"]), size=10, weight=500, color=LIGHT_GRAY)

        y_start += 65

    svg += _text(40, 335, "pop uses pop.tools.WebSearch() built-in + Agent(workers=[...]) shorthand", size=10, color=LIGHT_GRAY)

    svg += "</svg>\n"
    return svg


def main() -> None:
    ASSETS_DIR.mkdir(exist_ok=True)

    # Load benchmark data
    with open(RESULTS_DIR / "latest.json") as f:
        data = json.load(f)

    with open(RESULTS_DIR / "dx_comparison.json") as f:
        dx_data = json.load(f)

    charts = {
        "bench-import-time.svg": generate_import_time(data),
        "bench-overhead.svg": generate_overhead(data),
        "bench-deps.svg": generate_deps(),
        "bench-dx.svg": generate_dx(dx_data),
    }

    for filename, svg_content in charts.items():
        path = ASSETS_DIR / filename
        path.write_text(svg_content, encoding="utf-8")
        print(f"Generated {path}")


if __name__ == "__main__":
    main()
