"""Multi-agent orchestration patterns."""

from pop.multi.handoff import handoff
from pop.multi.patterns import (
    DebateResult,
    FanOutResult,
    PipelineResult,
    debate,
    fan_out,
    orchestrate,
    pipeline,
)

__all__ = [
    "DebateResult",
    "FanOutResult",
    "PipelineResult",
    "debate",
    "fan_out",
    "handoff",
    "orchestrate",
    "pipeline",
]
