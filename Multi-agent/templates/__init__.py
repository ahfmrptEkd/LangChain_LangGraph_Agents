"""Reusable multi-agent templates.

This package provides minimal, production-ready scaffolds for common
LangGraph multi-agent patterns: Network, Supervisor, Tool-calling Supervisor,
and Hierarchical architectures.

All modules expose a `build_graph()` function that returns a compiled graph
ready to `invoke()` and a `run(text: str)` helper for quick trials.
"""

from . import base  # noqa: F401
from . import network  # noqa: F401
from . import supervisor  # noqa: F401
from . import supervisor_tool  # noqa: F401
from . import hierarchical  # noqa: F401

__all__ = [
    "base",
    "network",
    "supervisor",
    "supervisor_tool",
    "hierarchical",
]


