"""Base building blocks for multi-agent templates.

Google-style docstrings are used as requested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Tuple

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


class AgentNode(Protocol):
    """Callable node interface for agents.

    Each node receives a state mapping (usually `MessagesState`) and returns either:
    - a mapping of state updates, or
    - a `Command` to control graph execution (handoff, state update, etc.).

    Args:
        state: Current graph state

    Returns:
        A mapping to merge into state or a `Command` for routing/handoffs.
    """

    def __call__(self, state: Mapping[str, Any], /) -> Mapping[str, Any] | Command:
        ...


@dataclass
class GraphSpec:
    """Graph specification for reusable compilation.

    Attributes:
        name: Logical graph name.
        builder: `StateGraph` instance before `compile()`.
        entry: Entry node name; defaults to `START` edge.
    """

    name: str
    builder: Any
    entry: str | None = None

    def compile(self):
        """Compile and return a runnable graph.

        Returns:
            Compiled graph that supports `.invoke()`.
        """

        # Supports either a StateGraph (needs compile) or a prebuilt graph with .invoke
        if hasattr(self.builder, "compile"):
            return self.builder.compile()
        return self.builder


def make_simple_messages_graph(name: str, nodes: Dict[str, AgentNode], start: str) -> GraphSpec:
    """Create a simple `MessagesState` graph.

    Args:
        name: Graph name.
        nodes: Mapping of node name to callable node.
        start: Name of the start node.

    Returns:
        GraphSpec: Uncompiled graph spec.
    """

    builder = StateGraph(MessagesState)
    for node_name, node_fn in nodes.items():
        builder.add_node(node_name, node_fn)
    builder.add_edge(START, start)
    return GraphSpec(name=name, builder=builder, entry=start)


def default_input_messages(user_text: str) -> Dict[str, List]:
    """Create a default message list from raw user text.

    Args:
        user_text: Input content from a user.

    Returns:
        Dict with key `messages` to seed `MessagesState`.
    """

    return {"messages": [HumanMessage(content=user_text)]}


def append_ai_message(state: Mapping[str, Any], content: str, name: Optional[str] = None) -> Dict[str, List[AIMessage]]:
    """Append an `AIMessage` to `messages` list.

    Args:
        state: Current state containing `messages`.
        content: Message content.
        name: Optional agent name to tag.

    Returns:
        Update dict for graph state.
    """

    msg = AIMessage(content=content, name=name) if name else AIMessage(content=content)
    return {"messages": [msg]}


