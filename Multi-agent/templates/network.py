"""Network architecture template (many-to-many agents).

Provides a minimal reusable scaffold with three agents: researcher, writer, reviewer.
Replace model calls with your own LLM/tool logic.
"""

from __future__ import annotations

from typing import Any, Literal, Mapping

from langgraph.graph import MessagesState
from langgraph.types import Command
from langchain_openai import ChatOpenAI

from .base import GraphSpec, make_simple_messages_graph, append_ai_message


model = ChatOpenAI()


def researcher_agent(state: MessagesState) -> Command[Literal["writer_agent", "reviewer_agent", "__end__"]]:
    """Research-focused agent.

    Args:
        state: Messages-based state.

    Returns:
        Command to route to next agent and update messages.
    """

    last = state["messages"][-1]
    response = model.invoke([
        {"role": "system", "content": "You are a research specialist."},
        {"role": "user", "content": f"Research the topic: {last.content}"},
    ])

    content = response.content or ""
    if "write" in content.lower():
        nxt = "writer_agent"
    elif "review" in content.lower():
        nxt = "reviewer_agent"
    else:
        nxt = "__end__"

    return Command(goto=nxt, update=append_ai_message(state, content, name="researcher"))


def writer_agent(state: MessagesState) -> Command[Literal["researcher_agent", "reviewer_agent", "__end__"]]:
    """Writing-focused agent.

    Args:
        state: Messages-based state.

    Returns:
        Command to route to next agent and update messages.
    """

    last = state["messages"][-1]
    response = model.invoke([
        {"role": "system", "content": "You are a professional writer."},
        {"role": "user", "content": f"Draft based on: {last.content}"},
    ])

    content = response.content or ""
    if "more research" in content.lower():
        nxt = "researcher_agent"
    elif "review" in content.lower():
        nxt = "reviewer_agent"
    else:
        nxt = "__end__"

    return Command(goto=nxt, update=append_ai_message(state, content, name="writer"))


def reviewer_agent(state: MessagesState) -> Command[Literal["researcher_agent", "writer_agent", "__end__"]]:
    """Review-focused agent.

    Args:
        state: Messages-based state.

    Returns:
        Command to route to next agent and update messages.
    """

    last = state["messages"][-1]
    response = model.invoke([
        {"role": "system", "content": "You are a critical reviewer."},
        {"role": "user", "content": f"Review the following: {last.content}"},
    ])

    content = response.content or ""
    if "revise" in content.lower():
        nxt = "writer_agent"
    elif "needs data" in content.lower():
        nxt = "researcher_agent"
    else:
        nxt = "__end__"

    return Command(goto=nxt, update=append_ai_message(state, content, name="reviewer"))


def build_graph() -> GraphSpec:
    """Build a reusable network-style graph.

    Returns:
        GraphSpec for compilation and invocation.
    """

    spec = make_simple_messages_graph(
        name="network",
        nodes={
            "researcher_agent": researcher_agent,
            "writer_agent": writer_agent,
            "reviewer_agent": reviewer_agent,
        },
        start="researcher_agent",
    )
    return spec


def run(text: str) -> Mapping[str, Any]:
    """Quick helper to run the compiled network graph with user input.

    Args:
        text: User prompt.

    Returns:
        Final state after invocation.
    """

    graph = build_graph().compile()
    return graph.invoke({"messages": [{"role": "user", "content": text}]})


