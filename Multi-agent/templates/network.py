"""Network architecture template for distributed multi-agent collaboration.

This module implements a network (many-to-many) pattern where agents can directly
communicate and route to each other without a central coordinator. Each agent
makes autonomous decisions about which agent to call next based on the current
conversation context.

The pattern is ideal for:
- Collaborative workflows where agents need peer-to-peer communication
- Scenarios requiring dynamic, context-driven routing
- Teams where agents have overlapping capabilities

Example:
    Basic usage::

        from templates import network
        
        # Build and compile the graph
        graph = network.build_graph().compile()
        
        # Run with user input
        result = graph.invoke({"messages": [{"role": "user", "content": "Research AI trends"}]})
        
        # Or use the convenience function
        result = network.run("Research AI trends")

Agents:
    researcher_agent: Handles research and data gathering tasks.
    writer_agent: Creates drafts and written content.
    reviewer_agent: Reviews and provides feedback on content.
"""

from __future__ import annotations

from typing import Any, Literal, Mapping

from langgraph.graph import MessagesState, END
from langgraph.types import Command
from langchain_openai import ChatOpenAI

from base import GraphSpec, make_simple_messages_graph, append_ai_message


model = ChatOpenAI()


def researcher_agent(state: MessagesState) -> Command:
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
        nxt = END

    return Command(goto=nxt, update=append_ai_message(state, content, name="researcher"))


def writer_agent(state: MessagesState) -> Command:
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
        nxt = END

    return Command(goto=nxt, update=append_ai_message(state, content, name="writer"))


def reviewer_agent(state: MessagesState) -> Command:
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
        nxt = END

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
    from langchain_core.messages import HumanMessage

    graph = build_graph().compile()
    return graph.invoke({"messages": [HumanMessage(content=text)]})


