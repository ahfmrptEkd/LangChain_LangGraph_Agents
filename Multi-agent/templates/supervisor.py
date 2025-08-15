"""Supervisor architecture template (central router).

Supervisor decides which specialist to call next based on the latest message.
"""

from __future__ import annotations

from typing import Any, Literal, Mapping

from langgraph.types import Command
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_openai import ChatOpenAI

from .base import GraphSpec, append_ai_message


model = ChatOpenAI()


def supervisor(state: MessagesState) -> Command[Literal["research_agent", "writing_agent", "review_agent", "__end__"]]:
    """Central decision-maker.

    Args:
        state: Messages-based state.

    Returns:
        Command to route to the selected specialist.
    """

    last = state["messages"][-1]
    response = model.invoke([
        {"role": "system", "content": "You are a routing supervisor."},
        {"role": "user", "content": f"Decide next agent for: {last.content}"},
    ])

    content = response.content or ""
    if any(k in content.lower() for k in ["research", "data", "find"]):
        nxt = "research_agent"
    elif any(k in content.lower() for k in ["write", "draft"]):
        nxt = "writing_agent"
    elif any(k in content.lower() for k in ["review", "check"]):
        nxt = "review_agent"
    else:
        nxt = "__end__"

    return Command(goto=nxt)


def research_agent(state: MessagesState) -> Command[Literal["supervisor"]]:
    last = state["messages"][-1]
    response = model.invoke([
        {"role": "system", "content": "You are a research specialist."},
        {"role": "user", "content": f"Research: {last.content}"},
    ])
    return Command(goto="supervisor", update=append_ai_message(state, response.content or "", name="researcher"))


def writing_agent(state: MessagesState) -> Command[Literal["supervisor"]]:
    last = state["messages"][-1]
    response = model.invoke([
        {"role": "system", "content": "You are a writing specialist."},
        {"role": "user", "content": f"Write from: {last.content}"},
    ])
    return Command(goto="supervisor", update=append_ai_message(state, response.content or "", name="writer"))


def review_agent(state: MessagesState) -> Command[Literal["supervisor"]]:
    last = state["messages"][-1]
    response = model.invoke([
        {"role": "system", "content": "You are a review specialist."},
        {"role": "user", "content": f"Review: {last.content}"},
    ])
    return Command(goto="supervisor", update=append_ai_message(state, response.content or "", name="reviewer"))


def build_graph() -> GraphSpec:
    """Build a supervisor-style graph.

    Returns:
        GraphSpec for compilation and invocation.
    """

    builder = StateGraph(MessagesState)
    builder.add_node("supervisor", supervisor)
    builder.add_node("research_agent", research_agent)
    builder.add_node("writing_agent", writing_agent)
    builder.add_node("review_agent", review_agent)
    builder.add_edge(START, "supervisor")
    builder.add_edge("research_agent", "supervisor")
    builder.add_edge("writing_agent", "supervisor")
    builder.add_edge("review_agent", "supervisor")
    return GraphSpec(name="supervisor", builder=builder, entry="supervisor")


def run(text: str) -> Mapping[str, Any]:
    graph = build_graph().compile()
    return graph.invoke({"messages": [{"role": "user", "content": text}]})


