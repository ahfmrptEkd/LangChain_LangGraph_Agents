"""Tool-calling supervisor template.

Supervisor is a ReAct-style agent that calls specialist tools.
"""

from __future__ import annotations

from typing import Any, Mapping

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent, InjectedState
from langchain_openai import ChatOpenAI

from .base import GraphSpec


model = ChatOpenAI()


@tool
def research_agent(query: str, state: InjectedState) -> str:
    """Research specialist as a tool.

    Args:
        query: Topic to research.
        state: Current graph state (injected).

    Returns:
        Research notes as text.
    """

    resp = model.invoke([
        {"role": "system", "content": "You are a research specialist."},
        {"role": "user", "content": f"Research: {query}"},
    ])
    return resp.content


@tool
def writing_agent(content: str, style: str = "professional", state: InjectedState | None = None) -> str:
    """Writing specialist as a tool.

    Args:
        content: Source content to transform into a draft.
        style: Writing style.
        state: Current graph state (injected).
    """

    resp = model.invoke([
        {"role": "system", "content": f"You write in {style} style."},
        {"role": "user", "content": f"Draft from: {content}"},
    ])
    return resp.content


@tool
def review_agent(content: str, criteria: str = "accuracy, clarity", state: InjectedState | None = None) -> str:
    """Review specialist as a tool.

    Args:
        content: Draft to review.
        criteria: Review criteria.
        state: Current graph state (injected).
    """

    resp = model.invoke([
        {"role": "system", "content": f"You review for {criteria}."},
        {"role": "user", "content": f"Review this: {content}"},
    ])
    return resp.content


def build_graph() -> GraphSpec:
    """Build a tool-calling supervisor graph using prebuilt ReAct agent."""

    tools = [research_agent, writing_agent, review_agent]
    supervisor = create_react_agent(model, tools)
    # Wrap in GraphSpec-like holder for API symmetry; it is already compiled
    return GraphSpec(name="supervisor_tool", builder=supervisor)


def run(text: str) -> Mapping[str, Any]:
    graph = build_graph().compile()
    return graph.invoke({"messages": [{"role": "user", "content": text}]})


