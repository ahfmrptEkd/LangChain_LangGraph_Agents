"""Tool-calling supervisor template using ReAct pattern.

This module implements a tool-calling supervisor that uses LangGraph's prebuilt
ReAct agent to coordinate specialist agents as tools. Unlike traditional routing,
the supervisor can call multiple tools in sequence, pass parameters, and reason
about tool outputs before making the next decision.

The pattern is ideal for:
- Complex reasoning workflows requiring tool chaining
- Scenarios where the supervisor needs to pass specific parameters to agents
- Dynamic workflows where tool selection depends on previous outputs
- ReAct-style reasoning (Reason + Act) with specialist capabilities

Example:
    Basic usage::

        from templates import supervisor_tool
        
        # Build and compile the graph (uses prebuilt ReAct agent)
        graph = supervisor_tool.build_graph().compile()
        
        # Run with user input
        result = graph.invoke({"messages": [{"role": "user", "content": "Research AI trends and write a summary"}]})
        
        # Or use the convenience function
        result = supervisor_tool.run("Research AI trends and write a summary")

Tools:
    research_agent: Research specialist tool with query parameter.
    writing_agent: Writing specialist tool with content and style parameters.
    review_agent: Review specialist tool with content and criteria parameters.

Note:
    Uses LangGraph's create_react_agent for built-in reasoning and tool execution.
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
    """Quick helper to run the compiled tool-calling supervisor graph with user input.

    Args:
        text: User prompt.

    Returns:
        Final state after invocation.
    """
    from langchain_core.messages import HumanMessage

    graph = build_graph().compile()
    return graph.invoke({"messages": [HumanMessage(content=text)]})


