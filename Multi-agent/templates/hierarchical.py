"""Hierarchical architecture template for nested team coordination.

This module implements a hierarchical pattern with multiple levels of supervision.
A top-level supervisor delegates work to specialized teams (subgraphs), where each
team has its own internal supervisor managing specialist agents. This creates a
tree-like organizational structure with clear chains of command.

The pattern is ideal for:
- Large-scale workflows requiring multiple specialized teams
- Complex projects with distinct phases handled by different teams
- Scenarios where teams need internal coordination but also inter-team collaboration
- Organizations with clear hierarchical structures and responsibilities

Example:
    Basic usage::

        from templates import hierarchical
        
        # Build and compile the graph
        graph = hierarchical.build_graph().compile()
        
        # Run with user input
        result = graph.invoke({"messages": [HumanMessage(content="Research AI papers and write technical docs")]})
        
        # Or use the convenience function
        result = hierarchical.run("Research AI papers and write technical docs")

Architecture:
    top_level_supervisor: Routes between research and content teams.
    
    Research Team:
        research_supervisor: Manages research specialists.
        data_researcher: Handles data and statistics research.
        academic_researcher: Handles literature and academic research.
    
    Content Team:
        content_supervisor: Manages content specialists.
        copywriter: Creates marketing and general content.
        technical_writer: Creates technical documentation and guides.

Note:
    Each team is a compiled subgraph that can be independently tested and reused.
"""

from __future__ import annotations

from typing import Any, Literal, Mapping

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
from langchain_openai import ChatOpenAI

from base import GraphSpec, append_ai_message


model = ChatOpenAI()


# === Research Team ===
def research_supervisor(state: MessagesState) -> Command:
    last = state["messages"][-1]
    text = (last.content or "").lower()
    if any(k in text for k in ["data", "stats", "table"]):
        return Command(goto="data_researcher")
    if any(k in text for k in ["paper", "study", "academic"]):
        return Command(goto="academic_researcher")
    return Command(goto=END)


def data_researcher(state: MessagesState) -> Command[Literal["research_supervisor"]]:
    last = state["messages"][-1]
    resp = model.invoke([
        {"role": "system", "content": "You are a data/statistics researcher."},
        {"role": "user", "content": last.content},
    ])
    return Command(goto="research_supervisor", update=append_ai_message(state, resp.content or "", name="data_researcher"))


def academic_researcher(state: MessagesState) -> Command[Literal["research_supervisor"]]:
    last = state["messages"][-1]
    resp = model.invoke([
        {"role": "system", "content": "You are a literature/academic researcher."},
        {"role": "user", "content": last.content},
    ])
    return Command(goto="research_supervisor", update=append_ai_message(state, resp.content or "", name="academic_researcher"))


def build_research_team():
    team = StateGraph(MessagesState)
    team.add_node("research_supervisor", research_supervisor)
    team.add_node("data_researcher", data_researcher)
    team.add_node("academic_researcher", academic_researcher)
    team.add_edge(START, "research_supervisor")
    return team.compile()


# === Content Team ===
def content_supervisor(state: MessagesState) -> Command:
    last = state["messages"][-1]
    text = (last.content or "").lower()
    if any(k in text for k in ["api", "code", "technical", "how-to"]):
        return Command(goto="technical_writer")
    return Command(goto="copywriter")


def copywriter(state: MessagesState) -> Command[Literal["content_supervisor"]]:
    last = state["messages"][-1]
    resp = model.invoke([
        {"role": "system", "content": "You are a marketing copywriter."},
        {"role": "user", "content": last.content},
    ])
    return Command(goto="content_supervisor", update=append_ai_message(state, resp.content or "", name="copywriter"))


def technical_writer(state: MessagesState) -> Command[Literal["content_supervisor"]]:
    last = state["messages"][-1]
    resp = model.invoke([
        {"role": "system", "content": "You are a technical writer."},
        {"role": "user", "content": last.content},
    ])
    return Command(goto="content_supervisor", update=append_ai_message(state, resp.content or "", name="technical_writer"))


def build_content_team():
    team = StateGraph(MessagesState)
    team.add_node("content_supervisor", content_supervisor)
    team.add_node("copywriter", copywriter)
    team.add_node("technical_writer", technical_writer)
    team.add_edge(START, "content_supervisor")
    return team.compile()


# === Top-level ===
def top_level_supervisor(state: MessagesState) -> Command:
    last = state["messages"][-1]
    text = (last.content or "").lower()
    if any(k in text for k in ["research", "find", "analyze", "data", "paper", "study"]):
        return Command(goto="research_team")
    if any(k in text for k in ["write", "draft", "content", "copy", "doc", "api"]):
        return Command(goto="content_team")
    return Command(goto=END)


def build_graph() -> GraphSpec:
    top = StateGraph(MessagesState)
    top.add_node("top_level_supervisor", top_level_supervisor)
    top.add_node("research_team", build_research_team())
    top.add_node("content_team", build_content_team())
    top.add_edge(START, "top_level_supervisor")
    top.add_edge("research_team", "top_level_supervisor")
    top.add_edge("content_team", "top_level_supervisor")
    return GraphSpec(name="hierarchical", builder=top, entry="top_level_supervisor")


def run(text: str) -> Mapping[str, Any]:
    graph = build_graph().compile()
    from langchain_core.messages import HumanMessage
    return graph.invoke({"messages": [HumanMessage(content=text)]})


