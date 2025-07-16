from dotenv import load_dotenv
from typing import List, Dict, Any, Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

class CAGState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    cached_knowledge: Dict[str, Any]
    query: str
    response: str

def knowledge_preloader(state: CAGState, config: dict) -> dict:
    """Load knowledge base from a web source passed at runtime."""
    print("--- Executing Knowledge Preloader ---")
    if state.get("cached_knowledge"):
        print("Knowledge already preloaded. Skipping.")
        return {}

    web_paths = config.get("configurable", {}).get("web_paths", [])
    if not web_paths:
        raise ValueError("web_paths must be provided in the runtime config")

    print(f"Loading documents from web source: {web_paths}")
    loader = WebBaseLoader(web_paths=web_paths)
    documents = loader.load()
    
    processed_context = "\n\n".join([doc.page_content for doc in documents])
    knowledge_base = {"documents": documents, "processed_context": processed_context}
    print(f"Successfully loaded {len(documents)} documents.")
    return {"cached_knowledge": knowledge_base}

def cag_generator(state: CAGState, config: dict) -> dict:
    """Generate a response using a real LLM passed at runtime."""
    print("--- Executing CAG Generator ---")
    cached_knowledge = state["cached_knowledge"]
    query = state["messages"][-1].content
    
    llm = config.get("configurable", {}).get("llm")
    if not llm:
        raise ValueError("llm instance must be provided in the runtime config")

    print(f"Generating response for query: '{query}' with model: {llm.model_name}")

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are an expert AI assistant. All the necessary information has been preloaded into your context.
        Please answer the user's question based ONLY on the preloaded knowledge provided below.
        Do not mention that the information is preloaded. Just answer the question directly.

        === PRELOADED KNOWLEDGE ===
        {preloaded_knowledge}
        ========================="""),
        ("human", "{question}")
    ])

    chain = prompt_template | llm
    knowledge_str = cached_knowledge.get("processed_context", "")
    response_message = chain.invoke({"preloaded_knowledge": knowledge_str, "question": query})
    
    print("Successfully generated response.")
    return {"messages": [response_message], "response": response_message.content, "query": query}

def should_preload(state: CAGState) -> str:
    """Conditional edge to decide if preloading is needed."""
    if "cached_knowledge" in state and state["cached_knowledge"]:
        return "continue_to_generator"
    else:
        return "preload_knowledge"

def setup_cag_graph():
    """Set up a generic CAG workflow. Specifics are passed at runtime."""
    workflow = StateGraph(CAGState)

    workflow.add_node("knowledge_preloader", knowledge_preloader)
    workflow.add_node("cag_generator", cag_generator)

    workflow.add_conditional_edges(
        START,
        should_preload,
        {
            "preload_knowledge": "knowledge_preloader",
            "continue_to_generator": "cag_generator",
        },
    )
    workflow.add_edge("knowledge_preloader", "cag_generator")
    workflow.add_edge("cag_generator", END)

    checkpointer = InMemorySaver()
    return workflow.compile(checkpointer=checkpointer)
