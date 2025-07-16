"""
Example of how to use the LangGraph-based CAG workflow with RunnableConfig.
"""

import asyncio
import sys
import os
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# Add the parent directory to the system path to allow imports from the CAG module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph_template import setup_cag_graph

async def main():
    """Main function to run the LangGraph CAG demo with RunnableConfig."""
    print("🚀 Starting LangGraph CAG Demo with RunnableConfig")
    
    # 1. Set up the generic graph
    cag_graph = setup_cag_graph()
    
    # 2. Define the specifics for this run
    llm_instance = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    source_web_paths = ["https://lilianweng.github.io/posts/2023-06-23-agent/"]

    # 3. Create the runtime configuration object
    config = {
        "configurable": {
            "thread_id": "cag-thread-runnable-config-1",
            "llm": llm_instance,
            "web_paths": source_web_paths
        }
    }

    # 4. Define queries
    query1 = "What are the key components of an LLM-powered autonomous agent system?"
    query2 = "How does planning work for these agents?"

    # --- First Invocation ---
    print(f"\n--- 1. First Query ---")
    print(f"🗣️ User: {query1}")
    print("🤖 Assistant:")
    
    # 비동기 스트리밍 대신 한 번에 응답을 받아 출력
    result = await cag_graph.ainvoke(
        {"messages": [HumanMessage(content=query1)]},
        config=config
    )
    if "messages" in result:
        for message in result["messages"]:
            if hasattr(message, 'content') and getattr(message, 'type', None) == "ai":
                print(message.content)
    print("\n")

    # --- Second Invocation ---
    print(f"\n--- 2. Second Query (in the same conversation) ---")
    print(f"🗣️ User: {query2}")
    print("🤖 Assistant:")

    result2 = await cag_graph.ainvoke(
        {"messages": [HumanMessage(content=query2)]},
        config=config
    )
    if "messages" in result2:
        for message in result2["messages"]:
            if hasattr(message, 'content') and getattr(message, 'type', None) == "ai":
                print(message.content)
    print("\n\n✅ Demo finished. The graph was configured at runtime and state was persisted.")

if __name__ == "__main__":
    asyncio.run(main())
