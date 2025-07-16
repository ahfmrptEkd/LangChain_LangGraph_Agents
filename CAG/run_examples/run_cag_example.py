"""Example of how to use the CachedRetriever with hybrid caching."""

import os
import sys
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Add parent directory to path to import CAG modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from caching import InMemoryCache
from cag_template import CachedRetriever


def main():
    """
    Main function to demonstrate Cache-Augmented Generation (CAG) with hybrid caching.
    
    This function showcases:
    1. Setting up a vector store with sample documents
    2. Creating a cached retriever with hybrid caching (retrieval + LLM)
    3. Building a RAG chain with the cached retriever
    4. Testing cache hits and misses with multiple queries
    5. Demonstrating cache management operations
    """
    
    # Load environment variables
    load_dotenv()

    # Ensure the OpenAI API key is set
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    # 1. Set up a simple vector store
    print("1. Setting up vector store...")
    texts = [
        "Cache-Augmented Generation (CAG) is a technique to speed up RAG pipelines.",
        "It works by caching the results of expensive retrieval operations.",
        "This is particularly useful for frequently asked questions.",
        "The cache can be in-memory, a file, or a dedicated service like Redis.",
        "LangChain provides built-in caching for LLM calls.",
        "Custom caching can be implemented for retrieval results."
    ]
    documents = [Document(page_content=t) for t in texts]

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(documents, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    # 2. Set up the cached retriever with hybrid caching
    print("\n2. Setting up hybrid caching (retrieval + LLM)...")
    cache = InMemoryCache()
    cached_retriever = CachedRetriever(
        retriever=retriever, 
        cache=cache,
        enable_llm_cache=True
    )

    # 3. Set up a simple RAG chain
    print("\n3. Setting up the RAG chain...")
    llm = ChatOpenAI(temperature=0)
    prompt = ChatPromptTemplate.from_template("""
                Answer the following question based only on the provided context:

                <context>
                {context}
                </context>

                Question: {input}
                """)

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(cached_retriever, document_chain)

    # 4. Run the chain multiple times to demonstrate caching
    print("\n4. Testing caching with multiple queries...")

    # First query
    query1 = "What is Cache-Augmented Generation?"
    print(f"\n--- First query: '{query1}' ---")
    response1 = retrieval_chain.invoke({"input": query1})
    print(f"Response: {response1['answer'][:100]}...")

    # Same query again (should hit retrieval cache)
    print(f"\n--- Same query again: '{query1}' ---")
    response2 = retrieval_chain.invoke({"input": query1})
    print(f"Response: {response2['answer'][:100]}...")

    # Different query
    query2 = "How does caching work in RAG?"
    print(f"\n--- Different query: '{query2}' ---")
    response3 = retrieval_chain.invoke({"input": query2})
    print(f"Response: {response3['answer'][:100]}...")

    # Same different query again
    print(f"\n--- Same different query again: '{query2}' ---")
    response4 = retrieval_chain.invoke({"input": query2})
    print(f"Response: {response4['answer'][:100]}...")

    # 5. Show cache information
    print("\n5. Cache information:")
    cache_info = cached_retriever.get_cache_info()
    print(f"Retrieval cache size: {cache_info['cache_size']}")
    print(f"Cached queries: {cache_info['cache_keys']}")

    # 6. Test cache clearing
    print("\n6. Testing cache clear...")
    cached_retriever.clear_cache()
    cache_info_after_clear = cached_retriever.get_cache_info()
    print(f"Cache size after clear: {cache_info_after_clear['cache_size']}")

    # 7. Query after cache clear (should be cache miss)
    print(f"\n--- Query after cache clear: '{query1}' ---")
    response5 = retrieval_chain.invoke({"input": query1})
    print(f"Response: {response5['answer'][:100]}...")

    print("\n✅ Demo completed! You can see the difference between cache hits and misses.")
    print("📊 Benefits:")
    print("  - Retrieval cache: Speeds up document search")
    print("  - LLM cache: Speeds up language model calls")
    print("  - Combined: Maximum efficiency for RAG pipelines")


if __name__ == "__main__":
    main()
