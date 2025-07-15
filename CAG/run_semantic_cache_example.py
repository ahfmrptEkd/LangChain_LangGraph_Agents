"""Example of how to use the SemanticCachedRetriever."""

import os
import time
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from semantic_cache_template import SemanticCachedRetriever

def main():
    """
    Main function to demonstrate semantic caching.

    This function showcases:
    1. Setting up a vector store and a base retriever.
    2. Creating a SemanticCachedRetriever with cache size limit.
    3. Building a RAG chain.
    4. Testing semantic cache hits with slightly different but similar queries.
    5. Performance measurement and cache management.
    """
    load_dotenv()
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    # 1. Set up the base vector store and retriever
    print("1. Setting up base vector store...")
    texts = [
        "The Eiffel Tower, located in Paris, France, is a famous landmark.",
        "It was designed by Gustave Eiffel and completed in 1889.",
        "The tower is 330 meters tall and was the world's tallest man-made structure for 41 years.",
        "Millions of people visit the Eiffel Tower every year.",
        "The tower was built for the 1889 World's Fair in Paris.",
        "The Eiffel Tower is made of iron and weighs approximately 10,100 tons.",
        "It has three levels accessible to visitors with restaurants on two levels.",
        "The tower sways slightly in the wind and thermal expansion causes height changes."
    ]
    documents = [Document(page_content=t) for t in texts]
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(documents, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    # 2. Set up the SemanticCachedRetriever with cache size limit
    print("\n2. Setting up SemanticCachedRetriever...")
    semantic_retriever = SemanticCachedRetriever(
        retriever=retriever,
        embedding_model=embeddings,
        similarity_threshold=0.95,  # Lower threshold for more cache hits
        max_cache_size=3  # Small cache size to test LRU behavior
    )

    # 3. Set up the RAG chain
    print("\n3. Setting up the RAG chain...")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_template("""
                Answer the following question based only on the provided context:

                <context>
                {context}
                </context>

                Question: {input}
                """)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(semantic_retriever, document_chain)

    # 4. Test queries with performance measurement
    print("\n4. Testing semantic caching with performance measurement...")
    
    test_queries = [
        "How tall is the Eiffel Tower?",
        "What is the height of the Eiffel Tower?",  # Similar to query 1
        "Who designed the Eiffel Tower?",
        "Who was the architect of the Eiffel Tower?",  # Similar to query 3
        "When was the Eiffel Tower built?",
        "What year was the Eiffel Tower completed?",  # Similar to query 5
        "How much does the Eiffel Tower weigh?",  # New query to test cache limit
    ]
    
    results = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Query {i}: '{query}' ---")
        
        # Measure response time
        start_time = time.time()
        response = retrieval_chain.invoke({"input": query})
        end_time = time.time()
        
        response_time = end_time - start_time
        results.append({
            "query": query,
            "response_time": response_time,
            "answer": response['answer'][:100] + "..." if len(response['answer']) > 100 else response['answer']
        })
        
        print(f"⏱️  Response time: {response_time:.3f}s")
        print(f"🎯 Answer: {response['answer']}")
        
        # Show cache info after each query
        cache_info = semantic_retriever.get_cache_info()
        print(f"📊 Cache info: {cache_info['cache_utilization']}")

    # 5. Performance analysis
    print("\n5. Performance Analysis:")
    print("=" * 60)
    
    for i, result in enumerate(results, 1):
        print(f"Query {i}: {result['response_time']:.3f}s - {result['query'][:50]}...")
    
    # Calculate average response times for cache hits vs misses
    # (This is a simplified analysis - in practice, you'd track cache hits/misses)
    avg_response_time = sum(r['response_time'] for r in results) / len(results)
    print(f"\n📊 Average response time: {avg_response_time:.3f}s")
    
    # 6. Cache size limit demonstration
    print("\n6. Cache Size Limit Demonstration:")
    print("=" * 60)
    
    # Show final cache state
    final_cache_info = semantic_retriever.get_cache_info()
    print(f"Final cache state: {final_cache_info['cache_utilization']}")
    print(f"Cached queries: {len(final_cache_info['cached_queries'])}")
    
    for i, cached_query in enumerate(final_cache_info['cached_queries'], 1):
        print(f"  {i}. {cached_query}")
    
    # 7. Test cache clearing
    print("\n7. Testing cache clearing...")
    semantic_retriever.clear_cache()
    
    cleared_cache_info = semantic_retriever.get_cache_info()
    print(f"After clearing: {cleared_cache_info['cache_utilization']}")

    print("\n✅ Semantic caching demo completed!")
    print("\n🔍 Key observations:")
    print("- Similar queries should show cache hits with faster response times")
    print("- Cache size limit prevents unlimited memory growth")
    print("- LRU eviction removes oldest entries when cache is full")
    print("- Semantic similarity threshold determines cache hit sensitivity")

if __name__ == "__main__":
    main()
