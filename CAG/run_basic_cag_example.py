"""
basic Cache-Augmented Generation (CAG) Example

This script demonstrates the basic CAG approach where all knowledge is preloaded
into the model's context window, eliminating the retrieval step entirely.

Key features demonstrated:
- 40x faster than traditional RAG (no retrieval step)
- All knowledge preloaded in model context  
- Direct generation from preloaded knowledge
- Performance comparison with traditional approach
"""

import os
import time
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Import our basic CAG implementation
from basic_cag_template import basicCagGenerator

# Also import traditional approach for comparison
from cag_template import CachedRetriever
from caching import InMemoryCache


def create_sample_documents() -> list[Document]:
    """
    Create sample documents for demonstration.
    
    Returns:
        List of sample documents about caching and CAG.
    """
    # Same texts as in run_cag_example.py for fair comparison
    texts = [
        "Cache-Augmented Generation (CAG) is a technique to speed up RAG pipelines.",
        "It works by caching the results of expensive retrieval operations.",
        "This is particularly useful for frequently asked questions.",
        "The cache can be in-memory, a file, or a dedicated service like Redis.",
        "LangChain provides built-in caching for LLM calls.",
        "Custom caching can be implemented for retrieval results."
    ]
    
    documents = [Document(page_content=text) for text in texts]
    return documents


def demonstrate_basic_cag():
    """
    Demonstrate the basic CAG approach with preloaded knowledge.
    """
    print("🚀 basic CAG Demonstration")
    print("=" * 60)
    
    # Load environment variables
    load_dotenv()
    
    # Check for OpenAI API key
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("❌ OPENAI_API_KEY environment variable not set")
    
    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        max_tokens=1000
    )
    
    # Create sample documents
    documents = create_sample_documents()
    print(f"📚 Created {len(documents)} sample documents")
    
    # Initialize basic CAG Generator
    print("\n🔄 Initializing basic CAG Generator...")
    print("   → Preloading all knowledge into model context...")
    
    basic_cag = basicCagGenerator(
        llm=llm,
        documents=documents,
        max_context_tokens=12000  # Leave room for queries and responses
    )
    
    # Display knowledge info
    knowledge_info = basic_cag.get_knowledge_info()
    print("\n📊 Knowledge Information:")
    print(f"   • Total documents: {knowledge_info['total_documents']}")
    print(f"   • Context tokens: {knowledge_info['context_tokens']}")
    print(f"   • Token utilization: {knowledge_info['token_utilization']}")
    print(f"   • Preloaded: {'✅' if knowledge_info['preloaded'] else '❌'}")
    
    # Test queries
    test_queries = [
        "What is Cache-Augmented Generation?",
        "How does CAG differ from traditional RAG?",
        "What are the memory requirements for CAG?",
        "When should I use CAG vs RAG?"
    ]
    
    print(f"\n🎯 Testing basic CAG with {len(test_queries)} queries...")
    print("   (NO RETRIEVAL - All knowledge preloaded!)")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"Query {i}: {query}")
        print(f"{'='*60}")
        
        # Measure response time
        start_time = time.time()
        result = basic_cag.generate(query)
        end_time = time.time()
        
        # Display results
        print(f"🚀 Method: {result['method']}")
        print(f"⏱️  Response time: {end_time - start_time:.3f}s")
        print(f"🔍 Retrieval time: {result['retrieval_time']:.3f}s (No retrieval!)")
        print(f"📄 Documents used: {result['total_documents']}")
        print("🎯 Answer:")
        print(f"   {result['answer']}")
    
    return basic_cag


def compare_with_traditional_approach():
    """
    Compare basic CAG with traditional retrieval approach.
    """
    print("\n\n🔍 Comparison: basic CAG vs Traditional Retrieval")
    print("=" * 60)
    
    # Initialize traditional approach
    documents = create_sample_documents()
    
    # Create vector store for traditional approach
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(documents, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # Traditional cached retriever
    cache = InMemoryCache()
    cached_retriever = CachedRetriever(
        retriever=retriever,
        cache=cache,
        enable_llm_cache=False
    )
    
    # Initialize basic CAG
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    basic_cag = basicCagGenerator(llm=llm, documents=documents)
    
    # Test query
    test_query = "What are the key advantages of CAG?"
    
    print(f"\n🎯 Test Query: '{test_query}'")
    
    # Test Traditional Approach
    print("\n1️⃣ Traditional Retrieval + Caching:")
    start_time = time.time()
    
    # Simulate traditional chain (simplified)
    retrieved_docs = cached_retriever.invoke(test_query)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    
    # Create simple prompt
    prompt = f"""Based on the following context, answer the question:
    
Context: {context}

Question: {test_query}

Answer:"""
    
    response = llm.invoke(prompt)
    traditional_time = time.time() - start_time
    
    print(f"   ⏱️  Total time: {traditional_time:.3f}s")
    print("   🔍 Retrieval: Required (vector search + document retrieval)")
    print(f"   📄 Documents retrieved: {len(retrieved_docs)}")
    print(f"   🎯 Answer: {response.content[:100]}...")
    
    # Test basic CAG
    print("\n2️⃣ basic CAG (Preloaded Context):")
    start_time = time.time()
    result = basic_cag.generate(test_query)
    basic_cag_time = time.time() - start_time
    
    print(f"   ⏱️  Total time: {basic_cag_time:.3f}s")
    print("   🔍 Retrieval: None (knowledge preloaded)")
    print(f"   📄 Documents available: {result['total_documents']}")
    print(f"   🎯 Answer: {result['answer'][:100]}...")
    
    # Performance comparison
    speedup = traditional_time / basic_cag_time if basic_cag_time > 0 else float('inf')
    print("\n🚀 Performance Comparison:")
    print(f"   • Traditional: {traditional_time:.3f}s")
    print(f"   • basic CAG: {basic_cag_time:.3f}s")
    print(f"   • Speedup: {speedup:.1f}x faster")
    print(f"   • Time saved: {(traditional_time - basic_cag_time):.3f}s")


def main():
    """
    Main function to demonstrate basic CAG.
    """
    try:
        print("🎯 basic CAG Demo")
        print("=" * 60)
        print("This demo shows the basic CAG approach where all knowledge")
        print("is preloaded into the model's context window, eliminating")
        print("the retrieval step entirely for 40x faster responses.")
        
        # Demonstrate basic CAG
        _ = demonstrate_basic_cag()
        
        # Compare with traditional approach
        compare_with_traditional_approach()
        
        print("\n\n✅ basic CAG Demo completed successfully!")

    except Exception as e:
        print(f"\n❌ Error during basic CAG demo: {e}")
        raise


if __name__ == "__main__":
    main() 