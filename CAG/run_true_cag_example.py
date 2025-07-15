"""
True Cache-Augmented Generation (CAG) Example - Long Context Experiment

This script demonstrates the True CAG approach using a real-world long document
(~9,411 tokens) to test true CAG performance with substantial context.

Key features demonstrated:
- Long context preloading (9,411 tokens vs 225 tokens in basic example)
- Performance comparison with realistic document size
- Context utilization analysis
- Real-world CAG performance testing
"""

import os
import time
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
import bs4

# Import our basic CAG implementation
from true_cag_template import basicCagGenerator

# Also import traditional approach for comparison
from cag_template import CachedRetriever
from caching import InMemoryCache


def load_web_document() -> list[Document]:
    """
    Load the long context document from web for realistic CAG testing.
    
    This loads Lilian Weng's blog post about LLM-powered autonomous agents
    which contains about 9,411 tokens - a realistic test case for CAG.
    
    Returns:
        List containing the loaded web document.
    """
    print("🌐 Loading document from web...")
    
    # WebBaseLoader setup with specific parsing
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        )
    )
    
    # Load the document
    docs = loader.load()
    
    # Display document info
    print(f"📚 Loaded {len(docs)} document(s)")
    if docs:
        content_length = len(docs[0].page_content)
        print(f"📝 Content length: {content_length:,} characters")
    
    return docs


def get_long_context_documents() -> list[Document]:
    """
    Get the long context document for realistic CAG testing.
    
    
    Returns:
        List containing the long document for CAG testing.
    """
    print("📚 Loading long context document (~9,411 tokens)")
    
    # Load document from web
    docs = load_web_document()
    
    return docs


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


def demonstrate_long_context_cag():
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
    
    # Get long context documents
    documents = get_long_context_documents()
    
    # Initialize basic CAG Generator with higher token limit
    print("\n🔄 Initializing basic CAG Generator for long context...")
    print("   → Preloading long document into model context...")
    
    basic_cag = basicCagGenerator(
        llm=llm,
        documents=documents,
        max_context_tokens=15000
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
        "What are LLM-powered autonomous agents?",
        "How does planning work in autonomous agents?",
        "What are the key components of an agent system?",
        "What challenges do LLM agents face?"
    ]
    
    print(f"\n🎯 Testing Long Context CAG with {len(test_queries)} queries...")
    print("   (NO RETRIEVAL - Full document preloaded in context!)")
    
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
        print(f"🔢 Context tokens: {result['context_tokens']}")
        print("🎯 Answer:")
        print(f"   {result['answer']}")
    
    return basic_cag


def compare_short_vs_long_context():
    """
    Compare basic CAG performance with short vs long context.
    """
    print("\n\n🔍 Context Size Comparison: Short vs Long Context CAG")
    print("=" * 60)
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    # Test with short context (original sample documents)
    print("\n1️⃣ Short Context CAG (225 tokens):")
    short_documents = create_sample_documents()
    short_cag = basicCagGenerator(llm=llm, documents=short_documents)
    short_info = short_cag.get_knowledge_info()
    
    print(f"   📄 Documents: {short_info['total_documents']}")
    print(f"   🔢 Context tokens: {short_info['context_tokens']}")
    
    # Test with long context
    print("\n2️⃣ Long Context CAG (9,411 tokens):")
    long_documents = get_long_context_documents()
    long_cag = basicCagGenerator(llm=llm, documents=long_documents, max_context_tokens=15000)
    long_info = long_cag.get_knowledge_info()
    
    print(f"   📄 Documents: {long_info['total_documents']}")
    print(f"   🔢 Context tokens: {long_info['context_tokens']}")
    
    # Performance test with same query
    test_query = "What are the key advantages of this approach?"
    
    print(f"\n🎯 Performance Test: '{test_query}'")
    
    # Short context test
    print("\n📝 Short Context Response:")
    start_time = time.time()
    short_result = short_cag.generate(test_query)
    short_time = time.time() - start_time
    print(f"   ⏱️  Time: {short_time:.3f}s")
    print(f"   🎯 Answer: {short_result['answer'][:150]}...")
    
    # Long context test  
    print("\n📋 Long Context Response:")
    start_time = time.time()
    long_result = long_cag.generate(test_query)
    long_time = time.time() - start_time
    print(f"   ⏱️  Time: {long_time:.3f}s")
    print(f"   🎯 Answer: {long_result['answer'][:150]}...")
    
    # Context utilization analysis
    print("\n📊 Context Utilization Analysis:")
    print(f"   • Short context: {short_info['context_tokens']} tokens")
    print(f"   • Long context: {long_info['context_tokens']} tokens")
    print(f"   • Context ratio: {long_info['context_tokens'] / short_info['context_tokens']:.1f}x larger")
    print(f"   • Performance impact: {long_time / short_time:.1f}x slower" if short_time > 0 else "   • Performance impact: Unable to calculate")


def compare_with_traditional_approach():
    """
    Compare basic CAG with traditional retrieval approach using long context.
    """
    print("\n\n🔍 Comparison: Long Context CAG vs Traditional Retrieval")
    print("=" * 60)
    
    # Initialize with long context documents
    documents = get_long_context_documents()
    
    # Create vector store for traditional approach
    print("\n🔄 Setting up traditional retrieval approach...")
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
    basic_cag = basicCagGenerator(llm=llm, documents=documents, max_context_tokens=15000)
    
    # Test query
    test_query = "How do autonomous agents handle planning and reasoning?"
    
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
    print("\n2️⃣ Long Context CAG (Preloaded Context):")
    start_time = time.time()
    result = basic_cag.generate(test_query)
    basic_cag_time = time.time() - start_time
    
    print(f"   ⏱️  Total time: {basic_cag_time:.3f}s")
    print("   🔍 Retrieval: None (knowledge preloaded)")
    print(f"   📄 Documents available: {result['total_documents']}")
    print(f"   🔢 Context tokens: {result['context_tokens']}")
    print(f"   🎯 Answer: {result['answer'][:100]}...")
    
    # Performance comparison
    speedup = traditional_time / basic_cag_time if basic_cag_time > 0 else float('inf')
    print("\n🚀 Long Context Performance Comparison:")
    print(f"   • Traditional: {traditional_time:.3f}s")
    print(f"   • Long Context CAG: {basic_cag_time:.3f}s")
    print(f"   • Speedup: {speedup:.1f}x faster")
    print(f"   • Time saved: {(traditional_time - basic_cag_time):.3f}s")
    print(f"   • Context efficiency: Full document vs {len(retrieved_docs)} chunks")


def main():
    """
    Main function to demonstrate long context CAG.
    """
    try:
        print("🎯 Long Context CAG Demo")
        print("=" * 60)
        print("This demo shows the basic CAG approach with a real-world")
        print("long document (~9,411 tokens) to test performance with")
        print("substantial context preloading.")
        
        # Demonstrate long context CAG
        _ = demonstrate_long_context_cag()
        
        # Compare short vs long context
        compare_short_vs_long_context()
        
        # Compare with traditional approach
        compare_with_traditional_approach()
        

    except Exception as e:
        print(f"\n❌ Error during long context CAG demo: {e}")
        raise


if __name__ == "__main__":
    main() 