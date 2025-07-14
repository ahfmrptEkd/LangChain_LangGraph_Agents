"""
Performance comparison between different caching strategies in RAG pipelines.

This script demonstrates the performance benefits of different caching approaches:
1. Regular RAG (no caching)
2. LLM Cache only (language model response caching)
3. Retrieval Cache only (document search result caching)
4. Full CAG (both retrieval and LLM caching)

The comparison shows the individual and combined impact of:
- Retrieval caching (document search results)
- LLM caching (language model responses)
"""

import os
import time
import statistics
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache

from caching import InMemoryCache as CustomInMemoryCache
from cag_template import CachedRetriever


def setup_vector_store() -> FAISS:
    """Set up a vector store with sample documents."""
    print("📚 Setting up vector store...")
    
    # Sample documents about Cache-Augmented Generation
    texts = [
        "Cache-Augmented Generation (CAG) is a technique that improves RAG performance by caching retrieval results and LLM responses to avoid redundant operations.",
        "Retrieval caching stores document search results to avoid repeated vector similarity searches for identical queries.",
        "LLM caching stores language model responses to avoid repeated expensive API calls for identical prompts.",
        "Hybrid caching combines both retrieval caching and LLM caching for maximum performance optimization.",
        "Cache hit occurs when a query or prompt is found in the cache, allowing instant response without computation.",
        "Cache miss occurs when a query or prompt is not in the cache, requiring actual computation and subsequent caching.",
        "In-memory caching provides fast access but is limited by memory size and process lifetime.",
        "Persistent caching using Redis or databases allows cache sharing across processes and sessions.",
    ]
    documents = [Document(page_content=text) for text in texts]
    
    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(documents, embeddings)
    
    print(f"✅ Vector store created with {len(documents)} documents")
    return vector_store


def create_prompt_template() -> ChatPromptTemplate:
    """Create a prompt template for the RAG chain."""
    return ChatPromptTemplate.from_template("""
    You are a helpful assistant that answers questions based on the provided context.
    
    Context: {context}
    
    Question: {input}
    
    Please provide a comprehensive answer based on the context above.
    """)


def setup_chains(vector_store: FAISS) -> Dict[str, Any]:
    """Set up different chain configurations for comparison."""
    print("🔧 Setting up different chain configurations...")
    
    # Base components
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    prompt = create_prompt_template()
    
    # Create document chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    chains = {}
    
    # 1. Regular RAG (no caching)
    print("   1️⃣ Regular RAG (no caching)")
    set_llm_cache(None)  # Disable LLM cache
    regular_retrieval_chain = create_retrieval_chain(base_retriever, document_chain)
    chains["regular"] = {
        "chain": regular_retrieval_chain,
        "retriever": base_retriever,
        "description": "Regular RAG (no caching)"
    }
    
    # 2. LLM Cache only
    print("   2️⃣ LLM Cache only")
    set_llm_cache(InMemoryCache())  # Enable LLM cache
    llm_cached_document_chain = create_stuff_documents_chain(llm, prompt)
    llm_cached_retrieval_chain = create_retrieval_chain(base_retriever, llm_cached_document_chain)
    chains["llm_only"] = {
        "chain": llm_cached_retrieval_chain,
        "retriever": base_retriever,
        "description": "LLM Cache only"
    }
    
    # 3. Retrieval Cache only
    print("   3️⃣ Retrieval Cache only")
    set_llm_cache(None)  # Disable LLM cache
    retrieval_cache = CustomInMemoryCache()
    cached_retriever_only = CachedRetriever(
        retriever=base_retriever,
        cache=retrieval_cache,
        enable_llm_cache=False
    )
    retrieval_cached_document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_cached_chain = create_retrieval_chain(cached_retriever_only, retrieval_cached_document_chain)
    chains["retrieval_only"] = {
        "chain": retrieval_cached_chain,
        "retriever": cached_retriever_only,
        "description": "Retrieval Cache only",
        "cache": retrieval_cache
    }
    
    # 4. Full CAG (both caches)
    print("   4️⃣ Full CAG (both caches)")
    set_llm_cache(InMemoryCache())  # Enable LLM cache
    full_cache = CustomInMemoryCache()
    full_cached_retriever = CachedRetriever(
        retriever=base_retriever,
        cache=full_cache,
        enable_llm_cache=True
    )
    full_cached_document_chain = create_stuff_documents_chain(llm, prompt)
    full_cached_chain = create_retrieval_chain(full_cached_retriever, full_cached_document_chain)
    chains["full_cag"] = {
        "chain": full_cached_chain,
        "retriever": full_cached_retriever,
        "description": "Full CAG (both caches)",
        "cache": full_cache
    }
    
    print("✅ All chain configurations ready")
    return chains


def run_performance_test(chains: Dict[str, Any], test_queries: List[str], runs_per_query: int = 3) -> Dict[str, Dict[str, List[float]]]:
    """Run performance tests for all chain configurations."""
    print("\n🏃 Running performance tests with cache persistence...")
    
    # Separate results for cache miss and cache hit
    results = {
        name: {
            "cache_miss": [],
            "cache_hit": [],
            "all_times": []
        } for name in chains.keys()
    }
    
    # Test each chain configuration
    for chain_name, chain_config in chains.items():
        print(f"\n🔄 Testing {chain_config['description']}...")
        
        # Clear cache once at the beginning for each chain
        if "cache" in chain_config:
            chain_config["cache"].clear()
        
        # Process each unique query
        unique_queries = list(set(test_queries))  # Remove duplicates
        
        for query in unique_queries:
            print(f"   📝 Query: '{query}'")
            
            # Run the query multiple times to see cache effect
            query_times = []
            for run in range(runs_per_query):
                start_time = time.time()
                
                try:
                    _ = chain_config["chain"].invoke({"input": query})
                    end_time = time.time()
                    
                    execution_time = end_time - start_time
                    query_times.append(execution_time)
                    
                    # Categorize as cache miss or hit
                    if run == 0:
                        results[chain_name]["cache_miss"].append(execution_time)
                        print(f"      ⏱️  Run {run + 1}: {execution_time:.3f}s (cache miss)")
                    else:
                        results[chain_name]["cache_hit"].append(execution_time)
                        print(f"      ⏱️  Run {run + 1}: {execution_time:.3f}s (cache hit)")
                    
                    results[chain_name]["all_times"].append(execution_time)
                        
                except Exception as e:
                    print(f"      ❌ Error in run {run + 1}: {e}")
                    continue
            
            # Show cache effect for this query
            if len(query_times) >= 2:
                cache_miss_time = query_times[0]
                cache_hit_avg = statistics.mean(query_times[1:])
                cache_improvement = ((cache_miss_time - cache_hit_avg) / cache_miss_time) * 100
                
                print(f"      📊 Cache miss: {cache_miss_time:.3f}s")
                print(f"      📊 Cache hit avg: {cache_hit_avg:.3f}s")
                print(f"      🚀 Cache improvement: {cache_improvement:.1f}%")
    
    return results


def analyze_results(results: Dict[str, Dict[str, List[float]]], chains: Dict[str, Any]) -> None:
    """Analyze and display performance results."""
    print("\n" + "="*80)
    print("🏆 PERFORMANCE ANALYSIS RESULTS")
    print("="*80)
    
    # Calculate statistics for each configuration
    stats = {}
    for chain_name, result_data in results.items():
        if result_data["all_times"]:
            cache_miss_times = result_data["cache_miss"]
            cache_hit_times = result_data["cache_hit"]
            all_times = result_data["all_times"]
            
            stats[chain_name] = {
                "overall_mean": statistics.mean(all_times),
                "overall_median": statistics.median(all_times),
                "overall_min": min(all_times),
                "overall_max": max(all_times),
                "total_time": sum(all_times),
                "total_runs": len(all_times),
                "cache_miss_mean": statistics.mean(cache_miss_times) if cache_miss_times else 0,
                "cache_hit_mean": statistics.mean(cache_hit_times) if cache_hit_times else 0,
                "cache_miss_count": len(cache_miss_times),
                "cache_hit_count": len(cache_hit_times),
                "cache_improvement": 0
            }
            
            # Calculate cache improvement
            if cache_miss_times and cache_hit_times:
                miss_avg = statistics.mean(cache_miss_times)
                hit_avg = statistics.mean(cache_hit_times)
                stats[chain_name]["cache_improvement"] = ((miss_avg - hit_avg) / miss_avg) * 100
    
    # Display detailed results
    print("\n📊 Detailed Performance Statistics:")
    print("-" * 80)
    
    for chain_name, stat in stats.items():
        description = chains[chain_name]["description"]
        print(f"\n🔸 {description}:")
        print(f"   Overall Average: {stat['overall_mean']:.3f}s")
        print(f"   Overall Median:  {stat['overall_median']:.3f}s")
        print(f"   Total Time:      {stat['total_time']:.3f}s")
        print(f"   Total Runs:      {stat['total_runs']}")
        
        if stat['cache_miss_count'] > 0:
            print(f"   Cache Miss Avg:  {stat['cache_miss_mean']:.3f}s ({stat['cache_miss_count']} runs)")
        if stat['cache_hit_count'] > 0:
            print(f"   Cache Hit Avg:   {stat['cache_hit_mean']:.3f}s ({stat['cache_hit_count']} runs)")
        if stat['cache_improvement'] > 0:
            print(f"   🚀 Cache Effect:  {stat['cache_improvement']:.1f}% improvement")
    
    # Performance comparison
    print("\n🔄 Performance Comparison (vs Regular RAG):")
    print("-" * 80)
    
    if "regular" in stats:
        regular_mean = stats["regular"]["overall_mean"]
        regular_total = stats["regular"]["total_time"]
        
        comparison_order = ["regular", "llm_only", "retrieval_only", "full_cag"]
        
        for chain_name in comparison_order:
            if chain_name in stats:
                description = chains[chain_name]["description"]
                current_mean = stats[chain_name]["overall_mean"]
                current_total = stats[chain_name]["total_time"]
                
                if chain_name == "regular":
                    print(f"📍 {description}: {current_mean:.3f}s (baseline)")
                else:
                    improvement = ((regular_mean - current_mean) / regular_mean) * 100
                    time_saved = regular_total - current_total
                    
                    if improvement > 0:
                        print(f"🚀 {description}: {current_mean:.3f}s ({improvement:.1f}% faster, saved {time_saved:.3f}s)")
                    else:
                        print(f"📉 {description}: {current_mean:.3f}s ({abs(improvement):.1f}% slower)")
    
    # Cache effectiveness analysis
    print("\n🎯 Cache Effectiveness Analysis:")
    print("-" * 80)
    
    # Best overall performance
    if "full_cag" in stats and "regular" in stats:
        full_cag_mean = stats["full_cag"]["overall_mean"]
        regular_mean = stats["regular"]["overall_mean"]
        overall_improvement = ((regular_mean - full_cag_mean) / regular_mean) * 100
        
        print("🏆 Best Overall Performance: Full CAG")
        print(f"🚀 Overall Improvement: {overall_improvement:.1f}%")
        print(f"⚡ Speed Multiplier: {regular_mean / full_cag_mean:.1f}x faster")
    
    # Individual cache impact
    if all(name in stats for name in ["regular", "llm_only", "retrieval_only"]):
        regular_mean = stats["regular"]["overall_mean"]
        llm_only_mean = stats["llm_only"]["overall_mean"]
        retrieval_only_mean = stats["retrieval_only"]["overall_mean"]
        
        llm_impact = ((regular_mean - llm_only_mean) / regular_mean) * 100
        retrieval_impact = ((regular_mean - retrieval_only_mean) / regular_mean) * 100
        
        print("\n📈 Individual Cache Impact:")
        print(f"   💭 LLM Cache only: {llm_impact:.1f}% overall improvement")
        print(f"   📚 Retrieval Cache only: {retrieval_impact:.1f}% overall improvement")
    
    # Cache hit vs miss comparison
    print("\n🔍 Cache Hit vs Miss Comparison:")
    print("-" * 80)
    
    for chain_name, stat in stats.items():
        if stat['cache_improvement'] > 0:
            description = chains[chain_name]["description"]
            print(f"🔸 {description}:")
            print(f"   Cache Miss: {stat['cache_miss_mean']:.3f}s")
            print(f"   Cache Hit:  {stat['cache_hit_mean']:.3f}s")
            print(f"   🚀 Cache Improvement: {stat['cache_improvement']:.1f}%")
    
    print("\n" + "="*80)


def main():
    """
    Main function to run comprehensive caching performance comparison.
    
    This function demonstrates the performance benefits of different caching strategies
    in RAG pipelines by comparing execution times across multiple configurations.
    """
    print("🚀 Starting Comprehensive Caching Performance Comparison")
    print("=" * 80)
    
    # Load environment variables
    load_dotenv()
    
    # Ensure the OpenAI API key is set
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("❌ OPENAI_API_KEY environment variable not set. Please set it in your .env file.")
    
    try:
        # Setup
        vector_store = setup_vector_store()
        chains = setup_chains(vector_store)
        
        # Test queries (unique queries that will be repeated multiple times)
        test_queries = [
            "What is Cache-Augmented Generation?",
            "How does retrieval caching work?",
            "What are the benefits of LLM caching?",
            "What is hybrid caching?",
            "Explain cache hit and cache miss.",
            "How does in-memory caching work?",
            "What is persistent caching?",
        ]
        
        # Run performance tests (each query will be run multiple times to test caching)
        results = run_performance_test(chains, test_queries, runs_per_query=3)
        
        # Analyze results
        analyze_results(results, chains)
        
        print("\n✅ Performance comparison completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during performance comparison: {e}")
        raise


if __name__ == "__main__":
    main() 