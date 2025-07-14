"""Template for Cache-Augmented Generation (CAG) retriever."""

from typing import List
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache

from caching import InMemoryCache as CustomInMemoryCache

class CachedRetriever(BaseRetriever):
    """
    A retriever that uses both custom retrieval caching and LangChain's LLM caching.
    
    This class provides a hybrid caching approach:
    - Custom cache for retrieval results (documents)
    - LangChain's global cache for LLM calls
    """
    
    retriever: BaseRetriever
    cache: CustomInMemoryCache
    
    def __init__(self, retriever: BaseRetriever, cache: CustomInMemoryCache = None, enable_llm_cache: bool = True):
        """Initialize the CachedRetriever.

        Args:
            retriever: The retriever to wrap.
            cache: Custom cache for retrieval results. If None, creates a new InMemoryCache.
            enable_llm_cache: Whether to enable LangChain's global LLM cache.
        """
        
        super().__init__(
            retriever=retriever,
            cache=cache if cache is not None else CustomInMemoryCache()
        )
        
        # Set up the global LLM cache for LangChain
        if enable_llm_cache:
            set_llm_cache(InMemoryCache())
            print("✓ LangChain LLM cache is enabled.")

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Retrieve relevant documents for the given query.
        
        This method first checks the cache for the query. If found (cache hit),
        it returns the cached documents. If not found (cache miss), it calls
        the underlying retriever, caches the result, and returns it.
        
        Args:
            query: The query string to search for.
            run_manager: The callback manager for the retriever run.
            
        Returns:
            List of relevant documents.
        """
        # Check cache first
        cached_docs = self.cache.get(query)
        if cached_docs is not None:
            print(f"🎯 Retrieval cache hit for query: '{query}'")
            return cached_docs
        
        # Cache miss - call the underlying retriever using the new invoke method
        print(f"🔍 Retrieval cache miss for query: '{query}' - performing new search")
        
        # Use invoke method instead of deprecated get_relevant_documents
        documents = self.retriever.invoke(
            query, 
            config={"callbacks": run_manager.get_child() if run_manager else None}
        )
        
        # Cache the result
        self.cache.set(query, documents)
        print(f"💾 Cached {len(documents)} documents for query: '{query}'")
        
        return documents
    
    def clear_cache(self):
        """Clear the retrieval cache."""
        self.cache.clear()
        print("🧹 Retrieval cache cleared.")
        
    def get_cache_info(self):
        """Get information about the current cache state."""
        return {
            "cache_size": len(self.cache._cache),
            "cache_keys": list(self.cache._cache.keys())
        }
