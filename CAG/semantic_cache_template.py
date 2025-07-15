"""Template for a retriever that uses semantic caching."""

import uuid
from typing import List
import numpy as np
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss

class SemanticCachedRetriever(BaseRetriever):
    """
    A retriever that uses a vector store to cache results based on semantic similarity.

    This retriever wraps a standard retriever and adds a semantic caching layer.
    When a query is received, it first checks a vector store for semantically
    similar queries. If a similar query is found within a defined threshold,
    it returns the cached documents associated with that query.
    """

    retriever: BaseRetriever
    embedding_model: Embeddings
    cache_vector_store: FAISS
    cache: dict
    similarity_threshold: float
    max_cache_size: int

    def __init__(
        self,
        retriever: BaseRetriever,
        embedding_model: Embeddings,
        similarity_threshold: float = 0.95,
        max_cache_size: int = 1000,
    ):
        """Initialize the SemanticCachedRetriever.

        Args:
            retriever: The base retriever to wrap.
            embedding_model: The embedding model for semantic comparison.
            similarity_threshold: Cosine similarity threshold (higher is more similar).
            max_cache_size: Maximum number of cached queries (default: 1000).
        """
        # Determine embedding dimension from the model
        dummy_embedding = embedding_model.embed_query("test")
        d = len(dummy_embedding)

        # Create a FAISS index that uses cosine similarity (IndexFlatIP)
        index = faiss.IndexFlatIP(d)
        
        # Create an empty FAISS vector store instance
        docstore = InMemoryDocstore({})
        index_to_docstore_id = {}
        
        cache_vector_store = FAISS(
            embedding_function=embedding_model,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id
        )

        super().__init__(
            retriever=retriever,
            embedding_model=embedding_model,
            cache_vector_store=cache_vector_store,
            cache={},
            similarity_threshold=similarity_threshold,
            max_cache_size=max_cache_size,
        )

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Retrieve documents, using the semantic cache if a similar query is found.
        """
        query_embedding = self.embedding_model.embed_query(query)
        query_vector = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_vector) # Normalize for cosine similarity

        try:
            # Search for similar queries in the cache vector store
            similar_results = self.cache_vector_store.similarity_search_with_score_by_vector(
                query_vector.flatten(), k=1
            )
        except ValueError as e:
            # This can happen if the cache is empty or vector dimensions don't match
            print(f"⚠️  Vector search failed: {e}")
            similar_results = []
        except Exception as e:
            # Catch any other unexpected errors
            print(f"❌ Unexpected error during similarity search: {e}")
            similar_results = []

        # Check for a semantic cache hit
        if similar_results:
            most_similar_doc, score = similar_results[0]
            
            # For IndexFlatIP on normalized vectors, score is cosine similarity
            if score >= self.similarity_threshold:
                cached_query_id = most_similar_doc.metadata.get("query_id")
                if cached_query_id and cached_query_id in self.cache:
                    print(f"🎯 Semantic cache hit! Similarity score: {score:.4f} >= {self.similarity_threshold}")
                    print(f"   Found similar query: '{most_similar_doc.page_content}'")
                    return self.cache[cached_query_id]

        # Cache miss: perform a new search
        print(f"🔍 Semantic cache miss for query: '{query}'. Performing new search.")
        documents = self.retriever.invoke(
            query, config={"callbacks": run_manager.get_child()}
        )

        # Add the new query and its results to the cache
        query_id = str(uuid.uuid4())
        
        # Create a normalized embedding for consistent storage bc of FAISS IndexFlatIP (cosine similarity)
        normalized_embedding = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(normalized_embedding)
        
        # Check cache size and remove oldest if necessary
        if len(self.cache) >= self.max_cache_size:
            self._remove_oldest_cache_entry()
        
        # Add to vector store with normalized embedding
        self.cache_vector_store.add_texts(
            [query], 
            metadatas=[{"query_id": query_id}],
            embeddings=[normalized_embedding.flatten().tolist()]
        )
        self.cache[query_id] = documents
        print(f"💾 Cached {len(documents)} documents for new query: '{query}'")
        print(f"📊 Cache size: {len(self.cache)}/{self.max_cache_size}")

        return documents
    
    def _remove_oldest_cache_entry(self):
        """Remove the oldest cache entry when cache size limit is reached."""
        if not self.cache:
            return
            
        try:
            # Get all cached documents from the vector store
            if not hasattr(self.cache_vector_store, 'docstore') or not self.cache_vector_store.docstore:
                return
                
            # Get the first query_id from cache (oldest)
            oldest_query_id = next(iter(self.cache))
            
            # Remove from cache dictionary
            if oldest_query_id in self.cache:
                del self.cache[oldest_query_id]
                print(f"🗑️  Removed oldest cache entry (query_id: {oldest_query_id})")
                
                # For simplicity, we'll rebuild the entire vector store without this entry
                # In a production system, you might want a more efficient approach
                remaining_queries = []
                remaining_metadatas = []
                
                for query_id, _ in self.cache.items():
                    # Find the query text in the docstore
                    for doc_id, doc in self.cache_vector_store.docstore._dict.items():
                        if doc and hasattr(doc, 'metadata') and doc.metadata.get('query_id') == query_id:
                            remaining_queries.append(doc.page_content)
                            remaining_metadatas.append(doc.metadata)
                            break
                
                # Rebuild vector store if there are remaining queries
                if remaining_queries:
                    self.cache_vector_store = FAISS.from_texts(
                        remaining_queries,
                        self.embedding_model,
                        metadatas=remaining_metadatas
                    )
                else:
                    # Clear the vector store if no queries remain
                    self._clear_vector_store()
                    
        except Exception as e:
            print(f"⚠️  Error removing oldest cache entry: {e}")
            
    def _clear_vector_store(self):
        """Clear the vector store by recreating it."""
        dummy_embedding = self.embedding_model.embed_query("test")
        d = len(dummy_embedding)
        
        # Create a new empty FAISS index
        index = faiss.IndexFlatIP(d)
        docstore = InMemoryDocstore({})
        index_to_docstore_id = {}
        
        self.cache_vector_store = FAISS(
            embedding_function=self.embedding_model,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id
        )

    def clear_cache(self):
        """Clear all caches by re-initializing."""
        self.__init__(self.retriever, self.embedding_model, self.similarity_threshold, self.max_cache_size)
        print("🧹 Semantic cache cleared.")

    def get_cache_info(self):
        """Get information about the cache state."""
        try:
            # Get all cached documents from the vector store
            if hasattr(self.cache_vector_store, 'docstore') and self.cache_vector_store.docstore:
                # Use the docstore's internal storage
                queries = []
                for doc_id, doc in self.cache_vector_store.docstore._dict.items():
                    if doc and hasattr(doc, 'page_content'):
                        queries.append(doc.page_content)
            else:
                queries = []
        except Exception as e:
            print(f"⚠️  Error getting cache info: {e}")
            queries = []
            
        return {
            "cached_query_count": len(self.cache),
            "max_cache_size": self.max_cache_size,
            "cache_utilization": f"{len(self.cache)}/{self.max_cache_size} ({len(self.cache)/self.max_cache_size*100:.1f}%)",
            "cached_queries": queries,
        }
