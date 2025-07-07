import os
from dotenv import load_dotenv
from typing import List

import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# LangChain의 Contextual Compression과 Re-ranking imports
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_community.llms import Cohere

# 선택적 imports
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    print("⚠️  sentence-transformers not available. Cross-encoder re-ranking disabled.")

# Custom Cross-Encoder Compressor
try:
    from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
    from langchain.schema import Document
    CUSTOM_COMPRESSOR_AVAILABLE = True
except ImportError:
    CUSTOM_COMPRESSOR_AVAILABLE = False
    print("⚠️  Custom compressor components not available.")


class CrossEncoderReranker(BaseDocumentCompressor):
    """Custom Cross-Encoder Reranker using sentence-transformers."""
    
    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2", top_k: int = 5):
        """Initialize Cross-Encoder Reranker.
        
        Args:
            model_name (str): Name of the cross-encoder model
            top_k (int): Number of documents to return after reranking
        """
        self.model_name = model_name
        self.top_k = top_k
        self.cross_encoder = None
        
        if CROSS_ENCODER_AVAILABLE:
            try:
                self.cross_encoder = CrossEncoder(model_name)
                print(f"✅ Cross-encoder model '{model_name}' loaded successfully")
            except Exception as e:
                print(f"⚠️  Cross-encoder loading failed: {e}")
    
    def compress_documents(
        self, 
        documents: List[Document], 
        query: str
    ) -> List[Document]:
        """Compress documents using cross-encoder reranking.
        
        Args:
            documents (List[Document]): List of documents to rerank
            query (str): Query string for ranking
            
        Returns:
            List[Document]: Reranked and compressed documents
        """
        if not self.cross_encoder:
            print("⚠️  Cross-encoder not available, returning original documents")
            return documents[:self.top_k]
        
        print(f"🔄 Cross-encoder reranking {len(documents)} documents...")
        
        try:
            # Create query-document pairs
            query_doc_pairs = []
            for doc in documents:
                query_doc_pairs.append([query, doc.page_content[:512]])  # 512자 제한
            
            # Get relevance scores
            scores = self.cross_encoder.predict(query_doc_pairs)
            
            # Pair documents with scores
            doc_scores = list(zip(documents, scores))
            
            # Sort by score (descending)
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Return top_k documents
            reranked_docs = [doc for doc, score in doc_scores[:self.top_k]]
            
            print(f"✅ Cross-encoder reranking completed: {len(reranked_docs)} documents")
            return reranked_docs
            
        except Exception as e:
            print(f"⚠️  Cross-encoder reranking failed: {e}")
            return documents[:self.top_k]


class LangChainReranker:
    """LangChain official re-ranking approach."""
    
    def __init__(self):
        """Initialize LangChain Reranker with available methods."""
        self.llm = ChatOpenAI(temperature=0)
        self.cohere_llm = None
        self.cohere_reranker = None
        self.cross_encoder_reranker = None
        
        # Cohere 초기화
        try:
            cohere_api_key = os.getenv("COHERE_API_KEY")
            if cohere_api_key:
                self.cohere_llm = Cohere(temperature=0)
                self.cohere_reranker = CohereRerank(model="rerank-english-v3.0")
                print("✅ Cohere reranker initialized successfully")
        except Exception as e:
            print(f"⚠️  Cohere initialization failed: {e}")
        
        # Custom Cross-Encoder 초기화
        if CUSTOM_COMPRESSOR_AVAILABLE and CROSS_ENCODER_AVAILABLE:
            try:
                self.cross_encoder_reranker = CrossEncoderReranker(
                    model_name="ms-marco-MiniLM-L-12-v2",
                    top_k=5
                )
                print("✅ Custom Cross-Encoder reranker initialized")
            except Exception as e:
                print(f"⚠️  Cross-Encoder reranker initialization failed: {e}")
    
    def create_cohere_compression_retriever(self, base_retriever):
        """Create Cohere-based compression retriever.
        
        Args:
            base_retriever: Base retriever for initial document retrieval
            
        Returns:
            ContextualCompressionRetriever: Cohere-enhanced retriever
        """
        if not self.cohere_reranker:
            print("⚠️  Cohere reranker not available")
            return base_retriever
        
        print("🔄 Creating Cohere compression retriever...")
        
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.cohere_reranker,
            base_retriever=base_retriever
        )
        
        print("✅ Cohere compression retriever created")
        return compression_retriever
    
    def create_cross_encoder_compression_retriever(self, base_retriever):
        """Create Cross-Encoder-based compression retriever.
        
        Args:
            base_retriever: Base retriever for initial document retrieval
            
        Returns:
            ContextualCompressionRetriever: Cross-Encoder-enhanced retriever
        """
        if not self.cross_encoder_reranker:
            print("⚠️  Cross-Encoder reranker not available")
            return base_retriever
        
        print("🔄 Creating Cross-Encoder compression retriever...")
        
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.cross_encoder_reranker,
            base_retriever=base_retriever
        )
        
        print("✅ Cross-Encoder compression retriever created")
        return compression_retriever


def load_and_split_documents():
    """Load web documents and split them into chunks.
    
    Returns:
        list: List of document chunks ready for indexing
    """
    print("🔄 Loading documents from web...")
    
    # Load web documents using WebBaseLoader
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    blog_docs = loader.load()
    
    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=300, 
        chunk_overlap=50
    )
    
    # Make splits
    splits = text_splitter.split_documents(blog_docs)
    print(f"✅ Loaded and split {len(splits)} document chunks")
    
    return splits


def create_vectorstore_and_base_retriever(splits):
    """Create vectorstore and base retriever from document splits.
    
    Args:
        splits (list): List of document chunks
        
    Returns:
        BaseRetriever: Base retriever instance
    """
    print("🔄 Creating vectorstore and base retriever...")
    
    # Create vectorstore from documents
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=OpenAIEmbeddings()
    )
    
    # Create base retriever with higher k for reranking
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
    
    print("✅ Vectorstore and base retriever created successfully")
    return base_retriever


def pretty_print_docs(docs):
    """Pretty print documents with scores if available."""
    print(f"\n📄 Retrieved {len(docs)} documents:")
    print("-" * 80)
    
    for i, doc in enumerate(docs):
        print(f"Document {i+1}:")
        print(f"Content: {doc.page_content[:200]}...")
        
        # Print metadata if available
        if hasattr(doc, 'metadata') and doc.metadata:
            print(f"Metadata: {doc.metadata}")
        
        print("-" * 40)


def test_retrieval_methods(base_retriever, reranker, query):
    """Test different retrieval methods with the same query.
    
    Args:
        base_retriever: Base retriever instance
        reranker: LangChain reranker instance
        query (str): Test query
    """
    print(f"\n🔍 Testing retrieval methods with query: '{query}'")
    
    # Test 1: Base retrieval (no reranking)
    print("\n" + "="*60)
    print("📊 TEST 1: BASE RETRIEVAL (NO RERANKING)")
    print("="*60)
    
    base_docs = base_retriever.invoke(query)
    print(f"Base retrieval returned {len(base_docs)} documents")
    pretty_print_docs(base_docs[:3])  # Show first 3
    
    # Test 2: Cohere reranking
    if reranker.cohere_reranker:
        print("\n" + "="*60)
        print("🌐 TEST 2: COHERE RERANKING")
        print("="*60)
        
        cohere_retriever = reranker.create_cohere_compression_retriever(base_retriever)
        cohere_docs = cohere_retriever.invoke(query)
        print(f"Cohere reranking returned {len(cohere_docs)} documents")
        pretty_print_docs(cohere_docs)
    
    # Test 3: Cross-Encoder reranking
    if reranker.cross_encoder_reranker:
        print("\n" + "="*60)
        print("🔄 TEST 3: CROSS-ENCODER RERANKING")
        print("="*60)
        
        cross_retriever = reranker.create_cross_encoder_compression_retriever(base_retriever)
        cross_docs = cross_retriever.invoke(query)
        print(f"Cross-Encoder reranking returned {len(cross_docs)} documents")
        pretty_print_docs(cross_docs)


def test_qa_chains(base_retriever, reranker, query):
    """Test QA chains with different reranking methods.
    
    Args:
        base_retriever: Base retriever instance
        reranker: LangChain reranker instance
        query (str): Test query
    """
    print(f"\n🤖 Testing QA chains with query: '{query}'")
    
    # Test 1: Base QA Chain
    print("\n" + "="*60)
    print("📊 QA TEST 1: BASE CHAIN (NO RERANKING)")
    print("="*60)
    
    base_chain = RetrievalQA.from_chain_type(
        llm=reranker.llm,
        retriever=base_retriever
    )
    
    base_result = base_chain.invoke({"query": query})
    print("Base QA Result:")
    print(base_result["result"][:500] + "..." if len(base_result["result"]) > 500 else base_result["result"])
    
    # Test 2: Cohere QA Chain
    if reranker.cohere_reranker and reranker.cohere_llm:
        print("\n" + "="*60)
        print("🌐 QA TEST 2: COHERE RERANKING CHAIN")
        print("="*60)
        
        cohere_retriever = reranker.create_cohere_compression_retriever(base_retriever)
        cohere_chain = RetrievalQA.from_chain_type(
            llm=reranker.cohere_llm,
            retriever=cohere_retriever
        )
        
        cohere_result = cohere_chain.invoke({"query": query})
        print("Cohere QA Result:")
        print(cohere_result["result"][:500] + "..." if len(cohere_result["result"]) > 500 else cohere_result["result"])
    
    # Test 3: Cross-Encoder QA Chain
    if reranker.cross_encoder_reranker:
        print("\n" + "="*60)
        print("🔄 QA TEST 3: CROSS-ENCODER RERANKING CHAIN")
        print("="*60)
        
        cross_retriever = reranker.create_cross_encoder_compression_retriever(base_retriever)
        cross_chain = RetrievalQA.from_chain_type(
            llm=reranker.llm,
            retriever=cross_retriever
        )
        
        cross_result = cross_chain.invoke({"query": query})
        print("Cross-Encoder QA Result:")
        print(cross_result["result"][:500] + "..." if len(cross_result["result"]) > 500 else cross_result["result"])


def main():
    """Main function to execute the LangChain Re-ranking pipeline.
    
    This function demonstrates LangChain's official re-ranking approach using:
    1. Cohere Rerank API
    2. Custom Cross-Encoder Compressor
    3. ContextualCompressionRetriever
    """
    print("🚀 Starting LangChain Re-ranking pipeline...")
    
    # Load environment variables
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    
    # Step 1: Load and split documents
    splits = load_and_split_documents()
    
    # Step 2: Create vectorstore and base retriever
    base_retriever = create_vectorstore_and_base_retriever(splits)
    
    # Step 3: Initialize LangChain Reranker
    reranker = LangChainReranker()
    
    # Step 4: Test queries
    test_queries = [
        "What is task decomposition for LLM agents?",
        "How do agents plan and execute tasks?",
        "What are the challenges in agent reasoning?"
    ]
    
    for query in test_queries:
        print("\n" + "="*80)
        print(f"🔍 TESTING QUERY: {query}")
        print("="*80)
        
        # Test retrieval methods
        test_retrieval_methods(base_retriever, reranker, query)
        
        # Test QA chains
        test_qa_chains(base_retriever, reranker, query)
    
    print("\n" + "="*80)
    print("✅ LangChain Re-ranking pipeline completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main() 