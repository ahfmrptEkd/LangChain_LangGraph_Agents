import os
from dotenv import load_dotenv
from operator import itemgetter

import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.load import dumps, loads


def reciprocal_rank_fusion(results: list[list], k=60):
    """Reciprocal Rank Fusion algorithm for combining multiple ranked lists.
    
    This function takes multiple lists of ranked documents and combines them
    using the RRF formula to produce a single reranked list.
    
    Args:
        results (list[list]): List of ranked document lists to be fused
        k (int, optional): Parameter used in RRF formula. Defaults to 60.
    
    Returns:
        list: List of tuples containing (document, fused_score) sorted by score
    """
    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
        
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = dumps(doc)
        
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
        
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results


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


def create_vectorstore_and_retriever(splits):
    """Create vectorstore and retriever from document splits.
    
    Args:
        splits (list): List of document chunks
        
    Returns:
        BaseRetriever: Configured retriever instance
    """
    print("🔄 Creating vectorstore and retriever...")
    
    # Create vectorstore from documents
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=OpenAIEmbeddings()
    )
    retriever = vectorstore.as_retriever()
    
    print("✅ Vectorstore and retriever created successfully")
    return retriever


def setup_rag_fusion_chain(retriever):
    """Set up RAG-Fusion chain for query generation and retrieval.
    
    Args:
        retriever (BaseRetriever): Document retriever instance
        
    Returns:
        Chain: Configured RAG-Fusion chain
    """
    print("🔄 Setting up RAG-Fusion chain...")
    
    # RAG-Fusion template for generating multiple queries
    template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
                Generate multiple search queries related to: {question} \n
                Output (4 queries):"""

    prompt_rag_fusion = ChatPromptTemplate.from_template(template)

    # Chain for generating multiple queries
    generate_queries = (
        prompt_rag_fusion 
        | ChatOpenAI(temperature=0)
        | StrOutputParser() 
        | (lambda x: x.split("\n"))
    )

    # Complete RAG-Fusion retrieval chain
    retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion
    
    print("✅ RAG-Fusion chain setup complete")
    return retrieval_chain_rag_fusion


def setup_final_rag_chain(retrieval_chain_rag_fusion):
    """Set up final RAG chain for answer generation.
    
    Args:
        retrieval_chain_rag_fusion (Chain): RAG-Fusion retrieval chain
        
    Returns:
        Chain: Final RAG chain for generating answers
    """
    print("🔄 Setting up final RAG chain...")
    
    # Final RAG template for answer generation
    template = """Answer the following question based on this context:

                {context}

                Question: {question}
                """

    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(temperature=0)

    # Final RAG chain combining retrieval and generation
    final_rag_chain = (
        {"context": retrieval_chain_rag_fusion, 
         "question": itemgetter("question")} 
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("✅ Final RAG chain setup complete")
    return final_rag_chain


def main():
    """Main function to execute the RAG-Fusion pipeline.
    
    This function orchestrates the entire RAG-Fusion workflow including:
    1. Document loading and splitting
    2. Vectorstore creation
    3. RAG-Fusion chain setup
    4. Query execution and answer generation
    """
    print("🚀 Starting RAG-Fusion pipeline...")
    
    # Load environment variables
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    
    # Step 1: Load and split documents
    splits = load_and_split_documents()
    
    # Step 2: Create vectorstore and retriever
    retriever = create_vectorstore_and_retriever(splits)
    
    # Step 3: Setup RAG-Fusion chain
    retrieval_chain_rag_fusion = setup_rag_fusion_chain(retriever)
    
    # Step 4: Test retrieval with sample question
    question = "What is task decomposition for LLM agents?"
    print(f"🔍 Testing retrieval with question: '{question}'")
    
    docs = retrieval_chain_rag_fusion.invoke({"question": question})
    print(f"📄 Retrieved {len(docs)} documents after fusion")
    
    # Step 5: Setup final RAG chain
    final_rag_chain = setup_final_rag_chain(retrieval_chain_rag_fusion)
    
    # Step 6: Generate final answer
    print("🤖 Generating final answer...")
    answer = final_rag_chain.invoke({"question": question})
    
    print("\n" + "="*50)
    print("📋 FINAL ANSWER:")
    print("="*50)
    print(answer)
    print("="*50)
    
    print("\n✅ RAG-Fusion pipeline completed successfully!")


if __name__ == "__main__":
    main()