# ruff: noqa: F841
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.load import dumps, loads
import bs4
from langchain_community.document_loaders import WebBaseLoader
from operator import itemgetter
from langchain.text_splitter import RecursiveCharacterTextSplitter

def reciprocal_rank_fusion(results: list[list], k=60):
    """ 
    Reciprocal_rank_fusion that takes multiple lists of ranked documents 
    and an optional parameter k used in the RRF formula 
    """
    
    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = dumps(doc)
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results

def main():
    """
    Main function to execute RAG-Fusion pipeline.
    
    This function demonstrates:
    1. Loading documents from web source
    2. Creating vector store
    3. Generating multiple related queries (RAG-Fusion)
    4. Using Reciprocal Rank Fusion for document reranking
    5. Generating final answer using reranked documents
    """
    
    # Load environment variables
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
    
    print("🚀 Starting RAG-Fusion Pipeline...\n")
    
    # 1. Load documents
    print("📚 Loading documents from web...")
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
        header_template={
            "User-Agent": "RAGFusion/1.0 (Educational Purpose)"
        }
    )
    blog_docs = loader.load()
    print(f"   ✅ Loaded {len(blog_docs)} documents")
    
    # 2. Split documents
    print("   Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=300, 
        chunk_overlap=50
    )
    splits = text_splitter.split_documents(blog_docs)
    print(f"   ✅ Created {len(splits)} chunks")
    
    # 3. Create vector store
    print("   Creating vector store...")
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=OpenAIEmbeddings()
    )
    retriever = vectorstore.as_retriever()
    print("   ✅ Vector store created\n")
    
    # 4. RAG-Fusion: Query Generation
    print("🔍 Setting up RAG-Fusion Query Generation...")
    template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
Generate multiple search queries related to: {question} \n
Output (4 queries):"""
    
    prompt_rag_fusion = ChatPromptTemplate.from_template(template)
    
    generate_queries = (
        prompt_rag_fusion 
        | ChatOpenAI(temperature=0)
        | StrOutputParser() 
        | (lambda x: x.split("\n"))
    )
    print("   ✅ RAG-Fusion query generator created")
    
    # 5. RAG-Fusion: Retrieval and Ranking
    print("   Creating RAG-Fusion retrieval chain...")
    question = "What is task decomposition for LLM agents?"
    retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion
    
    print(f"   Processing question: '{question}'")
    docs = retrieval_chain_rag_fusion.invoke({"question": question})
    print(f"   ✅ Retrieved and reranked {len(docs)} documents using RRF")
    
    # Show top documents with scores
    print("   📊 Top 3 documents with RRF scores:")
    for i, (doc, score) in enumerate(docs[:3]):
        print(f"      {i+1}. Score: {score:.4f} | Content: {doc.page_content[:100]}...")
    print()
    
    # 6. RAG Chain with Fusion Results
    print("✨ Setting up Final RAG Chain...")
    rag_template = """Answer the following question based on this context:

{context}

Question: {question}
"""
    
    prompt = ChatPromptTemplate.from_template(rag_template)
    llm = ChatOpenAI(temperature=0)
    
    def format_fusion_docs(fusion_results):
        """Format RAG-Fusion results for context"""
        return "\n\n".join([doc.page_content for doc, score in fusion_results])
    
    final_rag_chain = (
        {"context": retrieval_chain_rag_fusion | format_fusion_docs, 
         "question": itemgetter("question")} 
        | prompt
        | llm
        | StrOutputParser()
    )
    print("   ✅ RAG-Fusion chain created")
    
    # 7. Generate final answer
    print("   Generating final answer with RAG-Fusion...")
    result = final_rag_chain.invoke({"question": question})
    
    print("\n" + "="*50)
    print("📝 RAG-FUSION RESULT:")
    print("="*50)
    print(f"Question: {question}")
    print(f"Answer: {result}")
    print("="*50)
    print(f"📊 Used {len(docs)} documents reranked by Reciprocal Rank Fusion")
    
    return result

if __name__ == "__main__":
    main()
