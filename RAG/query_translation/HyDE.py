from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def main():
    """
    Main function to execute HyDE (Hypothetical Document Embeddings) RAG pipeline.
    
    HyDE works by:
    1. Generating a hypothetical document that would answer the question
    2. Using this generated document to search for similar real documents
    3. Using the retrieved real documents to generate the final answer
    """
    
    # Load environment variables
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
    
    print("🚀 Starting HyDE (Hypothetical Document Embeddings) RAG Pipeline...\n")
    
    # 1. Load documents
    print("📚 Loading documents from web...")
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
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
    
    # 4. Setup HyDE document generation
    print("🔮 Setting up HyDE Document Generation...")
    template = """Please write a scientific paper passage to answer the question
Question: {question}
Passage:"""
    
    prompt_hyde = ChatPromptTemplate.from_template(template)
    
    generate_docs_for_retrieval = (
        prompt_hyde | ChatOpenAI(temperature=0) | StrOutputParser() 
    )
    print("   ✅ HyDE document generator created\n")
    
    # 5. Generate hypothetical document
    print("📝 Generating Hypothetical Document...")
    question = "What is task decomposition for LLM agents?"
    print(f"   Original question: '{question}'")
    
    hypothetical_doc = generate_docs_for_retrieval.invoke({"question": question})
    print(f"   ✅ Generated hypothetical document ({len(hypothetical_doc)} chars)")
    print(f"   Preview: {hypothetical_doc[:200]}...")
    print()
    
    # 6. Retrieve documents using hypothetical document
    print("🔍 Retrieving documents using hypothetical document...")
    retrieval_chain = generate_docs_for_retrieval | retriever 
    retrieved_docs = retrieval_chain.invoke({"question": question})
    print(f"   ✅ Retrieved {len(retrieved_docs)} documents based on hypothetical document")
    
    # Show brief preview of retrieved docs
    if retrieved_docs:
        print(f"   First document preview: {retrieved_docs[0].page_content[:150]}...")
    print()
    
    # 7. Setup final RAG chain
    print("🎯 Setting up Final RAG Generation...")
    rag_template = """Answer the following question based on this context:

{context}

Question: {question}
"""
    
    prompt = ChatPromptTemplate.from_template(rag_template)
    
    final_rag_chain = (
        prompt
        | ChatOpenAI(temperature=0)
        | StrOutputParser()
    )
    print("   ✅ Final RAG chain created\n")
    
    # 8. Generate final answer
    print("🎬 Generating Final Answer...")
    final_answer = final_rag_chain.invoke({
        "context": retrieved_docs,
        "question": question
    })
    print("   ✅ Final answer generated\n")
    
    # 9. Display results
    print("="*60)
    print("📊 HyDE RAG RESULTS")
    print("="*60)
    
    print(f"Original Question: {question}\n")
    
    print("🔮 Hypothetical Document:")
    print("-" * 40)
    print(hypothetical_doc)
    print()
    
    print("📖 Retrieved Documents:")
    print("-" * 40)
    print(f"Total documents retrieved: {len(retrieved_docs)}")
    for i, doc in enumerate(retrieved_docs[:2], 1):  # Show first 2 docs
        print(f"\nDocument {i} (first 200 chars):")
        print(doc.page_content[:200] + "...")
    
    if len(retrieved_docs) > 2:
        print(f"\n... and {len(retrieved_docs) - 2} more documents")
    print()
    
    print("🎯 Final Answer:")
    print("-" * 40)
    print(final_answer)
    print()
    
    # 10. Show comparison with direct retrieval
    print("🔄 Comparison with Direct Retrieval:")
    print("-" * 40)
    print("   Comparing HyDE vs Direct question retrieval...")
    
    # Direct retrieval for comparison
    direct_retrieved = retriever.invoke(question)
    print(f"   Direct retrieval: {len(direct_retrieved)} documents")
    print(f"   HyDE retrieval: {len(retrieved_docs)} documents")
    
    # Check if documents are different
    direct_content = set([doc.page_content for doc in direct_retrieved])
    hyde_content = set([doc.page_content for doc in retrieved_docs])
    
    unique_to_direct = direct_content - hyde_content
    unique_to_hyde = hyde_content - direct_content
    overlap = direct_content & hyde_content
    
    print(f"   Unique to Direct: {len(unique_to_direct)}")
    print(f"   Unique to HyDE: {len(unique_to_hyde)}")
    print(f"   Overlapping: {len(overlap)}")
    print()
    
    print("💡 HyDE Benefits:")
    print("-" * 40)
    print("• Bridges vocabulary gap between question and documents")
    print("• Generates domain-specific language for better retrieval")
    print("• Potentially finds more relevant documents through semantic similarity")
    print("• Useful when questions are phrased differently than source documents")
    
    print("="*60)
    
    return {
        "original_question": question,
        "hypothetical_document": hypothetical_doc,
        "retrieved_documents": retrieved_docs,
        "final_answer": final_answer,
        "direct_retrieved": direct_retrieved,
        "retrieval_comparison": {
            "direct_docs": len(direct_retrieved),
            "hyde_docs": len(retrieved_docs),
            "unique_to_direct": len(unique_to_direct),
            "unique_to_hyde": len(unique_to_hyde),
            "overlap": len(overlap)
        }
    }

if __name__ == "__main__":
    main()