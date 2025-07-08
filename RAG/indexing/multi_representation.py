import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader

import uuid

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from langchain.storage import InMemoryByteStore
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers.multi_vector import MultiVectorRetriever


def load_documents():
    """Load documents from web sources.
    
    Returns:
        list: List of loaded documents from web sources.
    """
    # Load documents from two different web sources
    loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
    docs = loader.load()

    loader = WebBaseLoader("https://lilianweng.github.io/posts/2024-02-05-human-data-quality/")
    docs.extend(loader.load())
    
    return docs


def create_summaries(docs):
    """Create summaries for the given documents using LLM.
    
    Args:
        docs (list): List of documents to summarize.
        
    Returns:
        list: List of document summaries.
    """
    # Create chain for document summarization
    chain = (
        {"doc": lambda x: x.page_content}
        | ChatPromptTemplate.from_template("Summarize the following document: \n\n{doc}")
        | ChatOpenAI(model="gpt-4o-mini", temperature=0)
        | StrOutputParser()
    )

    summaries = [chain.invoke(doc) for doc in docs]
    return summaries


def setup_retriever():
    """Set up the MultiVectorRetriever with vectorstore and byte store.
    
    Returns:
        MultiVectorRetriever: Configured retriever instance.
    """
    # Initialize vectorstore for indexing child chunks
    vectorstore = Chroma(collection_name="summaries",
                         embedding_function=OpenAIEmbeddings())

    # Initialize storage for parent documents
    store = InMemoryByteStore()
    id_key = "doc_id"

    # Create and configure retriever
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=store,
        id_key=id_key,
        search_kwargs={"k": 1}  # Limit to 1 result
    )
    
    return retriever, vectorstore


def main():
    """Main function to execute the multi-representation RAG pipeline.
    
    This function demonstrates how to:
    1. Load documents from web sources
    2. Create summaries using LLM
    3. Set up MultiVectorRetriever
    4. Index documents and summaries
    5. Perform retrieval queries
    """
    # Load environment variables
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    
    # Load documents from web sources
    docs = load_documents()
    
    # Generate summaries for documents
    summaries = create_summaries(docs)
    print("\n" + "="*70)
    print("🔍 GENERATED SUMMARIES")
    print("="*70)
    for i, summary in enumerate(summaries, 1):
        print(f"\n📄 Document {i} Summary:")
        print("-" * 50)
        print(summary)
        print("-" * 50)
    
    # Set up retriever and vectorstore
    retriever, vectorstore = setup_retriever()
    
    # Generate unique IDs for documents
    doc_ids = [str(uuid.uuid4()) for _ in docs]

    # Create documents linked to summaries
    summary_docs = [
        Document(page_content=s, metadata={"doc_id": doc_ids[i]})
        for i, s in enumerate(summaries)
    ]

    # Add summary documents to vectorstore and original docs to docstore
    retriever.vectorstore.add_documents(summary_docs)
    retriever.docstore.mset(list(zip(doc_ids, docs)))

    # Perform similarity search
    query = "Memory in agents"
    sub_docs = vectorstore.similarity_search(query, k=1)

    print("\n" + "="*70)
    print("🔎 SIMILARITY SEARCH RESULTS")
    print("="*70)
    print(f"🔍 Query: '{query}'")
    print(f"📊 Found {len(sub_docs)} result(s)")
    print("\n📋 Search Results:")
    for i, doc in enumerate(sub_docs, 1):
        print(f"  {i}. {doc}")

    # Retrieve relevant documents using MultiVectorRetriever
    retrieved_docs = retriever.invoke(query)

    print("\n" + "="*70)
    print("📚 RETRIEVED DOCUMENTS")
    print("="*70)
    print(f"🔍 Query: '{query}'")
    print(f"📊 Retrieved {len(retrieved_docs)} document(s)")
    print("\n📋 Retrieved Documents:")
    
    if retrieved_docs:
        print("\n📄 First Retrieved Document Content (Preview):")
        print("-" * 60)
        print(retrieved_docs[0].page_content[0:100])
        print("-" * 60)
        print(f"💡 Content length: {len(retrieved_docs[0].page_content)} characters")
    print("="*70)
    


if __name__ == "__main__":
    main()

