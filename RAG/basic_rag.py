# ruff.noqa: F841
import tiktoken
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate

load_dotenv()

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def cosine_similarity(vec1, vec2):
    """Returns the cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

def format_docs(docs):
    """
    Formats retrieved documents into a single string for context.
    
    Args:
        docs: List of Document objects from retriever
        
    Returns:
        str: Formatted string containing all document contents
    """
    return "\n\n".join(doc.page_content for doc in docs)

def main():
    """
    Main function to demonstrate RAG indexing process.
    
    This function performs the following steps:
    1. Token counting demonstration
    2. Embedding generation and similarity calculation
    3. Web document loading for indexing
    """
    
    print("=== RAG Indexing Process Demo ===\n")
    
    # Token counting demonstration
    question = "What is the main topic of this article?"
    documents = "This article discusses artificial intelligence and machine learning applications in modern technology."
    
    question_tokens = num_tokens_from_string(question, "cl100k_base")
    document_tokens = num_tokens_from_string(documents, "cl100k_base")
    
    print("📝 Token Analysis:")
    print(f"   Question: '{question}' -> {question_tokens} tokens")
    print(f"   Document: '{documents}' -> {document_tokens} tokens\n")
    
    # Embedding and similarity demonstration
    print("🔍 Embedding & Similarity Analysis:")
    embed = OpenAIEmbeddings()
    query_result = embed.embed_query(question)
    document_result = embed.embed_documents([documents])
    
    print(f"   Query embedding length: {len(query_result)}")
    print(f"   Document embedding length: {len(document_result[0])}")
    
    similarity = cosine_similarity(query_result, document_result[0])
    print(f"   Cosine similarity: {similarity:.4f}\n")
    
    # Indexing phase
    # Web document loading for indexing
    print("🌐 Loading Web Documents for Indexing:")
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()
    print(f"   Loaded {len(docs)} documents from web source")
    print(f"   First document preview: {docs[0].page_content[:100]}...\n")
    
    # text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    # vector store
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

    print(f"   Vectorstore created with {len(splits)} chunks")

    # Retrieval phase
    retriever = vectorstore.as_retriever()
    docs = retriever.get_relevant_documents("What is Task Decomposition?")
    print(f"   Retrieved {len(docs)} documents")


    # Generation phase
    # prompt template
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    print(f"   Prompt: {prompt}")

    # llm
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # chain
    chain = prompt | llm
    result = chain.invoke({"context": docs, "question": "What is Task Decomposition?"})
    
    # result
    print(f"   Result: {result}")

    prompt_hub_rag = hub.pull("rlm/rag-prompt")
    print(f"   Prompt Hub RAG: {prompt_hub_rag}")

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_hub_rag
        | llm
        | StrOutputParser()
    )
    
    # RAG
    rag_result = rag_chain.invoke("What is Task Decomposition?")
    print(f"   RAG Chain Result: {rag_result}")

    # RAG
    return {
        "vectorstore": vectorstore,
        "retriever": retriever,
        "rag_chain": rag_chain,
        "demo_results": {
            "token_analysis": {
                "question_tokens": question_tokens,
                "document_tokens": document_tokens
            },
            "similarity_score": similarity,
            "retrieved_docs_count": len(docs),
            "generation_result": result.content if hasattr(result, 'content') else str(result)
        }
    }

if __name__ == "__main__":
    main()

