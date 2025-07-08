import os
from dotenv import load_dotenv
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.load import dumps, loads
from operator import itemgetter

def get_unique_union(documents: list[list]):
    """
    Get the unique union of retrieved documents.
    """
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]

def main():
    """
    Main function to execute Multi-Query RAG pipeline.
    
    This function demonstrates:
    1. Loading documents from web source
    2. Creating vector store
    3. Generating multiple query perspectives
    4. Retrieving unique documents from all queries
    5. Generating final answer using RAG
    """
    
    # Load environment variables
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
    
    print("🚀 Starting Multi-Query RAG Pipeline...\n")
    
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
    
    # 4. Multi Query: Different Perspectives
    print("🔍 Setting up Multi-Query Generation...")
    template = """You are an AI language model assistant. Your task is to generate five 
                different versions of the given user question to retrieve relevant documents from a vector 
                database. By generating multiple perspectives on the user question, your goal is to help
                the user overcome some of the limitations of the distance-based similarity search. 
                Provide these alternative questions separated by newlines. 
                Original question: {question}"""
    
    prompt_perspectives = ChatPromptTemplate.from_template(template)
    
    generate_queries = (
        prompt_perspectives 
        | ChatOpenAI(temperature=0) 
        | StrOutputParser() 
        | (lambda x: x.split("\n"))
    )
    print("   ✅ Multi-query generator created")
    
    # 5. Retrieve documents
    print("   Creating retrieval chain...")
    question = "What is task decomposition for LLM agents?"
    retrieval_chain = generate_queries | retriever.map() | get_unique_union
    
    print(f"   Processing question: '{question}'")
    docs = retrieval_chain.invoke({"question": question})
    print(f"   ✅ Retrieved {len(docs)} unique documents\n")
    
    # 6. RAG Chain
    print("✨ Setting up RAG Chain...")
    rag_template = """Answer the following question based on this context:

                    {context}

                    Question: {question}
                    """
    
    prompt = ChatPromptTemplate.from_template(rag_template)
    llm = ChatOpenAI(temperature=0)
    
    final_rag_chain = (
        {"context": retrieval_chain, 
         "question": itemgetter("question")} 
        | prompt
        | llm
        | StrOutputParser()
    )
    print("   ✅ RAG chain created")
    
    # 7. Generate final answer
    print("   Generating final answer...")
    result = final_rag_chain.invoke({"question": question})
    
    print("\n" + "="*50)
    print("📝 MULTI-QUERY RAG RESULT:")
    print("="*50)
    print(f"Question: {question}")
    print(f"Answer: {result}")
    print("="*50)
    
    return result

if __name__ == "__main__":
    main()