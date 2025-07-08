import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_rag_chain(web_url: str, model_name: str = "gpt-3.5-turbo"):
    """
    Creates a RAG chain for question answering.
    
    Args:
        web_url: Website URL to load documents from
        model_name: LLM model name to use
    
    Returns:
        rag_chain: Executable RAG chain
    """
    
    # 1. Load documents use different loader class by document type
    loader = WebBaseLoader(
        web_paths=(web_url,),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        )
    )
    docs = loader.load()
    
    # 2. Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)
    
    # 3. Create vector store
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=OpenAIEmbeddings()
    )
    retriever = vectorstore.as_retriever()
    
    # 4. Build RAG chain
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI(model_name=model_name, temperature=0)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

def main():
    """
    RAG chain execution example
    """
    # Create RAG chain
    rag_chain = create_rag_chain(
        web_url="https://lilianweng.github.io/posts/2023-06-23-agent/",
        model_name="gpt-3.5-turbo"
    )
    
    # Process question
    question = "What is Task Decomposition?"
    result = rag_chain.invoke(question)
    
    print(f"Question: {question}")
    print(f"Answer: {result}")
    
    return result

if __name__ == "__main__":
    main()