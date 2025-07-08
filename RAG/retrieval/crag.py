import os
from dotenv import load_dotenv
from typing import List
from typing_extensions import TypedDict

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_tavily import TavilySearch
from langchain.schema import Document

# LangGraph imports
from langgraph.graph import END, StateGraph, START


class GraphState(TypedDict):
    """Represents the state of our CRAG graph.

    Attributes:
        question (str): User's question
        generation (str): LLM generation result
        web_search (str): Whether to add web search ("Yes" or "No")
        documents (List[str]): List of retrieved documents
    """
    question: str
    generation: str
    web_search: str
    documents: List[str]


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents.
    
    Attributes:
        binary_score (str): Documents relevance score, 'yes' or 'no'
    """
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


class CRAGWorkflow:
    """CRAG (Corrective RAG) workflow implementation class.
    
    This class implements the Corrective RAG workflow that:
    1. Retrieves documents from vector store
    2. Grades document relevance
    3. Performs web search if documents are not relevant
    4. Transforms query if needed
    5. Generates final answer
    """
    
    def __init__(self):
        """Initialize CRAG workflow with necessary components."""
        print("🚀 Initializing CRAG workflow...")
        
        # Initialize components
        self.retriever = None
        self.retrieval_grader = None
        self.rag_chain = None
        self.question_rewriter = None
        self.web_search_tool = None
        self.compiled_workflow = None
        
        print("✅ CRAG workflow initialized")
    
    def setup_document_retrieval(self, urls: List[str]):
        """Set up document loading and vector store for retrieval.
        
        Args:
            urls (List[str]): List of URLs to load documents from
            
        Returns:
            None
        """
        print("🔄 Setting up document retrieval system...")
        
        # Load documents from URLs
        print(f"📄 Loading documents from {len(urls)} URLs...")
        docs = [WebBaseLoader(url).load() for url in urls]
        docs_list = [item for sublist in docs for item in sublist]
        print(f"✅ Loaded {len(docs_list)} documents")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=250, 
            chunk_overlap=0
        )
        doc_splits = text_splitter.split_documents(docs_list)
        print(f"✅ Split into {len(doc_splits)} chunks")
        
        # Create vector store
        print("🔄 Creating vector store...")
        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name="rag-chroma",
            embedding=OpenAIEmbeddings(),
        )
        self.retriever = vectorstore.as_retriever()
        print("✅ Vector store and retriever created")
    
    def setup_document_grader(self):
        """Set up the document relevance grader.
        
        Returns:
            None
        """
        print("🔄 Setting up document grader...")
        
        # LLM with structured output
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        structured_llm_grader = llm.with_structured_output(GradeDocuments)
        
        # Grading prompt
        system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
                    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
                    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
        
        grade_prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ])
        
        self.retrieval_grader = grade_prompt | structured_llm_grader
        print("✅ Document grader setup complete")
    
    def setup_rag_chain(self):
        """Set up the RAG chain for answer generation.
        
        Returns:
            None
        """
        print("🔄 Setting up RAG chain...")
        
        # Pull RAG prompt from hub
        prompt = hub.pull("rlm/rag-prompt")
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        
        self.rag_chain = prompt | llm | StrOutputParser()
        print("✅ RAG chain setup complete")
    
    def setup_question_rewriter(self):
        """Set up the question rewriter for query transformation.
        
        Returns:
            None
        """
        print("🔄 Setting up question rewriter...")
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        system = """You a question re-writer that converts an input question to a better version that is optimized \n 
                    for web search. Look at the input and try to reason about the underlying semantic intent / meaning."""
        
        re_write_prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
        ])
        
        self.question_rewriter = re_write_prompt | llm | StrOutputParser()
        print("✅ Question rewriter setup complete")
    
    def setup_web_search(self):
        """Set up web search tool for additional information retrieval.
        
        Returns:
            None
        """
        print("🔄 Setting up web search tool...")
        
        self.web_search_tool = TavilySearch(k=3)
        print("✅ Web search tool setup complete")
    
    def retrieve(self, state):
        """Retrieve documents based on the question.
        
        Args:
            state (dict): Current graph state containing question
            
        Returns:
            dict: Updated state with retrieved documents
        """
        print("---RETRIEVE---")
        question = state["question"]
        
        # Retrieve documents
        documents = self.retriever.invoke(question)
        print(f"📄 Retrieved {len(documents)} documents")
        
        return {"documents": documents, "question": question}
    
    def generate(self, state):
        """Generate answer based on retrieved documents.
        
        Args:
            state (dict): Current graph state with documents and question
            
        Returns:
            dict: Updated state with generated answer
        """
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]
        
        # Generate answer using RAG chain
        generation = self.rag_chain.invoke({"context": documents, "question": question})
        print("✅ Answer generated")
        
        return {"documents": documents, "question": question, "generation": generation}
    
    def grade_documents(self, state):
        """Grade retrieved documents for relevance to the question.
        
        Args:
            state (dict): Current graph state with documents and question
            
        Returns:
            dict: Updated state with filtered documents and web search flag
        """
        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]
        
        # Grade each document
        filtered_docs = []
        web_search = "No"
        
        for d in documents:
            score = self.retrieval_grader.invoke({
                "question": question, 
                "document": d.page_content
            })
            grade = score.binary_score
            
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                web_search = "Yes"
                continue
        
        print(f"📊 Filtered to {len(filtered_docs)} relevant documents")
        return {"documents": filtered_docs, "question": question, "web_search": web_search}
    
    def transform_query(self, state):
        """Transform the query to produce a better search question.
        
        Args:
            state (dict): Current graph state with question
            
        Returns:
            dict: Updated state with transformed question
        """
        print("---TRANSFORM QUERY---")
        question = state["question"]
        documents = state["documents"]
        
        # Re-write question for better search
        better_question = self.question_rewriter.invoke({"question": question})
        print(f"🔄 Query transformed: '{question}' -> '{better_question}'")
        
        return {"documents": documents, "question": better_question}
    
    def web_search(self, state):
        """Perform web search based on the transformed question.
        
        Args:
            state (dict): Current graph state with question and documents
            
        Returns:
            dict: Updated state with web search results appended
        """
        print("---WEB SEARCH---")
        question = state["question"]
        documents = state["documents"]
        
        # Perform web search
        docs = self.web_search_tool.invoke({"query": question})
        
        # Handle different return formats from TavilySearch
        if isinstance(docs, str):
            # TavilySearch returns a string directly
            web_results = docs
            search_count = 1
        elif isinstance(docs, list):
            if len(docs) > 0:
                # Check if it's a list of dicts with 'content' key
                if isinstance(docs[0], dict) and 'content' in docs[0]:
                    web_results = "\n".join([d["content"] for d in docs])
                    search_count = len(docs)
                # Check if it's a list of strings
                elif isinstance(docs[0], str):
                    web_results = "\n".join(docs)
                    search_count = len(docs)
                else:
                    # Convert to string representation
                    web_results = "\n".join([str(d) for d in docs])
                    search_count = len(docs)
            else:
                web_results = "No search results found"
                search_count = 0
        else:
            # Convert to string representation
            web_results = str(docs)
            search_count = 1
        
        # Create document from web results
        web_results_doc = Document(page_content=web_results)
        documents.append(web_results_doc)
        
        print(f"🌐 Web search completed, added {search_count} results")
        return {"documents": documents, "question": question}
    
    def decide_to_generate(self, state):
        """Decide whether to generate answer or transform query.
        
        Args:
            state (dict): Current graph state
            
        Returns:
            str: Next node to call ("transform_query" or "generate")
        """
        print("---ASSESS GRADED DOCUMENTS---")
        web_search = state["web_search"]
        
        if web_search == "Yes":
            print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---")
            return "transform_query"
        else:
            print("---DECISION: GENERATE---")
            return "generate"
    
    def build_workflow(self):
        """Build the CRAG workflow graph.
        
        Returns:
            None
        """
        print("🔄 Building CRAG workflow graph...")
        
        # Create workflow
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("grade_documents", self.grade_documents)
        workflow.add_node("generate", self.generate)
        workflow.add_node("transform_query", self.transform_query)
        workflow.add_node("web_search_node", self.web_search)
        
        # Add edges
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )
        workflow.add_edge("transform_query", "web_search_node")
        workflow.add_edge("web_search_node", "generate")
        workflow.add_edge("generate", END)
        
        # Compile workflow
        self.compiled_workflow = workflow.compile()
        print("✅ CRAG workflow graph built and compiled")
    
    def run_workflow(self, question: str):
        """Run the complete CRAG workflow for a given question.
        
        Args:
            question (str): Question to process
            
        Returns:
            dict: Final result with generated answer
        """
        print(f"\n🔍 Running CRAG workflow for question: '{question}'")
        print("="*80)
        
        # Run workflow
        result = self.compiled_workflow.invoke({"question": question})
        
        print("\n" + "="*80)
        print("📋 CRAG WORKFLOW RESULT:")
        print("="*80)
        print(f"Question: {question}")
        print(f"Answer: {result.get('generation', 'No answer generated')}")
        print("="*80)
        
        return result
    
    def run_streaming_workflow(self, question: str):
        """Run the CRAG workflow with streaming output.
        
        Args:
            question (str): Question to process
            
        Returns:
            dict: Final result with generated answer
        """
        print(f"\n🔍 Running CRAG workflow (streaming) for question: '{question}'")
        print("="*80)
        
        inputs = {"question": question}
        final_result = None
        
        # Stream workflow execution
        for output in self.compiled_workflow.stream(inputs):
            for key, value in output.items():
                print(f"📍 Node '{key}' completed")
                final_result = value
        
        print("\n" + "="*80)
        print("📋 FINAL CRAG RESULT:")
        print("="*80)
        if final_result and "generation" in final_result:
            print(f"Answer: {final_result['generation']}")
        else:
            print("No final answer generated")
        print("="*80)
        
        return final_result


def format_docs(docs):
    """Format documents for display.
    
    Args:
        docs: List of documents
        
    Returns:
        str: Formatted document string
    """
    return "\n\n".join(doc.page_content for doc in docs)


def main():
    """Main function to execute the CRAG (Corrective RAG) pipeline.
    
    This function demonstrates the complete CRAG workflow including:
    1. Document loading and vector store setup
    2. Document relevance grading
    3. Query transformation
    4. Web search integration
    5. Answer generation
    """
    print("🚀 Starting CRAG (Corrective RAG) pipeline...")
    
    # Load environment variables
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    
    # Initialize CRAG workflow
    crag = CRAGWorkflow()
    
    # Step 1: Setup document retrieval
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]
    crag.setup_document_retrieval(urls)
    
    # Step 2: Setup all components
    crag.setup_document_grader()
    crag.setup_rag_chain()
    crag.setup_question_rewriter()
    crag.setup_web_search()
    
    # Step 3: Build workflow
    crag.build_workflow()
    
    # Step 4: Test with sample questions
    test_questions = [
        "What are the types of agent memory?",
        "What is task decomposition for LLM agents?",
        "How do agents handle planning and reasoning?",
        "What are the security concerns with LLM agents?"
    ]
    
    print("\n🧪 Testing CRAG workflow with sample questions...")
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*20} TEST {i}/{len(test_questions)} {'='*20}")
        
        # Run workflow
        crag.run_workflow(question)
        
        # Optional: Run streaming version for one question
        if i == 1:
            print(f"\n{'='*20} STREAMING VERSION {'='*20}")
            crag.run_streaming_workflow(question)
    
    print("\n✅ CRAG pipeline completed successfully!")
    print("\n📊 CRAG 워크플로우 특징:")
    print("• 문서 관련성 자동 평가")
    print("• 관련성 낮은 경우 웹 검색 수행")
    print("• 질문 자동 변환 및 최적화")
    print("• LangGraph 기반 워크플로우")


if __name__ == "__main__":
    main()