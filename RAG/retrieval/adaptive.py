import os 
from dotenv import load_dotenv
from typing import Literal, List, TypedDict
from pprint import pprint

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_tavily import TavilySearch

from langchain.schema import Document
from langgraph.graph import END, StateGraph, START


# Data models
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]


def format_docs(docs):
    """
    Format documents for context.
    
    Args:
        docs: List of documents
        
    Returns:
        str: Formatted document content
    """
    return "\n\n".join(doc.page_content for doc in docs)


def create_adaptive_rag_system():
    """
    Create and configure the adaptive RAG system with all components.
    
    Returns:
        tuple: Contains all the necessary components for the adaptive RAG system
    """
    # Load environment variables
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    
    # Set embeddings
    embd = OpenAIEmbeddings()
    
    # Documents to index
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]
    
    # Load and process documents
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)
    
    # Create vectorstore
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=embd,
    )
    retriever = vectorstore.as_retriever()
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Create question router
    structured_llm_router = llm.with_structured_output(RouteQuery)
    system = """You are an expert at routing a user question to a vectorstore or web search.
    The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
    Use the vectorstore for questions on these topics. Otherwise, use web-search."""
    
    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )
    question_router = route_prompt | structured_llm_router
    
    # Create retrieval grader
    structured_llm_grader = llm.with_structured_output(GradeDocuments)
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )
    retrieval_grader = grade_prompt | structured_llm_grader
    
    # Create RAG chain
    prompt = hub.pull("rlm/rag-prompt")
    rag_chain = prompt | llm | StrOutputParser()
    
    # Create hallucination grader
    structured_llm_grader = llm.with_structured_output(GradeHallucinations)
    system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
         Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
    
    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
        ]
    )
    hallucination_grader = hallucination_prompt | structured_llm_grader
    
    # Create answer grader
    structured_llm_grader = llm.with_structured_output(GradeAnswer)
    system = """You are a grader assessing whether an answer addresses / resolves a question \n 
         Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
    
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
        ]
    )
    answer_grader = answer_prompt | structured_llm_grader
    
    # Create question rewriter
    system = """You a question re-writer that converts an input question to a better version that is optimized \n 
         for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
    
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Here is the initial question: \n\n {question} \n Formulate an improved question.",
            ),
        ]
    )
    question_rewriter = re_write_prompt | llm | StrOutputParser()
    
    # Create web search tool
    web_search_tool = TavilySearch(k=3)
    
    return (
        retriever, question_router, retrieval_grader, rag_chain, 
        hallucination_grader, answer_grader, question_rewriter, web_search_tool
    )


def create_node_functions(retriever, rag_chain, retrieval_grader, question_rewriter, web_search_tool, question_router, hallucination_grader, answer_grader):
    """
    Create all node functions for the adaptive RAG workflow.
    
    Args:
        retriever: Document retriever
        rag_chain: RAG generation chain
        retrieval_grader: Document relevance grader
        question_rewriter: Query rewriter
        web_search_tool: Web search tool
        question_router: Question router
        hallucination_grader: Hallucination grader
        answer_grader: Answer grader
        
    Returns:
        dict: Dictionary of node functions
    """
    
    def retrieve(state):
        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("---RETRIEVE---")
        question = state["question"]

        # Retrieval
        documents = retriever.invoke(question)
        return {"documents": documents, "question": question}

    def generate(state):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]

        # RAG generation
        generation = rag_chain.invoke({"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}

    def grade_documents(state):
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with only filtered relevant documents
        """
        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        # Score each doc
        filtered_docs = []
        for d in documents:
            score = retrieval_grader.invoke(
                {"question": question, "document": d.page_content}
            )
            grade = score.binary_score
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                continue
        return {"documents": filtered_docs, "question": question}

    def transform_query(state):
        """
        Transform the query to produce a better question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates question key with a re-phrased question
        """
        print("---TRANSFORM QUERY---")
        question = state["question"]
        documents = state["documents"]

        # Re-write question
        better_question = question_rewriter.invoke({"question": question})
        return {"documents": documents, "question": better_question}

    def web_search(state):
        """
        Web search based on the re-phrased question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with appended web results
        """
        print("---WEB SEARCH---")
        question = state["question"]

        try:
            # Web search
            docs = web_search_tool.invoke({"query": question})
            print(f"Web search results type: {type(docs)}")
            print(f"Web search results: {docs}")
            
            # Handle different return formats from TavilySearch
            if isinstance(docs, list):
                if docs and isinstance(docs[0], dict):
                    # Format: [{'content': '...', 'url': '...'}, ...]
                    web_results = "\n".join([d.get("content", str(d)) for d in docs])
                elif docs and isinstance(docs[0], str):
                    # Format: ['result1', 'result2', ...]
                    web_results = "\n".join(docs)
                else:
                    web_results = str(docs)
            elif isinstance(docs, dict):
                # Format: {'content': '...', 'url': '...'}
                web_results = docs.get("content", str(docs))
            else:
                # Format: string or other
                web_results = str(docs)
                
            web_results = Document(page_content=web_results)
            
        except Exception as e:
            print(f"Web search error: {e}")
            web_results = Document(page_content=f"Web search failed: {str(e)}")

        return {"documents": web_results, "question": question}

    def route_question(state):
        """
        Route question to web search or RAG.

        Args:
            state (dict): The current graph state

        Returns:
            str: Next node to call
        """
        print("---ROUTE QUESTION---")
        question = state["question"]
        source = question_router.invoke({"question": question})
        if source.datasource == "web_search":
            print("---ROUTE QUESTION TO WEB SEARCH---")
            return "web_search"
        elif source.datasource == "vectorstore":
            print("---ROUTE QUESTION TO RAG---")
            return "vectorstore"

    def decide_to_generate(state):
        """
        Determines whether to generate an answer, or re-generate a question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """
        print("---ASSESS GRADED DOCUMENTS---")
        state["question"]
        filtered_documents = state["documents"]

        if not filtered_documents:
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            print(
                "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
            )
            return "transform_query"
        else:
            # We have relevant documents, so generate answer
            print("---DECISION: GENERATE---")
            return "generate"

    def grade_generation_v_documents_and_question(state):
        """
        Determines whether the generation is grounded in the document and answers question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Decision for next node to call
        """
        print("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        score = hallucination_grader.invoke(
            {"documents": documents, "generation": generation}
        )
        grade = score.binary_score

        # Check hallucination
        if grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            # Check question-answering
            print("---GRADE GENERATION vs QUESTION---")
            score = answer_grader.invoke({"question": question, "generation": generation})
            grade = score.binary_score
            if grade == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"

    return {
        "retrieve": retrieve,
        "generate": generate,
        "grade_documents": grade_documents,
        "transform_query": transform_query,
        "web_search": web_search,
        "route_question": route_question,
        "decide_to_generate": decide_to_generate,
        "grade_generation_v_documents_and_question": grade_generation_v_documents_and_question
    }


def create_adaptive_rag_graph(node_functions):
    """
    Create the adaptive RAG graph workflow.
    
    Args:
        node_functions (dict): Dictionary of node functions
        
    Returns:
        Compiled graph application
    """
    # Create workflow graph
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("web_search", node_functions["web_search"])
    workflow.add_node("retrieve", node_functions["retrieve"])
    workflow.add_node("grade_documents", node_functions["grade_documents"])
    workflow.add_node("transform_query", node_functions["transform_query"])
    workflow.add_node("generate", node_functions["generate"])
    workflow.add_node("grade_generation_v_documents_and_question", node_functions["grade_generation_v_documents_and_question"])

    # Add edges
    workflow.add_conditional_edges(
        START,
        node_functions["route_question"],
        {
            "web_search": "web_search",
            "vectorstore": "retrieve",
        },
    )
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        node_functions["decide_to_generate"],
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_conditional_edges(
        "generate",
        node_functions["grade_generation_v_documents_and_question"],
        {
            "useful": END,
            "not useful": "transform_query",
            "not supported": "generate",
        },
    )

    return workflow.compile()


def run_adaptive_rag_demo(test_questions):
    """
    Run demonstration of the adaptive RAG system.
    실제 질문이 들어와서 adaptive하게 처리되는 전체 워크플로우를 보여줍니다.
    
    Args:
        test_questions (list): List of test questions to run through the workflow
    """
    print("=== Adaptive RAG Full Workflow Demo ===")
    print("질문이 들어오면 자동으로 vectorstore/web search 경로를 선택하여 처리합니다.\n")
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n--- Workflow Test {i}: {question} ---")
        inputs = {"question": question}
        
        for output in app.stream(inputs):
            for key, value in output.items():
                pprint(f"Node '{key}':")
            pprint("\n---\n")
        
        print(f"Final Answer: {value.get('generation', 'No generation found')}")
        print("=" * 50)


def main():
    """
    Main function to run the adaptive RAG system.
    """
    global app  # Make app global so it can be used in demo function
    
    try:
        print("Initializing Adaptive RAG System...")
        
        # Create all system components
        components = create_adaptive_rag_system()
        (retriever, question_router, retrieval_grader, rag_chain, 
         hallucination_grader, answer_grader, question_rewriter, web_search_tool) = components
        
        # Define test questions for component testing (router 성능 확인용)
        component_test_questions = [
            "What are the types of agent memory?",  # vectorstore question 
            "Who will the Bears draft first in the NFL draft?",  # web search question
            "What is prompt engineering?",  # vectorstore question
            "What happened in the 2024 Olympics?"  # web search question
        ]
        
        # Define questions for full workflow demo (실제 adaptive RAG 동작 확인용)
        workflow_demo_questions = [
            "What are the types of agent memory?",  # vectorstore 경로 테스트
            "Who will the Bears draft first in the NFL draft?"  # web search 경로 테스트
        ]
        
        # Test individual components
        print("\n=== Testing Individual Components ===")
        
        # Test question router
        print("\n1. Testing Question Router:")
        for q in component_test_questions:
            result = question_router.invoke({"question": q})
            print(f"Question: {q}")
            print(f"Route: {result.datasource}\n")
        
        # Test retrieval grader
        print("2. Testing Retrieval Grader:")
        question = "agent memory"
        docs = retriever.invoke(question)
        if docs:
            doc_txt = docs[0].page_content
            grade_result = retrieval_grader.invoke({"question": question, "document": doc_txt})
            print(f"Question: {question}")
            print(f"Document relevance: {grade_result.binary_score}\n")
        
        # Test RAG chain
        print("3. Testing RAG Chain:")
        generation = rag_chain.invoke({"context": docs, "question": question})
        print(f"Generated answer: {generation[:200]}...\n")
        
        # Test hallucination grader
        print("4. Testing Hallucination Grader:")
        hallucination_result = hallucination_grader.invoke({"documents": docs, "generation": generation})
        print(f"Hallucination check: {hallucination_result.binary_score}\n")
        
        # Test answer grader
        print("5. Testing Answer Grader:")
        answer_result = answer_grader.invoke({"question": question, "generation": generation})
        print(f"Answer quality: {answer_result.binary_score}\n")
        
        # Test question rewriter
        print("6. Testing Question Rewriter:")
        rewritten = question_rewriter.invoke({"question": question})
        print(f"Original: {question}")
        print(f"Rewritten: {rewritten}\n")
        
        # Create node functions
        print("Creating workflow nodes...")
        node_functions = create_node_functions(
            retriever, rag_chain, retrieval_grader, question_rewriter, 
            web_search_tool, question_router, hallucination_grader, answer_grader
        )
        
        # Create and compile the graph
        print("Compiling adaptive RAG graph...")
        app = create_adaptive_rag_graph(node_functions)
        
        # Run the demo with workflow-specific questions
        run_adaptive_rag_demo(workflow_demo_questions)
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()