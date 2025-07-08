import os
from dotenv import load_dotenv
from typing import List, Dict, Any
from typing_extensions import TypedDict

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser

# LangGraph imports
from langgraph.graph import END, StateGraph, START


class GraphState(TypedDict):
    """Represents the state of our Self-RAG graph.

    Attributes:
        question (str): User's question
        generation (str): LLM generation result
        documents (List[str]): List of retrieved documents
        retry_count (int): Number of retry attempts
        confidence_score (float): Confidence score of the final answer
        evaluation_results (Dict): Results of self-evaluation
    """
    question: str
    generation: str
    documents: List[str]
    retry_count: int
    confidence_score: float
    evaluation_results: Dict[str, Any]


class FactualAccuracy(BaseModel):
    """Binary score for factual accuracy check on generated answer.
    
    Attributes:
        binary_score (str): Answer is factually accurate based on documents, 'yes' or 'no'
        confidence (float): Confidence level (0.0 to 1.0)
        reasoning (str): Explanation for the score
    """
    binary_score: str = Field(
        description="Answer is factually accurate based on documents, 'yes' or 'no'"
    )
    confidence: float = Field(
        description="Confidence level from 0.0 to 1.0"
    )
    reasoning: str = Field(
        description="Brief explanation for the factual accuracy score"
    )


class QuestionRelevance(BaseModel):
    """Binary score for question relevance check on generated answer.
    
    Attributes:
        binary_score (str): Answer addresses the question properly, 'yes' or 'no'
        confidence (float): Confidence level (0.0 to 1.0)
        reasoning (str): Explanation for the score
    """
    binary_score: str = Field(
        description="Answer addresses the question properly, 'yes' or 'no'"
    )
    confidence: float = Field(
        description="Confidence level from 0.0 to 1.0"
    )
    reasoning: str = Field(
        description="Brief explanation for the question relevance score"
    )


class AnswerCompleteness(BaseModel):
    """Binary score for answer completeness check.
    
    Attributes:
        binary_score (str): Answer is complete and comprehensive, 'yes' or 'no'
        confidence (float): Confidence level (0.0 to 1.0)
        reasoning (str): Explanation for the score
    """
    binary_score: str = Field(
        description="Answer is complete and comprehensive, 'yes' or 'no'"
    )
    confidence: float = Field(
        description="Confidence level from 0.0 to 1.0"
    )
    reasoning: str = Field(
        description="Brief explanation for the completeness score"
    )


class SelfRAGWorkflow:
    """Self-RAG workflow implementation class.
    
    This class implements the Self-RAG workflow that:
    1. Retrieves documents from vector store
    2. Generates initial answer
    3. Performs self-evaluation on multiple criteria
    4. Decides whether to retry or finalize answer
    5. Provides confidence score with final answer
    """
    
    def __init__(self):
        """Initialize Self-RAG workflow with necessary components."""
        print("🚀 Initializing Self-RAG workflow...")
        
        # Initialize components
        self.retriever = None
        self.rag_chain = None
        self.factual_grader = None
        self.relevance_grader = None
        self.completeness_grader = None
        self.compiled_workflow = None
        self.max_retries = 3
        
        print("✅ Self-RAG workflow initialized")
    
    def setup_document_retrieval(self, urls: List[str]):
        """Set up document loading and vector store for retrieval.
        
        Args:
            urls (List[str]): List of URLs to load documents from
        """
        print("🔄 Setting up document retrieval system...")
        
        # Load documents from URLs
        print(f"📄 Loading documents from {len(urls)} URLs...")
        docs = [WebBaseLoader(url).load() for url in urls]
        docs_list = [item for sublist in docs for item in sublist]
        print(f"✅ Loaded {len(docs_list)} documents")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=500, 
            chunk_overlap=50
        )
        doc_splits = text_splitter.split_documents(docs_list)
        print(f"✅ Split into {len(doc_splits)} chunks")
        
        # Create vector store
        print("🔄 Creating vector store...")
        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name="self-rag-chroma",
            embedding=OpenAIEmbeddings(),
        )
        self.retriever = vectorstore.as_retriever()
        print("✅ Vector store and retriever created")
    
    def setup_rag_chain(self):
        """Set up the RAG chain for answer generation."""
        print("🔄 Setting up RAG chain...")
        
        # Pull RAG prompt from hub
        prompt = hub.pull("rlm/rag-prompt")
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        
        self.rag_chain = prompt | llm | StrOutputParser()
        print("✅ RAG chain setup complete")
    
    def setup_self_evaluation_graders(self):
        """Set up all self-evaluation graders."""
        print("🔄 Setting up self-evaluation graders...")
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # Factual accuracy grader
        factual_llm = llm.with_structured_output(FactualAccuracy)
        factual_system = """You are a strict fact-checker evaluating whether an answer is factually accurate based on the provided documents.
        
        Evaluate:
        1. Are all facts in the answer supported by the documents?
        2. Are there any factual errors or unsupported claims?
        3. Is the answer grounded in the provided evidence?
        
        Give a binary score 'yes' or 'no', confidence level, and brief reasoning."""
        
        factual_prompt = ChatPromptTemplate.from_messages([
            ("system", factual_system),
            ("human", "Documents: \n\n {documents} \n\n Answer: {generation} \n\n Evaluate factual accuracy:"),
        ])
        
        self.factual_grader = factual_prompt | factual_llm
        
        # Question relevance grader
        relevance_llm = llm.with_structured_output(QuestionRelevance)
        relevance_system = """You are an evaluator checking whether an answer properly addresses the asked question.
        
        Evaluate:
        1. Does the answer directly address the question?
        2. Are all aspects of the question covered?
        3. Is the answer on-topic and relevant?
        
        Give a binary score 'yes' or 'no', confidence level, and brief reasoning."""
        
        relevance_prompt = ChatPromptTemplate.from_messages([
            ("system", relevance_system),
            ("human", "Question: {question} \n\n Answer: {generation} \n\n Evaluate question relevance:"),
        ])
        
        self.relevance_grader = relevance_prompt | relevance_llm
        
        # Answer completeness grader
        completeness_llm = llm.with_structured_output(AnswerCompleteness)
        completeness_system = """You are an evaluator checking whether an answer is complete and comprehensive.
        
        Evaluate:
        1. Is the answer comprehensive enough?
        2. Are important aspects missing?
        3. Does it provide sufficient detail?
        
        Give a binary score 'yes' or 'no', confidence level, and brief reasoning."""
        
        completeness_prompt = ChatPromptTemplate.from_messages([
            ("system", completeness_system),
            ("human", "Question: {question} \n\n Answer: {generation} \n\n Evaluate answer completeness:"),
        ])
        
        self.completeness_grader = completeness_prompt | completeness_llm
        
        print("✅ Self-evaluation graders setup complete")
    
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
        
        return {
            "documents": documents, 
            "question": question,
            "retry_count": state.get("retry_count", 0)
        }
    
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
        retry_count = state.get("retry_count", 0)
        
        # Generate answer using RAG chain
        generation = self.rag_chain.invoke({"context": documents, "question": question})
        print(f"✅ Answer generated (attempt {retry_count + 1})")
        
        return {
            "documents": documents, 
            "question": question, 
            "generation": generation,
            "retry_count": retry_count
        }
    
    def self_evaluate(self, state):
        """Perform comprehensive self-evaluation of the generated answer.
        
        Args:
            state (dict): Current graph state with all information
            
        Returns:
            dict: Updated state with evaluation results and confidence score
        """
        print("---SELF EVALUATE---")
        question = state["question"]
        generation = state["generation"]
        documents = state["documents"]
        retry_count = state.get("retry_count", 0)
        
        # Perform all evaluations
        print("🔍 Evaluating factual accuracy...")
        factual_result = self.factual_grader.invoke({
            "documents": documents, 
            "generation": generation
        })
        
        print("🔍 Evaluating question relevance...")
        relevance_result = self.relevance_grader.invoke({
            "question": question, 
            "generation": generation
        })
        
        print("🔍 Evaluating answer completeness...")
        completeness_result = self.completeness_grader.invoke({
            "question": question, 
            "generation": generation
        })
        
        # Store evaluation results
        evaluation_results = {
            "factual_accuracy": {
                "score": factual_result.binary_score,
                "confidence": factual_result.confidence,
                "reasoning": factual_result.reasoning
            },
            "question_relevance": {
                "score": relevance_result.binary_score,
                "confidence": relevance_result.confidence,
                "reasoning": relevance_result.reasoning
            },
            "answer_completeness": {
                "score": completeness_result.binary_score,
                "confidence": completeness_result.confidence,
                "reasoning": completeness_result.reasoning
            }
        }
        
        # Calculate overall confidence score
        scores = [
            1.0 if factual_result.binary_score == "yes" else 0.0,
            1.0 if relevance_result.binary_score == "yes" else 0.0,
            1.0 if completeness_result.binary_score == "yes" else 0.0
        ]
        
        confidences = [
            factual_result.confidence,
            relevance_result.confidence,
            completeness_result.confidence
        ]
        
        # Weighted average (factual accuracy has higher weight)
        weights = [0.5, 0.3, 0.2]  # factual, relevance, completeness
        overall_score = sum(s * w for s, w in zip(scores, weights))
        overall_confidence = sum(c * w for c, w in zip(confidences, weights))
        
        final_confidence = overall_score * overall_confidence
        
        print("📊 Evaluation Results:")
        print(f"   Factual Accuracy: {factual_result.binary_score} (conf: {factual_result.confidence:.2f})")
        print(f"   Question Relevance: {relevance_result.binary_score} (conf: {relevance_result.confidence:.2f})")
        print(f"   Answer Completeness: {completeness_result.binary_score} (conf: {completeness_result.confidence:.2f})")
        print(f"   Overall Confidence: {final_confidence:.2f}")
        
        return {
            "documents": documents,
            "question": question,
            "generation": generation,
            "retry_count": retry_count,
            "confidence_score": final_confidence,
            "evaluation_results": evaluation_results
        }
    
    def decide_to_retry(self, state):
        """Decide whether to retry generation or finalize answer.
        
        Args:
            state (dict): Current graph state
            
        Returns:
            str: Next node to call ("retry" or "finalize")
        """
        print("---DECIDE TO RETRY---")
        
        confidence_score = state["confidence_score"]
        retry_count = state["retry_count"]
        evaluation_results = state["evaluation_results"]
        
        # Check if we should retry
        should_retry = False
        
        # Retry if confidence is too low
        if confidence_score < 0.7:
            should_retry = True
            print(f"   Low confidence score: {confidence_score:.2f}")
        
        # Retry if factual accuracy is "no"
        if evaluation_results["factual_accuracy"]["score"] == "no":
            should_retry = True
            print("   Factual accuracy failed")
        
        # Don't retry if max retries reached
        if retry_count >= self.max_retries:
            should_retry = False
            print(f"   Max retries reached: {retry_count}")
        
        if should_retry:
            print("---DECISION: RETRY GENERATION---")
            return "retry"
        else:
            print("---DECISION: FINALIZE ANSWER---")
            return "finalize"
    
    def retry_generation(self, state):
        """Retry generation with improved strategy.
        
        Args:
            state (dict): Current graph state
            
        Returns:
            dict: Updated state with incremented retry count
        """
        print("---RETRY GENERATION---")
        
        retry_count = state["retry_count"] + 1
        question = state["question"]
        documents = state["documents"]
        evaluation_results = state["evaluation_results"]
        
        print(f"🔄 Retry attempt {retry_count}")
        
        # Analyze what went wrong and adjust strategy
        issues = []
        if evaluation_results["factual_accuracy"]["score"] == "no":
            issues.append("factual accuracy")
        if evaluation_results["question_relevance"]["score"] == "no":
            issues.append("question relevance")
        if evaluation_results["answer_completeness"]["score"] == "no":
            issues.append("answer completeness")
        
        if issues:
            print(f"   Issues to address: {', '.join(issues)}")
        
        # Enhanced prompt for retry
        enhanced_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant. Based on the previous evaluation feedback, please provide a more accurate, relevant, and complete answer. 
            
            Previous issues to address: {issues}
            
            Focus on:
            1. Factual accuracy: Only use information from the provided documents
            2. Question relevance: Directly address all aspects of the question
            3. Completeness: Provide comprehensive coverage of the topic
            
            Use the following documents to answer the question:
            {context}"""),
            ("human", "{question}")
        ])
        
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        enhanced_chain = enhanced_prompt | llm | StrOutputParser()
        
        # Generate improved answer
        generation = enhanced_chain.invoke({
            "context": documents,
            "question": question,
            "issues": ", ".join(issues) if issues else "general improvement"
        })
        
        print("✅ Retry generation complete")
        
        return {
            "documents": documents,
            "question": question,
            "generation": generation,
            "retry_count": retry_count,
            "evaluation_results": evaluation_results
        }
    
    def finalize_answer(self, state):
        """Finalize the answer with confidence score.
        
        Args:
            state (dict): Current graph state
            
        Returns:
            dict: Final state with formatted answer
        """
        print("---FINALIZE ANSWER---")
        
        generation = state["generation"]
        confidence_score = state["confidence_score"]
        evaluation_results = state["evaluation_results"]
        retry_count = state["retry_count"]
        
        # Format final answer with confidence information
        final_answer = f"""
{generation}

---
Confidence Score: {confidence_score:.2f}/1.0

Evaluation Details:
• Factual Accuracy: {evaluation_results['factual_accuracy']['score']} (confidence: {evaluation_results['factual_accuracy']['confidence']:.2f})
  Reasoning: {evaluation_results['factual_accuracy']['reasoning']}

• Question Relevance: {evaluation_results['question_relevance']['score']} (confidence: {evaluation_results['question_relevance']['confidence']:.2f})
  Reasoning: {evaluation_results['question_relevance']['reasoning']}

• Answer Completeness: {evaluation_results['answer_completeness']['score']} (confidence: {evaluation_results['answer_completeness']['confidence']:.2f})
  Reasoning: {evaluation_results['answer_completeness']['reasoning']}

Generation attempts: {retry_count + 1}
"""
        
        print(f"✅ Answer finalized with confidence: {confidence_score:.2f}")
        
        return {
            "documents": state["documents"],
            "question": state["question"],
            "generation": final_answer,
            "retry_count": retry_count,
            "confidence_score": confidence_score,
            "evaluation_results": evaluation_results
        }
    
    def build_workflow(self):
        """Build the Self-RAG workflow graph."""
        print("🔄 Building Self-RAG workflow graph...")
        
        # Create workflow
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("generate", self.generate)
        workflow.add_node("self_evaluate", self.self_evaluate)
        workflow.add_node("retry_generation", self.retry_generation)
        workflow.add_node("finalize_answer", self.finalize_answer)
        
        # Add edges
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "self_evaluate")
        workflow.add_conditional_edges(
            "self_evaluate",
            self.decide_to_retry,
            {
                "retry": "retry_generation",
                "finalize": "finalize_answer"
            }
        )
        workflow.add_edge("retry_generation", "self_evaluate")
        workflow.add_edge("finalize_answer", END)
        
        # Compile workflow
        self.compiled_workflow = workflow.compile()
        print("✅ Self-RAG workflow graph built and compiled")
    
    def run_workflow(self, question: str):
        """Run the complete Self-RAG workflow for a given question.
        
        Args:
            question (str): Question to process
            
        Returns:
            dict: Final result with self-evaluated answer
        """
        print(f"\n🔍 Running Self-RAG workflow for question: '{question}'")
        print("="*80)
        
        # Run workflow
        result = self.compiled_workflow.invoke({
            "question": question,
            "retry_count": 0
        })
        
        print("\n" + "="*80)
        print("📋 SELF-RAG WORKFLOW RESULT:")
        print("="*80)
        print(f"Question: {question}")
        print(f"Final Answer:\n{result.get('generation', 'No answer generated')}")
        print("="*80)
        
        return result
    
    def run_streaming_workflow(self, question: str):
        """Run the Self-RAG workflow with streaming output.
        
        Args:
            question (str): Question to process
            
        Returns:
            dict: Final result with self-evaluated answer
        """
        print(f"\n🔍 Running Self-RAG workflow (streaming) for question: '{question}'")
        print("="*80)
        
        inputs = {"question": question, "retry_count": 0}
        final_result = None
        
        # Stream workflow execution
        for output in self.compiled_workflow.stream(inputs):
            for key, value in output.items():
                print(f"📍 Node '{key}' completed")
                if key == "self_evaluate" and "confidence_score" in value:
                    print(f"   Current confidence: {value['confidence_score']:.2f}")
                final_result = value
        
        print("\n" + "="*80)
        print("📋 FINAL SELF-RAG RESULT:")
        print("="*80)
        if final_result and "generation" in final_result:
            print(f"Final Answer:\n{final_result['generation']}")
        else:
            print("No final answer generated")
        print("="*80)
        
        return final_result


def main():
    """Main function to execute the Self-RAG pipeline.
    
    This function demonstrates the complete Self-RAG workflow including:
    1. Document loading and vector store setup
    2. Answer generation with self-evaluation
    3. Iterative improvement based on evaluation
    4. Confidence scoring and detailed feedback
    """
    print("🚀 Starting Self-RAG (Self-Reflective RAG) pipeline...")
    
    # Load environment variables
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    
    # Initialize Self-RAG workflow
    self_rag = SelfRAGWorkflow()
    
    # Step 1: Setup document retrieval
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]
    self_rag.setup_document_retrieval(urls)
    
    # Step 2: Setup all components
    self_rag.setup_rag_chain()
    self_rag.setup_self_evaluation_graders()
    
    # Step 3: Build workflow
    self_rag.build_workflow()
    
    # Step 4: Test with sample questions
    test_questions = [
        "What are the types of agent memory?",
        "How do LLM agents handle complex reasoning?",
        "What are the main challenges in adversarial attacks on LLMs?",
        "Explain the concept of prompt engineering in detail"
    ]
    
    print("\n🧪 Testing Self-RAG workflow with sample questions...")
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*20} TEST {i}/{len(test_questions)} {'='*20}")
        
        # Run workflow
        result = self_rag.run_workflow(question)
        
        # Show detailed results
        if "confidence_score" in result:
            print("\n📊 Quality Metrics:")
            print(f"   Final Confidence: {result['confidence_score']:.2f}/1.0")
            print(f"   Retry Attempts: {result['retry_count']}")
            
            eval_results = result.get("evaluation_results", {})
            if eval_results:
                print(f"   Factual Accuracy: {eval_results['factual_accuracy']['score']}")
                print(f"   Question Relevance: {eval_results['question_relevance']['score']}")
                print(f"   Answer Completeness: {eval_results['answer_completeness']['score']}")
        
        # Optional: Run streaming version for first question
        if i == 1:
            print(f"\n{'='*20} STREAMING VERSION {'='*20}")
            self_rag.run_streaming_workflow(question)
    
if __name__ == "__main__":
    main() 