from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import bs4
from langchain_community.document_loaders import WebBaseLoader
from operator import itemgetter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import hub

def format_qa_pair(question, answer):
    """Format Q and A pair"""
    formatted_string = ""
    formatted_string += f"Question: {question}\nAnswer: {answer}\n\n"
    return formatted_string.strip()

def retrieve_and_rag(question, prompt_rag, sub_question_generator_chain, retriever, llm):
    """RAG on each sub-question"""
    
    # Use our decomposition  
    sub_questions = sub_question_generator_chain.invoke({"question": question})
    
    # Initialize a list to hold RAG chain results
    rag_results = []
    
    for sub_question in sub_questions:
        # Retrieve documents for each sub-question
        retrieved_docs = retriever.invoke(sub_question)
        
        # Use retrieved documents and sub-question in RAG chain
        answer = (prompt_rag | llm | StrOutputParser()).invoke({
            "context": retrieved_docs, 
            "question": sub_question
        })
        rag_results.append(answer)
    
    return rag_results, sub_questions

def format_qa_pairs(questions, answers):
    """Format Q and A pairs"""
    formatted_string = ""
    for i, (question, answer) in enumerate(zip(questions, answers), start=1):
        formatted_string += f"Question {i}: {question}\nAnswer {i}: {answer}\n\n"
    return formatted_string.strip()

def main():
    """
    Main function to execute Query Decomposition RAG pipeline.
    
    This function demonstrates two approaches:
    1. Recursive Approach: Sequential answering with context accumulation
    2. Individual Approach: Independent answering then synthesis
    """
    
    # Load environment variables
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
    
    print("🚀 Starting Query Decomposition RAG Pipeline...\n")
    
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
    
    # 4. Setup Query Decomposition
    print("🔍 Setting up Query Decomposition...")
    template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
Generate multiple search queries related to: {question} \n

Output (3 queries):"""
    
    prompt_decomposition = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(temperature=0)
    
    generate_queries_decomposition = (
        prompt_decomposition 
        | llm 
        | StrOutputParser() 
        | (lambda x: x.split("\n"))
    )
    print("   ✅ Query decomposition generator created")
    
    # 5. Generate sub-questions
    question = "What are the main components of an LLM-powered autonomous agent system?"
    print(f"   Original question: '{question}'")
    
    sub_questions = generate_queries_decomposition.invoke({"question": question})
    print(f"   ✅ Generated {len(sub_questions)} sub-questions:")
    for i, q in enumerate(sub_questions, 1):
        print(f"      {i}. {q}")
    print()
    
    # 6. Approach 1: Recursive Method
    print("🔄 Approach 1: Recursive Method (Sequential with Context)")
    print("   Processing sub-questions sequentially...")
    
    decomposition_template = """Here is the question you need to answer:

\n --- \n {question} \n --- \n

Here is any available background question + answer pairs:

\n --- \n {q_a_pairs} \n --- \n

Here is additional context relevant to the question: 

\n --- \n {context} \n --- \n

Use the above context and any background question + answer pairs to answer the question: \n {question}
"""
    
    decomposition_prompt = ChatPromptTemplate.from_template(decomposition_template)
    
    q_a_pairs = ""
    recursive_answers = []
    
    for i, q in enumerate(sub_questions):
        rag_chain = (
            {"context": itemgetter("question") | retriever, 
             "question": itemgetter("question"),
             "q_a_pairs": itemgetter("q_a_pairs")} 
            | decomposition_prompt
            | llm
            | StrOutputParser()
        )
        
        answer = rag_chain.invoke({"question": q, "q_a_pairs": q_a_pairs})
        recursive_answers.append(answer)
        
        q_a_pair = format_qa_pair(q, answer)
        q_a_pairs = q_a_pairs + "\n---\n" + q_a_pair
        
        print(f"      ✅ Sub-question {i+1} answered")
    
    print("   ✅ Recursive approach completed\n")
    
    # 7. Approach 2: Individual Method
    print("🔀 Approach 2: Individual Method (Independent then Synthesis)")
    print("   Processing sub-questions independently...")
    
    prompt_rag = hub.pull("rlm/rag-prompt")
    individual_answers, questions = retrieve_and_rag(
        question, prompt_rag, generate_queries_decomposition, retriever, llm
    )
    
    print(f"   ✅ {len(individual_answers)} independent answers generated")
    
    # Synthesis
    print("   Synthesizing individual answers...")
    context = format_qa_pairs(questions, individual_answers)
    
    synthesis_template = """Here is a set of Q+A pairs:

{context}

Use these to synthesize an answer to the question: {question}
"""
    
    synthesis_prompt = ChatPromptTemplate.from_template(synthesis_template)
    
    final_rag_chain = (
        synthesis_prompt
        | llm
        | StrOutputParser()
    )
    
    individual_final_answer = final_rag_chain.invoke({
        "context": context, 
        "question": question
    })
    print("   ✅ Individual approach completed\n")
    
    # 8. Results Comparison
    print("="*60)
    print("📊 QUERY DECOMPOSITION RESULTS COMPARISON")
    print("="*60)
    
    print(f"Original Question: {question}\n")
    
    print("🔄 Recursive Approach (Final Context):")
    print("-" * 40)
    print(recursive_answers[-1][:300] + "..." if len(recursive_answers[-1]) > 300 else recursive_answers[-1])
    print()
    
    print("🔀 Individual Approach (Synthesized):")
    print("-" * 40)
    print(individual_final_answer[:300] + "..." if len(individual_final_answer) > 300 else individual_final_answer)
    print()
    
    print("📋 Sub-questions Used:")
    print("-" * 40)
    for i, q in enumerate(sub_questions, 1):
        print(f"{i}. {q}")
    
    print("="*60)
    
    return {
        "question": question,
        "sub_questions": sub_questions,
        "recursive_answer": recursive_answers[-1],
        "individual_answer": individual_final_answer,
        "all_recursive_answers": recursive_answers,
        "all_individual_answers": individual_answers
    }

if __name__ == "__main__":
    main()