from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from operator import itemgetter


def main():
    """
    Main function to execute Step-back Prompting RAG pipeline.
    
    Step-back prompting generates broader, more generic questions
    to retrieve more comprehensive context for answering specific questions.
    """
    
    # Load environment variables
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
    
    print("🚀 Starting Step-back Prompting RAG Pipeline...\n")
    
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
    
    # 4. Setup Step-back Prompting
    print("🔄 Setting up Step-back Prompting...")
    
    # Few-shot examples for step-back question generation
    examples = [
        {
            "input": "Could the members of The Police perform lawful arrests?",
            "output": "what can the members of The Police do?",
        },
        {
            "input": "Jan Sindel's was born in what country?",
            "output": "what is Jan Sindel's personal history?",
        },
    ]
    print(f"   ✅ Prepared {len(examples)} few-shot examples")

    # Create few-shot prompt template
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )
    
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )
    print("   ✅ Few-shot prompt template created")
    
    # Create step-back prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:""",
            ),
            # Few shot examples
            few_shot_prompt,
            # New question
            ("user", "{question}"),
        ]
    )
    
    # Create step-back question generator
    generate_queries_step_back = prompt | ChatOpenAI(temperature=0) | StrOutputParser()
    print("   ✅ Step-back question generator created\n")
    
    # 5. Generate step-back question
    print("🔍 Generating Step-back Question...")
    question = "What is task decomposition for LLM agents?"
    print(f"   Original question: '{question}'")
    
    step_back_question = generate_queries_step_back.invoke({"question": question})
    print(f"   ✅ Step-back question: '{step_back_question}'\n")
    
    # 6. Retrieve contexts
    print("📖 Retrieving contexts...")
    print("   Retrieving context for original question...")
    normal_context = retriever.invoke(question)
    print(f"   ✅ Retrieved {len(normal_context)} documents for original question")
    
    print("   Retrieving context for step-back question...")
    step_back_context = retriever.invoke(step_back_question)
    print(f"   ✅ Retrieved {len(step_back_context)} documents for step-back question\n")
    
    # 7. Setup response generation
    print("🎯 Setting up Response Generation...")
    response_prompt_template = """You are an expert of world knowledge. I am going to ask you a question. Your response should be comprehensive and not contradicted with the following context if they are relevant. Otherwise, ignore them if they are not relevant.

    # {normal_context}
    # {step_back_context}

    # Original Question: {question}
    # Answer:"""
    response_prompt = ChatPromptTemplate.from_template(response_prompt_template)
    print("   ✅ Response prompt template created")
    
    # Create complete chain
    chain = (
        {
            # Retrieve context using the normal question
            "normal_context": itemgetter("question") | retriever,
            
            # Retrieve context using the step-back question
            "step_back_context": generate_queries_step_back | retriever,
           
            # Pass on the question
            "question": itemgetter("question"),
        }
        | response_prompt
        | ChatOpenAI(temperature=0)
        | StrOutputParser()
    )
    print("   ✅ Complete RAG chain created\n")
    
    # 8. Generate final answer
    print("🎬 Generating Final Answer...")
    final_answer = chain.invoke({"question": question})
    print("   ✅ Final answer generated\n")
    
    # 9. Display results
    print("="*60)
    print("📊 STEP-BACK PROMPTING RESULTS")
    print("="*60)
    
    print(f"Original Question: {question}")
    print(f"Step-back Question: {step_back_question}\n")
    
    print("📋 Context Comparison:")
    print("-" * 40)
    print(f"Normal Context Documents: {len(normal_context)}")
    print(f"Step-back Context Documents: {len(step_back_context)}")
    
    # Show unique vs overlapping documents
    normal_content = set([doc.page_content for doc in normal_context])
    step_back_content = set([doc.page_content for doc in step_back_context])
    unique_normal = normal_content - step_back_content
    unique_step_back = step_back_content - normal_content
    overlap = normal_content & step_back_content
    
    print(f"Unique to Normal: {len(unique_normal)}")
    print(f"Unique to Step-back: {len(unique_step_back)}")
    print(f"Overlapping: {len(overlap)}\n")
    
    print("🎯 Final Answer:")
    print("-" * 40)
    print(final_answer)
    print()
    
    print("💡 Step-back Benefits:")
    print("-" * 40)
    print("• Broader context retrieval through generic questions")
    print("• Better handling of specific, detailed questions")
    print("• More comprehensive background information")
    print("• Improved answer quality through dual-context approach")
    
    print("="*60)
    
    return {
        "original_question": question,
        "step_back_question": step_back_question,
        "normal_context": normal_context,
        "step_back_context": step_back_context,
        "final_answer": final_answer,
        "context_stats": {
            "normal_docs": len(normal_context),
            "step_back_docs": len(step_back_context),
            "unique_normal": len(unique_normal),
            "unique_step_back": len(unique_step_back),
            "overlap": len(overlap)
        }
    }

if __name__ == "__main__":
    main()