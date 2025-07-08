import os
from dotenv import load_dotenv

from langchain.utils.math import cosine_similarity
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_core.runnables import RunnableLambda


# Two prompts for different domains
physics_template = """You are a very smart physics professor.
You are great at answering questions about physics in a concise and easy to understand manner.
When you don't know the answer to a question you admit that you don't know.

Here is a question:
{query}
"""

math_template = """You are a very good mathematician. You are great at answering math questions.
You are so good because you are able to break down hard problems into their component parts,
answer the component parts, and then put them together to answer the broader question.

Here is a question:
{query}
"""

def prompt_router(input):
    """
    Routes user query to the most appropriate prompt template based on semantic similarity.
    
    Args:
        input: Dictionary containing the user query
        
    Returns:
        PromptTemplate: The most suitable prompt template for the query
    """
    # Embed question to vector representation
    query_embedding = embeddings.embed_query(input["query"])
    
    # Compute cosine similarity with prompt embeddings
    similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
    most_similar = prompt_templates[similarity.argmax()]
    
    # Display chosen prompt type
    template_type = "MATH" if most_similar == math_template else "PHYSICS"
    print(f"🎯 Selected Template: {template_type}")
    
    return PromptTemplate.from_template(most_similar)

def main():
    """
    Main execution function.
    Executes the complete semantic routing process for question answering.
    """
    # Load environment variables
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    
    # Initialize global variables for embeddings
    global embeddings, prompt_templates, prompt_embeddings
    
    print("=" * 80)
    print("🧠 SEMANTIC ROUTING SYSTEM")
    print("=" * 80)
    
    # Initialize embeddings model
    embeddings = OpenAIEmbeddings()
    prompt_templates = [physics_template, math_template]
    
    print("\n⚡ Initializing System...")
    print("-" * 50)
    print("🔗 Loading embeddings model...")
    print("📝 Preparing prompt templates...")
    
    # Generate embeddings for prompt templates
    prompt_embeddings = embeddings.embed_documents(prompt_templates)
    print("✅ Embeddings generated successfully!")
    print("-" * 50)
    
    # Create processing chain - simplified structure
    chain = (
        RunnableLambda(prompt_router)
        | ChatOpenAI(model="gpt-4o-mini", temperature=0)
        | StrOutputParser()
    )
    
    # Test query
    test_query = "What is a black hole?"
    
    print("\n📝 Input Query:")
    print("-" * 50)
    print(f"❓ Question: {test_query}")
    print("-" * 50)
    
    print("\n🤖 Processing Query...")
    print("-" * 50)
    
    # Execute chain and get response
    response = chain.invoke({"query": test_query})
    
    print("\n✅ Generated Response:")
    print("-" * 50)
    print(f"💬 Answer: {response}")
    print("-" * 50)
    
    print("\n" + "=" * 80)
    print("✨ SEMANTIC ROUTING COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    main()