import os
from dotenv import load_dotenv
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

from langchain_core.runnables import RunnableLambda

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["python_docs", "js_docs", "golang_docs"] = Field(
        ...,
        description="Given a user question choose which datasource would be most relevant for answering their question",
    )

def choose_route(result):
    """
    Routes user query to appropriate datasource based on routing result.
    
    Args:
        result: RouteQuery object containing routing result
        
    Returns:
        str: Chain information for the selected datasource
    """
    if "python_docs" in result.datasource.lower():
        ### Logic here 
        return "chain for python_docs"
    elif "js_docs" in result.datasource.lower():
        ### Logic here 
        return "chain for js_docs"
    else:
        ### Logic here 
        return "golang_docs"

def main():
    """
    Main execution function.
    Executes the complete process of routing user questions to appropriate datasource.
    """
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

    # LLM w/ function calling
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    system = """You are an expert at routing a user question to the appropriate data source.
    Based on the programming language the question is reffering to, route it to the relevant data source.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "{question}"),
    ])

    # Define router - Create router combining LLM with structured output
    router = prompt | llm.with_structured_output(RouteQuery)

    # Define test question
    question = """Why dosen't the following code work?:
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages(["human", "speak in {language}"])
    prompt.invoke("french")
    """

    print("=" * 80)
    print("🚀 LOGICAL ROUTING SYSTEM")
    print("=" * 80)
    
    print("\n📝 Input Question:")
    print("-" * 50)
    print(f"{question}")
    print("-" * 50)

    # Execute router and print results
    result = router.invoke({"question": question})

    print("\n🎯 Routing Analysis:")
    print("-" * 50)
    print(f"📊 Full Result: {result}")
    print(f"🎯 Selected Datasource: {result.datasource}")
    print("-" * 50)

    # Create full chain - Connect router with selection function
    full_chain = router | RunnableLambda(choose_route)

    # Execute full chain
    final_result = full_chain.invoke({"question": question})
    
    print("\n✅ Final Output:")
    print("-" * 50)
    print(f"🔗 Chain Result: {final_result}")
    print("-" * 50)
    
    print("\n" + "=" * 80)
    print("✨ ROUTING COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    main()
