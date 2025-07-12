from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage
from langchain.chat_models import init_chat_model

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, MessagesState

load_dotenv()


class ChatBot:
    """
    A chatbot implementation using LangChain and LangGraph.
    
    This class provides both basic chatbot functionality and message persistence
    using LangGraph's state management and memory capabilities.
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini", model_provider: str = "openai"):
        """
        Initialize the ChatBot with specified model.
        
        Args:
            model_name: Name of the chat model to use
            model_provider: Provider of the chat model (e.g., "openai")
        """
        self.model = init_chat_model(model_name, model_provider=model_provider)
        self.workflow = None
        self.app = None
        self.memory = None
        self._setup_persistent_chat()
    
    def _setup_persistent_chat(self):
        """Set up the persistent chat workflow with memory."""
        # Create workflow for message persistence
        self.workflow = StateGraph(state_schema=MessagesState)
        
        def call_model(state: MessagesState):
            """
            Process the current state and generate model response.
            
            Args:
                state: Current conversation state containing messages
                
            Returns:
                Dictionary containing the model's response message
            """
            response = self.model.invoke(state["messages"])
            return {"messages": response}
        
        self.workflow.add_node("call_model", call_model)
        self.workflow.add_edge(START, "call_model")
        
        # Setup memory for conversation persistence
        self.memory = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.memory)
    
    def basic_chat_test(self):
        """
        Run basic chatbot tests without persistence.
        Demonstrates simple model interactions.
        """
        print("=== Basic Chat Test (Without Memory) ===")
        
        # Test 1: Simple greeting
        print("\n1. Simple greeting test:")
        response1 = self.model.invoke([HumanMessage(content="안녕하세요 나는 유저입니다.")])
        print("User: 안녕하세요 나는 유저입니다.")
        print(f"Bot: {response1.content}")
        
        # Test 2: Question without context
        print("\n2. Question without context:")
        response2 = self.model.invoke([HumanMessage(content="내이름은 뭐야?")])
        print("User: 내이름은 뭐야?")
        print(f"Bot: {response2.content}")
        
        # Test 3: Conversation with context
        print("\n3. Conversation with manual context:")
        response3 = self.model.invoke([
            HumanMessage(content="안녕 내 이름은 홍길동이야."),
            AIMessage(content="안녕하세요 홍길동님, 반갑습니다.\n 무엇을 도와드릴까요?"),
            HumanMessage(content="내 이름은 뭐야?")
        ])
        print("User: 안녕 내 이름은 홍길동이야.")
        print("Bot: 안녕하세요 홍길동님, 반갑습니다. 무엇을 도와드릴까요?")
        print("User: 내 이름은 뭐야?")
        print(f"Bot: {response3.content}")
    
    def persistent_chat_test(self):
        """
        Run persistent chat test with memory.
        Demonstrates conversation continuity across multiple interactions.
        """
        print("\n=== Persistent Chat Test (With Memory) ===")
        
        # Configuration for memory persistence
        config = {"configurable": {"thread_id": "test_conversation"}}
        
        # Test conversation with memory
        print("\n1. First message with memory:")
        query1 = "안녕하세요 나는 홍길동이야."
        input_messages1 = [HumanMessage(content=query1)]
        output1 = self.app.invoke({"messages": input_messages1}, config)
        print(f"User: {query1}")
        print(f"Bot: {output1['messages'][-1].content}")
        
        print("\n2. Follow-up question (should remember name):")
        query2 = "내 이름은 뭐야?"
        input_messages2 = [HumanMessage(content=query2)]
        output2 = self.app.invoke({"messages": input_messages2}, config)
        print(f"User: {query2}")
        print(f"Bot: {output2['messages'][-1].content}")
    
    def interactive_chat(self):
        """
        Start an interactive chat session with the user.
        User can type messages and get responses with conversation memory.
        """
        print("\n=== Interactive Chat Session ===")
        print("Type your messages below. Type 'quit' to exit.")
        
        # Create unique config for this session
        config = {"configurable": {"thread_id": "interactive_session"}}
        
        while True:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            # Check for exit condition
            if user_input.lower() in ['quit', 'exit', '종료', '나가기']:
                print("Chat session ended. Goodbye!")
                break
            
            if not user_input:
                continue
            
            try:
                # Send message to chatbot
                input_messages = [HumanMessage(content=user_input)]
                output = self.app.invoke({"messages": input_messages}, config)
                
                # Display response
                print(f"Bot: {output['messages'][-1].content}")
                
            except Exception as e:
                print(f"Error: {str(e)}")
                print("Please try again.")


def main():
    """
    Main function to demonstrate chatbot functionality.
    Runs basic tests and starts interactive chat session.
    """
    print("🤖 LangChain ChatBot Demo")
    print("=" * 40)
    
    # Initialize chatbot
    chatbot = ChatBot()
    
    # Run basic tests
    chatbot.basic_chat_test()
    
    # Run persistent chat test
    chatbot.persistent_chat_test()
    
    # Start interactive session
    print("\n" + "=" * 40)
    user_choice = input("\nWould you like to start an interactive chat? (y/n): ").strip().lower()
    
    if user_choice in ['y', 'yes', 'ㅇ', '네']:
        chatbot.interactive_chat()
    else:
        print("Demo completed. Thank you!")


if __name__ == "__main__":
    main()