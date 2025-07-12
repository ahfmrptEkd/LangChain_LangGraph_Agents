from dotenv import load_dotenv
import bs4
import faiss

from langchain_openai import OpenAIEmbeddings

from langchain.chat_models import init_chat_model

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain_chroma import Chroma

from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph.graph import MessagesState, StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition, create_react_agent

load_dotenv()


class ConversationalRAG:
    """
    A conversational RAG (Retrieval-Augmented Generation) implementation using LangChain.
    
    This class provides both Chain-based and Agent-based RAG capabilities
    with conversation memory and multiple vector store options.
    """
    
    def __init__(self):
        """
        Initialize the ConversationalRAG system.
        Sets up LLM, embeddings, and prepares for vector store configuration.
        """
        self.llm = init_chat_model("gpt-4o-mini", model_provider="openai")
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.vector_store = None
        self.workflow = None
        self.app = None
        self.agent_app = None
        self.memory = MemorySaver()
        
    def setup_vector_store(self, choice: str = "1"):
        """
        Set up the vector store based on user choice.
        
        Args:
            choice: Vector store choice ("1" for InMemory, "2" for Chroma, "3" for FAISS)
        """
        print("벡터 스토어를 설정하는 중...")
        
        if choice == "1":
            print("InMemory 벡터 스토어 사용")
            self.vector_store = InMemoryVectorStore(self.embeddings)
        elif choice == "2":
            print("Chroma 벡터 스토어 사용")
            self.vector_store = Chroma(
                collection_name="example_collection",
                embedding_function=self.embeddings,
                persist_directory="./chroma_langchain_db",
            )
        elif choice == "3":
            print("FAISS 벡터 스토어 사용")
            faiss_embedding_dim = len(self.embeddings.embed_query("Hello, world!"))
            index = faiss.IndexFlatL2(faiss_embedding_dim)
            self.vector_store = FAISS(
                embedding_function=self.embeddings,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={}
            )
        else:
            print("잘못된 선택입니다. InMemory를 기본값으로 사용합니다.")
            self.vector_store = InMemoryVectorStore(self.embeddings)
    
    def load_documents(self):
        """
        Load and process documents from web source.
        Uses Lilian Weng's agent blog post as the knowledge base.
        """
        print("문서를 로딩하고 처리하는 중...")
        
        # Load documents from web
        loader = WebBaseLoader(
            web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header")
                )
            ),
        )
        docs = loader.load()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        all_splits = text_splitter.split_documents(docs)
        
        # Add documents to vector store
        self.vector_store.add_documents(documents=all_splits)
        print(f"총 {len(all_splits)}개의 문서 청크가 벡터 스토어에 추가되었습니다.")
    
    def create_retrieval_tool(self):
        """
        Create the retrieval tool for document search.
        
        Returns:
            tool: LangChain tool for document retrieval
        """
        @tool(response_format="content_and_artifact")
        def retrieve(query: str):
            """Retrieve information related to a query."""
            retrieved_docs = self.vector_store.similarity_search(query, k=2)
            serialized = "\n\n".join(
                (f"Source: {doc.metadata}\n" f"content: {doc.page_content}")
                for doc in retrieved_docs
            )
            return serialized, retrieved_docs
        
        return retrieve
    
    def setup_workflow(self):
        """
        Set up the Chain-based RAG workflow with query processing, retrieval, and generation steps.
        """
        retrieve_tool = self.create_retrieval_tool()
        
        def query_or_response(state: MessagesState):
            """
            Generate tool call for retrieval or respond directly.
            
            Args:
                state: Current conversation state
                
            Returns:
                Dictionary containing the response message
            """
            llm_with_tools = self.llm.bind_tools([retrieve_tool])
            response = llm_with_tools.invoke(state["messages"])
            return {"messages": [response]}
        
        def generate(state: MessagesState):
            """
            Generate answer using retrieved content.
            
            Args:
                state: Current conversation state containing tool messages
                
            Returns:
                Dictionary containing the generated response
            """
            # Get generated ToolMessages
            recent_tool_messages = []
            for message in reversed(state["messages"]):
                if message.type == "tool":
                    recent_tool_messages.append(message)
                else:
                    break
            
            tool_messages = recent_tool_messages[::-1]
            
            # Format into prompt
            docs_content = "\n\n".join(doc.content for doc in tool_messages)
            system_message_content = (
                f"""You are an assistant for question-answering tasks.
                Use the following pieces of retrieved context to answer the question.
                If you don't know the answer, just say that you don't know.
                Do not try to make up an answer.
                Use three sentences maximum and keep the answer concise.
                \n\n
                {docs_content}
                """
            )
            
            # Filter conversation messages
            conversation_messages = [
                message
                for message in state["messages"]
                if message.type in ("human", "system")
                or (message.type == "ai" and not message.tool_calls)
            ]
            
            prompt = [SystemMessage(content=system_message_content)] + conversation_messages
            response = self.llm.invoke(prompt)
            return {"messages": [response]}
        
        # Create workflow
        self.workflow = StateGraph(MessagesState)
        tools = ToolNode([retrieve_tool])
        
        # Add nodes
        self.workflow.add_node("query_or_response", query_or_response)
        self.workflow.add_node("tools", tools)
        self.workflow.add_node("generate", generate)
        
        # Add edges
        self.workflow.add_edge(START, "query_or_response")
        self.workflow.add_conditional_edges(
            "query_or_response",
            tools_condition,
            {
                END: END,
                "tools": "tools"
            }
        )
        self.workflow.add_edge("tools", "generate")
        self.workflow.add_edge("generate", END)
        
        # Compile workflow
        self.app = self.workflow.compile(checkpointer=self.memory)
    
    def setup_agent_workflow(self):
        """
        Set up the Agent-based RAG workflow.
        Agent can decide when to search and how many times to search.
        """
        retrieve_tool = self.create_retrieval_tool()
        
        # System message for the agent
        system_message = """You are a helpful assistant with access to a document retrieval tool.
        
        When a user asks a question:
        1. If the question is about general topics or common knowledge, you can answer directly without searching.
        2. If the question requires specific information from the documents (especially about AI agents, task decomposition, planning, memory, etc.), use the retrieve tool to find relevant information.
        3. You can use the retrieve tool multiple times if needed to get comprehensive information.
        4. After retrieving information, provide a clear and concise answer based on the retrieved content.
        5. If you can't find relevant information in the documents, say so honestly.
        
        Be conversational and helpful while being accurate with the information."""
        
        # Store system message for later use
        self.agent_system_message = system_message
        
        # Create agent using create_react_agent
        self.agent_app = create_react_agent(
            self.llm,
            tools=[retrieve_tool],
            checkpointer=self.memory
        )
    
    def run_basic_test(self):
        """
        Run basic Chain-based RAG functionality test without memory.
        """
        print("\n=== 기본 Chain RAG 테스트 (메모리 없음) ===")
        
        # Create app without memory for basic test
        basic_app = self.workflow.compile()
        
        test_query = "What is Task Decomposition?"
        print(f"\n질문: {test_query}")
        print("답변:")
        
        for step in basic_app.stream(
            {"messages": [{"role": "user", "content": test_query}]},
            stream_mode="values"
        ):
            if step["messages"][-1].type == "ai" and not step["messages"][-1].tool_calls:
                print(f"Bot: {step['messages'][-1].content}")
    
    def run_persistent_test(self):
        """
        Run Chain-based RAG test with conversation memory.
        """
        print("\n=== 지속적 Chain RAG 테스트 (메모리 있음) ===")
        
        config = {"configurable": {"thread_id": "test_chain_rag_conversation"}}
        
        # First question
        query1 = "What is Task Decomposition?"
        print(f"\n1. 첫 번째 질문: {query1}")
        print("답변:")
        
        for step in self.app.stream(
            {"messages": [{"role": "user", "content": query1}]},
            stream_mode="values",
            config=config,
        ):
            if step["messages"][-1].type == "ai" and not step["messages"][-1].tool_calls:
                print(f"Bot: {step['messages'][-1].content}")
        
        # Follow-up question
        query2 = "What are the main challenges with it?"
        print(f"\n2. 후속 질문: {query2}")
        print("답변:")
        
        for step in self.app.stream(
            {"messages": [{"role": "user", "content": query2}]},
            stream_mode="values",
            config=config,
        ):
            if step["messages"][-1].type == "ai" and not step["messages"][-1].tool_calls:
                print(f"Bot: {step['messages'][-1].content}")
    
    def run_agent_test(self):
        """
        Run Agent-based RAG test.
        Agent decides when to search and how many times to search.
        """
        print("\n=== Agent 기반 RAG 테스트 ===")
        
        config = {"configurable": {"thread_id": "test_agent_conversation"}}
        
        # Test questions for agent
        test_questions = [
            "Hello, how are you?",  # General question - should not trigger search
            "What is Task Decomposition?",  # Document-specific - should trigger search
            "Can you tell me more about the challenges?",  # Follow-up - should use context
            "What's the weather like today?",  # Unrelated - should not search
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n{i}. 질문: {question}")
            print("="*60)
            
            # For first question, include system message
            if i == 1:
                messages = [
                    {"role": "system", "content": self.agent_system_message},
                    {"role": "user", "content": question}
                ]
            else:
                messages = [{"role": "user", "content": question}]
            
            # Track agent steps
            step_count = 0
            tool_call_count = 0
            final_response = None
            
            print("🤖 Agent 추론 과정:")
            print("-"*40)
            
            # Stream agent response with detailed steps
            for chunk in self.agent_app.stream(
                {"messages": messages},
                config=config,
                stream_mode="updates"  # Changed to see each step
            ):
                step_count += 1
                
                # Process each node update
                for node_name, node_data in chunk.items():
                    if node_name == "__start__":
                        continue
                        
                    print(f"📍 Step {step_count}: {node_name}")
                    
                    if "messages" in node_data:
                        last_message = node_data["messages"][-1]
                        
                        # Check if this is a tool call
                        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                            tool_call_count += 1
                            tool_name = last_message.tool_calls[0]['name']
                            tool_args = last_message.tool_calls[0]['args']
                            print(f"  🔧 도구 호출 #{tool_call_count}: {tool_name}")
                            print(f"     📋 검색어: {tool_args.get('query', 'N/A')}")
                            
                        # Check if this is a tool result
                        elif hasattr(last_message, 'type') and last_message.type == "tool":
                            print("  📄 도구 결과: 검색 완료")
                            print(f"     📊 문서 발견: {len(last_message.content.split('Source:')) - 1}개")
                            
                        # Check if this is final AI response
                        elif hasattr(last_message, 'content') and last_message.type == "ai" and not (hasattr(last_message, 'tool_calls') and last_message.tool_calls):
                            print("  💭 최종 답변 생성 중...")
                            final_response = last_message
                            
                    print()
            
            # Summary
            print("-"*40)
            print("📊 Agent 활동 요약:")
            print(f"   • 총 단계: {step_count}")
            print(f"   • 도구 호출 횟수: {tool_call_count}")
            print(f"   • 최종 답변: {'생성됨' if final_response else '없음'}")
            print("-"*40)
            
            # Print final response
            if final_response and hasattr(final_response, 'content'):
                print(f"🎯 최종 답변:\n{final_response.content}")
            else:
                print("❌ 응답 없음")
            
            print("="*60)
    
    def interactive_chat(self):
        """
        Start an interactive Chain-based RAG chat session.
        Users can ask questions about the loaded documents.
        """
        print("\n=== 인터랙티브 Chain RAG 채팅 ===")
        print("로딩된 문서에 대해 질문해보세요. 'quit'을 입력하면 종료됩니다.")
        print("(주제: AI Agent의 Task Decomposition, Planning, Memory 등)")
        
        config = {"configurable": {"thread_id": "interactive_chain_rag_session"}}
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '종료', '나가기']:
                print("Chain RAG 채팅 세션이 종료되었습니다. 안녕히 가세요!")
                break
            
            if not user_input:
                continue
            
            try:
                print("답변:")
                for step in self.app.stream(
                    {"messages": [{"role": "user", "content": user_input}]},
                    stream_mode="values",
                    config=config,
                ):
                    if step["messages"][-1].type == "ai" and not step["messages"][-1].tool_calls:
                        print(f"Bot: {step['messages'][-1].content}")
                        
            except Exception as e:
                print(f"오류 발생: {str(e)}")
                print("다시 시도해주세요.")
    
    def interactive_agent_chat(self):
        """
        Start an interactive Agent-based RAG chat session.
        Agent decides when to search documents.
        """
        print("\n=== 인터랙티브 Agent RAG 채팅 ===")
        print("Agent가 스스로 판단해서 검색 여부를 결정합니다.")
        print("일반적인 질문은 직접 답변하고, 문서 관련 질문은 검색을 수행합니다.")
        print("Agent의 추론 과정과 도구 사용을 실시간으로 확인할 수 있습니다.")
        print("'quit'을 입력하면 종료됩니다.")
        
        config = {"configurable": {"thread_id": "interactive_agent_session"}}
        first_message = True
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '종료', '나가기']:
                print("Agent RAG 채팅 세션이 종료되었습니다. 안녕히 가세요!")
                break
            
            if not user_input:
                continue
            
            try:
                print("\n" + "="*50)
                print("🤖 Agent 추론 과정:")
                print("-"*30)
                
                # Include system message for first interaction
                if first_message:
                    messages = [
                        {"role": "system", "content": self.agent_system_message},
                        {"role": "user", "content": user_input}
                    ]
                    first_message = False
                else:
                    messages = [{"role": "user", "content": user_input}]
                
                # Track agent steps
                step_count = 0
                tool_call_count = 0
                final_response = None
                
                # Stream agent response with detailed steps
                for chunk in self.agent_app.stream(
                    {"messages": messages},
                    config=config,
                    stream_mode="updates"
                ):
                    step_count += 1
                    
                    # Process each node update
                    for node_name, node_data in chunk.items():
                        if node_name == "__start__":
                            continue
                            
                        print(f"📍 Step {step_count}: {node_name}")
                        
                        if "messages" in node_data:
                            last_message = node_data["messages"][-1]
                            
                            # Check if this is a tool call
                            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                                tool_call_count += 1
                                tool_name = last_message.tool_calls[0]['name']
                                tool_args = last_message.tool_calls[0]['args']
                                print(f"  🔧 도구 호출 #{tool_call_count}: {tool_name}")
                                print(f"     📋 검색어: {tool_args.get('query', 'N/A')}")
                                
                            # Check if this is a tool result
                            elif hasattr(last_message, 'type') and last_message.type == "tool":
                                print("  📄 도구 결과: 검색 완료")
                                doc_count = len(last_message.content.split('Source:')) - 1
                                print(f"     📊 문서 발견: {doc_count}개")
                                
                            # Check if this is final AI response
                            elif hasattr(last_message, 'content') and last_message.type == "ai" and not (hasattr(last_message, 'tool_calls') and last_message.tool_calls):
                                print("  💭 최종 답변 생성 중...")
                                final_response = last_message
                                
                        print()
                
                # Summary
                print("-"*30)
                print("📊 Agent 활동 요약:")
                print(f"   • 총 단계: {step_count}")
                print(f"   • 도구 호출 횟수: {tool_call_count}")
                print(f"   • 검색 여부: {'검색함' if tool_call_count > 0 else '검색 안함'}")
                print("-"*30)
                
                # Print final response
                if final_response and hasattr(final_response, 'content'):
                    print(f"🎯 최종 답변:\n{final_response.content}")
                else:
                    print("❌ 응답 없음")
                
                print("="*50)
                        
            except Exception as e:
                print(f"오류 발생: {str(e)}")
                print("다시 시도해주세요.")


def main():
    """
    Main function to demonstrate conversational RAG functionality.
    """
    print("🤖 Conversational RAG (검색 증강 생성) 데모")
    print("=" * 50)
    
    # Initialize RAG system
    rag_system = ConversationalRAG()
    
    # Get vector store choice
    print("\n사용할 벡터 스토어를 선택하세요:")
    print("1. InMemory (메모리 내 저장)")
    print("2. Chroma (로컬 디스크 저장)")  
    print("3. FAISS (고성능 검색)")
    
    vector_choice = input("선택 (1/2/3): ").strip()
    if vector_choice not in ["1", "2", "3"]:
        print("잘못된 입력입니다. 기본값(1)을 사용합니다.")
        vector_choice = "1"
    
    # Setup system
    rag_system.setup_vector_store(vector_choice)
    rag_system.load_documents()
    
    # Get RAG type choice
    print("\n사용할 RAG 방식을 선택하세요:")
    print("1. Chain RAG (체인 방식 - 모든 질문에 검색 수행)")
    print("2. Agent RAG (에이전트 방식 - 스스로 판단하여 검색)")
    
    rag_choice = input("선택 (1/2): ").strip()
    
    if rag_choice == "1":
        # Chain-based RAG
        rag_system.setup_workflow()
        
        # Run tests
        rag_system.run_basic_test()
        rag_system.run_persistent_test()
        
        # Ask for interactive session
        print("\n" + "=" * 50)
        user_choice = input("\n인터랙티브 Chain RAG 채팅을 시작하시겠습니까? (y/n): ").strip().lower()
        
        if user_choice in ['y', 'yes', 'ㅇ', '네']:
            rag_system.interactive_chat()
        else:
            print("데모가 완료되었습니다. 감사합니다!")
    
    elif rag_choice == "2":
        # Agent-based RAG
        rag_system.setup_workflow()  # Also setup chain workflow for comparison
        rag_system.setup_agent_workflow()
        
        # Run tests
        rag_system.run_basic_test()  # Chain test for comparison
        rag_system.run_agent_test()  # Agent test
        
        # Ask for interactive session
        print("\n" + "=" * 50)
        user_choice = input("\n인터랙티브 Agent RAG 채팅을 시작하시겠습니까? (y/n): ").strip().lower()
        
        if user_choice in ['y', 'yes', 'ㅇ', '네']:
            rag_system.interactive_agent_chat()
        else:
            print("데모가 완료되었습니다. 감사합니다!")
    
    else:
        print("잘못된 선택입니다. 데모를 종료합니다.")


if __name__ == "__main__":
    main()