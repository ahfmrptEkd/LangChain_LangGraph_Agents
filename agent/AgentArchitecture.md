# 🤖 LangGraph Agent Architecture 완전 가이드

> LangGraph에서 제공하는 다양한 Agent 아키텍처 패턴과 구현 방법

*참고: [LangGraph Agent Architectures](https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/)*

## 🎯 Agent의 정의

**Agent는 LLM이 애플리케이션의 제어 흐름(control flow)을 결정하는 시스템이다.**

### 기존 방식 vs Agent 방식

**기존 방식 (고정된 제어 흐름)**:
```
사용자 질문 → 문서 검색 → LLM 호출 → 답변 생성
```

**Agent 방식 (동적 제어 흐름)**:
```
사용자 질문 → LLM이 판단 → [검색 / 계산 / 직접 답변] → 필요시 추가 작업 → 최종 답변
```

### 🔧 LLM이 제어할 수 있는 영역

1. **경로 선택**: 두 개 이상의 잠재적 경로 중 선택
2. **도구 선택**: 여러 도구 중 어떤 것을 호출할지 결정
3. **완성도 판단**: 생성된 답변이 충분한지, 더 작업이 필요한지 결정

## 📋 Agent Architecture 유형 개요

![agent_types](../src/imgs/agent_types.png)

| 아키텍처 | 제어 수준 | 복잡도 | 주요 용도 |
|---------|----------|-------|----------|
| **Router** | 제한적 | 낮음 | 단일 결정, 분기 처리 |
| **Tool-calling Agent** | 중간 | 중간 | 다단계 작업, 도구 사용 |
| **Custom Architecture** | 높음 | 높음 | 복잡한 워크플로우, 특수 요구사항 |

---

## 🧭 1. Router Architecture

### 특징
- **단일 결정**: LLM이 미리 정의된 옵션 중 하나를 선택
- **제한된 제어**: 비교적 간단한 분기 처리
- **빠른 처리**: 단일 LLM 호출로 결정

### 핵심 개념: Structured Output

Router는 **구조화된 출력**을 통해 작동한다:

```python
from pydantic import BaseModel
from typing import Literal

class RouteChoice(BaseModel):
    """라우팅 결정을 위한 구조화된 출력"""
    datasource: Literal["web_search", "vectorstore", "direct_answer"]
    reasoning: str

# LLM에게 구조화된 출력 요구
llm_with_structured_output = llm.with_structured_output(RouteChoice)
```

### 구조화된 출력 구현 방법

1. **프롬프트 엔지니어링**
```python
system_prompt = """
다음 중 하나로 답변하세요:
- web_search: 최신 정보가 필요한 경우
- vectorstore: 문서 검색이 필요한 경우  
- direct_answer: 일반 지식으로 답변 가능한 경우
"""
```

2. **출력 파서 사용**
```python
from langchain.output_parsers import PydanticOutputParser

parser = PydanticOutputParser(pydantic_object=RouteChoice)
```

3. **Tool Calling 활용**
```python
# LLM의 내장 tool calling 기능 사용
structured_llm = llm.with_structured_output(RouteChoice)
```

### Router 사용 사례
- ✅ **질문 분류**: 고객 문의를 부서별로 라우팅
- ✅ **데이터 소스 선택**: 질문에 따라 적절한 DB 선택
- ✅ **처리 방식 결정**: 간단한 답변 vs 복잡한 분석

---

## 🛠️ 2. Tool-calling Agent (ReAct)

### ReAct 아키텍처의 3가지 핵심 요소

![tool_calling](../src/imgs/tool_call.png)

#### 🔧 Tool Calling
- **외부 시스템과의 상호작용**
- API 호출, 데이터베이스 쿼리, 계산 등
- LLM이 필요한 도구를 선택하고 적절한 입력 제공

```python
from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """수학 계산을 수행합니다."""
    try:
        result = eval(expression)
        return f"계산 결과: {result}"
    except:
        return "계산 오류가 발생했습니다."

@tool  
def web_search(query: str) -> str:
    """웹에서 정보를 검색합니다."""
    # 실제 검색 로직 구현
    return f"'{query}'에 대한 검색 결과..."

# 도구를 LLM에 바인딩
llm_with_tools = llm.bind_tools([calculator, web_search])
```

#### 🧠 Memory
메모리는 다중 단계 문제 해결에서 정보를 보존한다:

**단기 메모리**:
```python
# 현재 세션 내에서 이전 단계의 정보 유지
state = {
    "messages": [...],  # 대화 히스토리
    "intermediate_results": {...}  # 중간 계산 결과
}
```

**장기 메모리**:
```python
# 세션 간 정보 보존
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "user_123"}}
```

#### 📋 Planning
- **반복적 의사결정**: while-loop에서 LLM을 반복 호출
- **도구 선택**: 각 단계에서 어떤 도구를 사용할지 결정
- **종료 조건**: 충분한 정보를 얻었다고 판단하면 종료

```python
def planning_loop(initial_question):
    """Agent의 계획 및 실행 루프"""
    current_state = {"question": initial_question, "steps": []}
    
    while not is_task_complete(current_state):
        # 1. 현재 상황 분석
        analysis = llm.invoke(f"현재 상황: {current_state}")
        
        # 2. 다음 행동 결정
        next_action = decide_next_action(analysis)
        
        # 3. 도구 실행
        if next_action["type"] == "tool_call":
            result = execute_tool(next_action["tool"], next_action["args"])
            current_state["steps"].append(result)
        
        # 4. 완료 여부 확인
        if next_action["type"] == "finish":
            break
    
    return generate_final_answer(current_state)
```

### LangGraph에서 Tool-calling Agent 구현

```python
from langgraph.prebuilt import create_react_agent

# 사전 구축된 ReAct Agent 사용
agent = create_react_agent(
    model=llm,
    tools=[calculator, web_search],
    checkpointer=memory
)

# 실행
config = {"configurable": {"thread_id": "agent_session"}}
result = agent.invoke(
    {"messages": [("user", "2024년 한국 GDP는 얼마이고, 이를 인구로 나눈 1인당 GDP는?")]},
    config
)
```

---

## 🎨 3. Custom Agent Architectures

복잡한 작업을 위해서는 맞춤형 아키텍처가 필요합니다.

### 👤 Human-in-the-loop

인간의 개입이 필요한 상황들:

**승인이 필요한 작업**:
```python
def approval_required_node(state):
    """중요한 결정 전 인간 승인 요청"""
    if state["action_type"] == "high_risk":
        # 인간의 승인 대기
        approval = input(f"다음 작업을 승인하시겠습니까? {state['proposed_action']}")
        if approval.lower() != 'yes':
            return {"status": "cancelled"}
    
    return execute_action(state["proposed_action"])
```

**피드백 제공**:
```python
def feedback_node(state):
    """인간이 Agent 상태에 피드백 제공"""
    current_progress = summarize_progress(state)
    feedback = input(f"현재 진행상황: {current_progress}\n피드백을 입력하세요: ")
    
    return {"feedback": feedback, "human_guidance": True}
```

### ⚡ Parallelization

병렬 처리를 통한 효율성 향상:

```python
from langgraph.constants import Send

def parallel_processing_node(state):
    """여러 작업을 병렬로 처리"""
    tasks = state["pending_tasks"]
    
    # 각 작업을 별도 노드로 병렬 실행
    parallel_sends = []
    for task in tasks:
        parallel_sends.append(
            Send("process_single_task", {"task": task, "task_id": task["id"]})
        )
    
    return parallel_sends

def process_single_task(state):
    """개별 작업 처리"""
    task = state["task"]
    result = execute_task(task)
    return {"task_result": result, "task_id": state["task_id"]}
```

### 📊 Map-Reduce 패턴

```python
def map_reduce_workflow():
    """Map-Reduce 패턴 구현"""
    workflow = StateGraph(...)
    
    # Map 단계: 데이터를 청크로 분할하여 병렬 처리
    workflow.add_node("map_phase", map_data_chunks)
    
    # Reduce 단계: 결과를 집계
    workflow.add_node("reduce_phase", aggregate_results)
    
    workflow.add_edge("map_phase", "reduce_phase")
    return workflow.compile()
```

### 🏗️ Subgraphs

복잡한 멀티 에이전트 시스템을 위한 계층적 구조:

```python
def create_specialist_subgraph():
    """전문 분야별 서브그래프"""
    subgraph = StateGraph(...)
    
    # 전문가 에이전트들
    subgraph.add_node("data_analyst", data_analysis_node)
    subgraph.add_node("researcher", research_node)
    subgraph.add_node("writer", writing_node)
    
    return subgraph.compile()

def main_coordinator_graph():
    """메인 조정자 그래프"""
    main_graph = StateGraph(...)
    
    # 서브그래프를 노드로 포함
    specialist_graph = create_specialist_subgraph()
    main_graph.add_node("specialists", specialist_graph)
    
    return main_graph.compile()
```

**상태 공유 메커니즘**:
```python
class MainState(TypedDict):
    user_query: str
    overall_progress: str
    specialist_results: dict  # 서브그래프와 공유

class SpecialistState(TypedDict):
    user_query: str  # 메인 그래프와 공유  
    specialist_results: dict  # 메인 그래프와 공유
    internal_analysis: str  # 서브그래프 전용
```

### 🔄 Reflection

자체 평가 및 개선 메커니즘:

**LLM 기반 반성**:
```python
def reflection_node(state):
    """LLM을 사용한 자체 평가"""
    current_answer = state["generated_answer"]
    original_question = state["user_question"]
    
    reflection_prompt = f"""
    질문: {original_question}
    생성된 답변: {current_answer}
    
    이 답변을 평가하세요:
    1. 질문에 완전히 답했는가?
    2. 정확한가?
    3. 개선이 필요한 부분은?
    """
    
    evaluation = llm.invoke(reflection_prompt)
    
    if "개선 필요" in evaluation.content:
        return {"needs_revision": True, "feedback": evaluation.content}
    else:
        return {"needs_revision": False, "final_answer": current_answer}
```

**결정론적 반성** (코딩 예시):
```python
def code_validation_node(state):
    """코드 컴파일/실행을 통한 검증"""
    generated_code = state["generated_code"]
    
    try:
        # 코드 실행 시도
        exec(generated_code)
        return {"code_valid": True, "code": generated_code}
    except Exception as e:
        # 오류 발생 시 피드백 제공
        return {
            "code_valid": False, 
            "error_feedback": str(e),
            "needs_fix": True
        }
```

---

## 🎯 Architecture 선택 가이드

### Router 선택 기준
- ✅ **단순한 분기 처리**가 필요한 경우
- ✅ **빠른 응답**이 중요한 경우
- ✅ **명확한 선택지**가 제한적인 경우

### Tool-calling Agent 선택 기준  
- ✅ **다단계 작업**이 필요한 경우
- ✅ **외부 도구 활용**이 필요한 경우
- ✅ **동적 의사결정**이 중요한 경우

### Custom Architecture 선택 기준
- ✅ **복잡한 워크플로우**가 필요한 경우
- ✅ **특수한 요구사항**이 있는 경우
- ✅ **높은 성능**이 필요한 경우

## 🔧 실제 구현 패턴

### 점진적 복잡도 증가

```python
# 1단계: Router로 시작
simple_router = create_router_agent(...)

# 2단계: Tool-calling Agent로 확장  
tool_agent = create_react_agent(model, tools, checkpointer)

# 3단계: Custom Architecture로 고도화
custom_agent = StateGraph(CustomState)
custom_agent.add_node("analysis", analysis_node)
custom_agent.add_node("planning", planning_node)  
custom_agent.add_node("execution", execution_node)
custom_agent.add_node("reflection", reflection_node)
```

### 하이브리드 접근

```python
def hybrid_agent_architecture():
    """여러 아키텍처를 결합한 하이브리드 시스템"""
    main_workflow = StateGraph(...)
    
    # Router로 초기 분류
    main_workflow.add_node("router", routing_node)
    
    # 복잡한 작업은 Tool-calling Agent로
    main_workflow.add_node("complex_agent", tool_calling_subgraph)
    
    # 단순한 작업은 직접 처리
    main_workflow.add_node("simple_response", direct_response_node)
    
    # 조건부 분기
    main_workflow.add_conditional_edges(
        "router",
        route_decision,
        {
            "complex": "complex_agent",
            "simple": "simple_response"
        }
    )
    
    return main_workflow.compile()
```

## 🎉 결론

LangGraph는 다양한 복잡도의 Agent Architecture를 지원한다:

- **Router**: 간단한 분기 처리
- **Tool-calling Agent**: 범용적인 다단계 작업
- **Custom Architecture**: 특수 요구사항에 맞춘 고도화

> 💡 **핵심 원칙**: 가장 간단한 아키텍처(Router)부터 시작하여 요구사항에 따라 점진적으로 복잡도를 높여가자.

각 아키텍처는 고유한 장단점이 있으므로, 프로젝트의 요구사항과 복잡도에 맞는 적절한 선택이 중요하다.

## 📚 참고 자료

- [LangGraph Agent Architectures](https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/)
- [LangGraph Structured Outputs Guide](https://langchain-ai.github.io/langgraph/how-tos/structured_output/)
- [Human-in-the-loop Guide](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/)
- [Map-Reduce Tutorial](https://langchain-ai.github.io/langgraph/tutorials/map-reduce/)
- [Subgraph How-to Guide](https://langchain-ai.github.io/langgraph/how-tos/subgraph/) 