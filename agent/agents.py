from dotenv import load_dotenv
from datetime import datetime
import random

from langchain_tavily import TavilySearch
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from pydantic import BaseModel, Field

load_dotenv()



# ===== Pydantic 모델 정의 =====

class WeatherInfo(BaseModel):
    """날씨 정보를 위한 구조화된 모델"""
    city: str = Field(description="도시명")
    condition: str = Field(description="날씨 상태")
    temperature: int = Field(description="기온 (섭씨)")
    mood: str = Field(description="날씨에 따른 기분")

class CalculationResult(BaseModel):
    """계산 결과를 위한 구조화된 모델"""
    expression: str = Field(description="계산 식")
    result: float = Field(description="계산 결과")
    success: bool = Field(description="계산 성공 여부")

class TextAnalysisResult(BaseModel):
    """텍스트 분석 결과를 위한 구조화된 모델"""
    text: str = Field(description="분석된 텍스트")
    char_count: int = Field(description="전체 문자 수")
    word_count: int = Field(description="단어 수")
    sentence_count: int = Field(description="문장 수")
    korean_chars: int = Field(description="한글 문자 수")
    avg_word_length: float = Field(description="평균 단어 길이")

# ===== Custom Tools 정의 =====

@tool
def calculator(expression: str) -> str:
    """
    수학 계산을 수행합니다.
    
    Args:
        expression: 계산할 수식 (예: "2 + 3 * 4")
    
    Returns:
        계산 결과 (구조화된 형태)
    """
    try:
        # 안전한 계산을 위해 eval 대신 제한적 연산만 허용
        allowed_chars = "0123456789+-*/.() "
        if not all(c in allowed_chars for c in expression):
            result = CalculationResult(
                expression=expression,
                result=0.0,
                success=False
            )
            return f"❌ 오류: 허용되지 않는 문자가 포함되어 있습니다.\n구조화된 결과: {result.model_dump_json()}"
        
        calc_result = eval(expression)
        result = CalculationResult(
            expression=expression,
            result=float(calc_result),
            success=True
        )
        return f"✅ 계산 완료: {expression} = {calc_result}\n구조화된 결과: {result.model_dump_json()}"
    except Exception as e:
        result = CalculationResult(
            expression=expression,
            result=0.0,
            success=False
        )
        return f"❌ 계산 오류: {str(e)}\n구조화된 결과: {result.model_dump_json()}"


@tool
def current_time() -> str:
    """
    현재 한국 시간을 반환합니다.
    
    Returns:
        현재 날짜와 시간
    """
    now = datetime.now()
    return f"📅 현재 시간: {now.strftime('%Y년 %m월 %d일 %H시 %M분 %S초')}"


@tool
def korean_name_generator(count: int = 1) -> str:
    """
    한국어 이름을 생성합니다.
    
    Args:
        count: 생성할 이름 개수 (기본값: 1)
    
    Returns:
        생성된 한국어 이름들
    """
    surnames = ["김", "이", "박", "최", "정", "강", "조", "윤", "장", "임"]
    given_names = ["민준", "서연", "지우", "하은", "도윤", "소율", "시우", "지유", "예준", "채원"]
    
    names = []
    for _ in range(min(count, 10)):  # 최대 10개까지 제한
        surname = random.choice(surnames)
        given_name = random.choice(given_names)
        names.append(f"{surname}{given_name}")
    
    return f"🎯 생성된 한국어 이름: {', '.join(names)}"


@tool
def text_analyzer(text: str) -> str:
    """
    텍스트를 분석하여 통계 정보를 제공합니다.
    
    Args:
        text: 분석할 텍스트
    
    Returns:
        텍스트 분석 결과 (구조화된 형태)
    """
    if not text:
        return "❌ 분석할 텍스트가 없습니다."
    
    # 기본 통계
    char_count = len(text)
    word_count = len(text.split())
    sentence_count = text.count('.') + text.count('!') + text.count('?')
    
    # 한글 문자 수
    korean_chars = sum(1 for c in text if '가' <= c <= '힣')
    
    # 평균 단어 길이
    avg_word_length = round(char_count / word_count, 2) if word_count > 0 else 0
    
    # 구조화된 결과 생성
    result = TextAnalysisResult(
        text=text[:50] + "..." if len(text) > 50 else text,
        char_count=char_count,
        word_count=word_count,
        sentence_count=sentence_count,
        korean_chars=korean_chars,
        avg_word_length=avg_word_length
    )
    
    return f"📊 텍스트 분석 완료:\n{result.model_dump_json(indent=2)}"


@tool
def weather_mood(city: str) -> str:
    """
    도시 이름을 받아 일관된 형식의 날씨 정보를 생성합니다.
    (실제 API 대신 데모용 구조화된 데이터)
    
    Args:
        city: 도시 이름
    
    Returns:
        해당 도시의 구조화된 날씨 정보
    """
    # 고정된 날씨 조건들
    weather_conditions = ["맑음", "흐림", "비", "눈", "안개", "바람"]
    moods = [
        "쨍쨍한 햇빛이 나고 있습니다!",
        "선선한 바람이 불어 좋습니다!",
        "쌀쌀한 핫초코가 생각나는 날입니다!",
        "습도가 높아 끈적끈적한 느낌입니다!",
        "화기애애한 분위기입니다!"
    ]
    
    # 일관된 형식으로 데이터 생성
    weather_info = WeatherInfo(
        city=city,
        condition=random.choice(weather_conditions),
        temperature=random.randint(-10, 35),
        mood=random.choice(moods)
    )
    
    return f"🌤️ {city} 날씨 정보:\n{weather_info.model_dump_json(indent=2)}"


# ===== Agent 설정 및 데모 함수들 =====

def setup_tools_and_model():
    """
    도구와 모델을 설정합니다.
    기본 검색 도구와 커스텀 도구들을 모두 포함합니다.
    
    Returns:
        tuple: (tools, model) 튜플
    """
    print("🔧 도구와 모델 설정 중...")
    
    # 기본 검색 도구
    search = TavilySearch(max_results=2)
    
    # 동기 커스텀 도구들
    sync_tools = [
        calculator,
        current_time,
        korean_name_generator,
        text_analyzer,
        weather_mood
    ]
    
    # 모든 도구 결합
    tools = [search] + sync_tools
    
    # 모델 설정
    model = init_chat_model("gpt-4o-mini", model_provider="openai")
    
    print(f"✅ 도구와 모델 설정 완료 (총 {len(tools)}개 도구)")
    print(f"📋 동기 도구: {', '.join([tool.name for tool in sync_tools])}")
    
    return tools, model


def basic_tools_demo(model, tools):
    """
    기본 도구들(동기)만 사용하는 Agent 데모입니다.
    
    Args:
        model: 언어 모델
        tools: 사용할 도구들
    """
    print("\n🔧 기본 도구 Agent 데모 시작...")
    
    # 기본 도구들만 필터링 (동기 도구들)
    basic_tools = [
        tool for tool in tools 
        if tool.name in ['tavily_search_results_json', 'calculator', 'current_time', 
                        'korean_name_generator', 'text_analyzer', 'weather_mood']
    ]
    
    # Agent 생성
    agent_executor = create_react_agent(model, basic_tools)
    
    # 기본 도구 테스트 쿼리들
    queries = [
        "현재 시간을 알려주고, 10 + 20 * 3을 계산해주세요.",
        "한국어 이름을 3개 생성해주고, 서울의 날씨 기분을 알려주세요.",
        "'안녕하세요! 반갑습니다. 좋은 하루 되세요!'라는 텍스트를 분석해주세요."
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n📝 기본 도구 테스트 {i}: {query}")
        print("-" * 60)
        
        result = agent_executor.invoke({"messages": [("user", query)]})
        print("🤖 답변:")
        print(result["messages"][-1].content)
        print("-" * 60)
    
    print("✅ 기본 도구 Agent 데모 완료")


def basic_agent_demo(model, tools):
    """
    기본 Agent 실행을 데모합니다.
    
    Args:
        model: 언어 모델
        tools: 사용할 도구들
    """
    print("\n🤖 기본 Agent 데모 시작...")
    
    # Agent 생성
    agent_executor = create_react_agent(model, tools)
    
    # 기본 쿼리 실행
    query = "안녕하세요! 최근 AI 기술 동향에 대해 알려주세요."
    print(f"질문: {query}")
    
    result = agent_executor.invoke({"messages": [("user", query)]})
    
    print("답변:")
    print(result["messages"][-1].content)
    print("✅ 기본 Agent 데모 완료")


def streaming_agent_demo(model, tools):
    """
    스트리밍 Agent 실행을 데모합니다.
    다양한 stream_mode의 차이점을 보여줍니다.
    
    Args:
        model: 언어 모델
        tools: 사용할 도구들
    """
    print("\n🌊 스트리밍 Agent 데모 시작...")
    
    # Agent 생성
    agent_executor = create_react_agent(model, tools)
    
    # 스트리밍 쿼리 실행
    query = "2024년 최신 머신러닝 트렌드를 조사해주세요."
    print(f"질문: {query}")
    
    # 1. stream_mode="values" (기본값) - 전체 상태를 스트리밍
    print("\n📊 Mode 1: stream_mode='values' (전체 상태)")
    print("=" * 50)
    for step in agent_executor.stream({"messages": [("user", query)]}, stream_mode="values"):
        step["messages"][-1].pretty_print()
    
    print("\n" + "=" * 50)
    
    # 2. stream_mode="messages" - 메시지만 스트리밍
    print("\n💬 Mode 2: stream_mode='messages' (메시지만)")
    print("=" * 50)
    for step, metadata in agent_executor.stream(
        {"messages": [("user", query)]}, stream_mode="messages"
    ):
        # Agent 노드에서 나온 텍스트만 출력
        if metadata["langgraph_node"] == "agent" and (text := step.text()):
            print(text, end="|")
    
    print("\n" + "=" * 50)
    
    # 3. stream_mode="updates" - 업데이트만 스트리밍
    print("\n🔄 Mode 3: stream_mode='updates' (업데이트만)")
    print("=" * 50)
    for step in agent_executor.stream({"messages": [("user", query)]}, stream_mode="updates"):
        print(f"노드: {list(step.keys())}")
        if "agent" in step:
            print("Agent 응답:", step["agent"]["messages"][-1].content[:100] + "...")
        elif "tools" in step:
            print("도구 실행:", step["tools"]["messages"][-1].content[:100] + "...")
    
    print("\n" + "=" * 50)
    print("✅ 스트리밍 Agent 데모 완료")
    
    # 설명 추가
    print("\n📝 Stream Mode 설명:")
    print("- 'values': 전체 상태값을 스트리밍 (가장 상세)")
    print("- 'messages': 메시지만 스트리밍 (텍스트 중심)")
    print("- 'updates': 노드별 업데이트만 스트리밍 (구조 중심)")


def memory_agent_demo(model, tools):
    """
    메모리 Agent 실행을 데모합니다.
    
    Args:
        model: 언어 모델
        tools: 사용할 도구들
    """
    print("\n🧠 메모리 Agent 데모 시작...")
    
    # 메모리 설정
    memory = MemorySaver()
    
    # 메모리 Agent 생성
    agent_executor = create_react_agent(model, tools, checkpointer=memory)
    
    # 대화 설정
    config = {"configurable": {"thread_id": "demo-thread"}}
    
    # 첫 번째 질문
    query1 = "제 이름은 김철수입니다."
    print(f"첫 번째 질문: {query1}")
    
    result1 = agent_executor.invoke(
        {"messages": [("user", query1)]},
        config=config
    )
    print("첫 번째 답변:")
    print(result1["messages"][-1].content)
    
    # 두 번째 질문 (이전 대화 기억 테스트)
    query2 = "제 이름을 기억하시나요?"
    print(f"\n두 번째 질문: {query2}")
    
    result2 = agent_executor.invoke(
        {"messages": [("user", query2)]},
        config=config
    )
    print("두 번째 답변:")
    print(result2["messages"][-1].content)

    # 세 번째 질문 (이전 대화 기억 X)
    query3 = "제 이름을 기억하시나요?"
    print(f"\n세 번째 질문: {query3}")
    third_query3 = {"configurable": {"thread_id": "demo-thread2"}}

    result3 = agent_executor.invoke(
        {"messages": [("user", query3)]},
        config=third_query3
    )
    print("세 번째 답변:")
    print(result3["messages"][-1].content)
    
    print("✅ 메모리 Agent 데모 완료")


def main():
    """
    LangChain Agent 데모를 실행합니다.
    
    다음 기능들을 순서대로 데모합니다:
    1. 도구와 모델 설정
    2. 기본 도구들(동기) 데모
    3. 기본 Agent 실행
    4. 스트리밍 Agent 실행
    5. 메모리 Agent 실행
    """
    print("🚀 LangChain Agent 데모를 시작합니다!")

    try:
        # 1. 도구와 모델 설정
        tools, model = setup_tools_and_model()
        
        # 2. 기본 도구들(동기) 데모
        basic_tools_demo(model, tools)
        
        # 3. 기본 Agent 데모
        basic_agent_demo(model, tools)
        
        # 4. 스트리밍 Agent 데모
        streaming_agent_demo(model, tools)
        
        # 5. 메모리 Agent 데모
        memory_agent_demo(model, tools)
        
        return True
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return False


if __name__ == "__main__":
    main() 