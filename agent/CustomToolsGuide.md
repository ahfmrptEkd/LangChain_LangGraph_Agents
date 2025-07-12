# 🛠️ LangChain Custom Tools 완전 가이드

> 기본 Tool부터 비동기 병렬 처리 Tool까지 모든 것을 다루는 완전한 가이드

## 📋 목차
1. [Tool 기본 개념](#tool-기본-개념)
2. [단순 Tool 만들기](#단순-tool-만들기)
3. [복잡한 Tool 만들기](#복잡한-tool-만들기)
4. [비동기 Tool 만들기](#비동기-tool-만들기)
5. [병렬 처리 Tool 만들기](#병렬-처리-tool-만들기)
6. [Tool 매개변수와 타입](#tool-매개변수와-타입)
7. [에러 처리](#에러-처리)
8. [실제 예제들](#실제-예제들)
9. [베스트 프랙티스](#베스트-프랙티스)
10. [트러블슈팅](#트러블슈팅)

---

## 1. Tool 기본 개념

### 🎯 Tool이란?
- **LangChain에서 Agent가 사용할 수 있는 기능**
- **외부 API, 데이터베이스, 파일 시스템 등과 상호작용**
- **LLM이 직접 할 수 없는 작업들을 수행**

### 📊 Tool의 구조
```python
from langchain_core.tools import tool

@tool
def my_tool(param1: str, param2: int) -> str:
    """
    Tool의 설명을 여기에 작성합니다.
    
    Args:
        param1: 첫 번째 매개변수 설명
        param2: 두 번째 매개변수 설명
    
    Returns:
        반환값 설명
    """
    # 실제 로직 구현
    return "결과"
```

### 🔧 Tool의 핵심 구성요소
1. **@tool 데코레이터**: 함수를 Tool로 변환
2. **매개변수**: LLM이 전달하는 입력값
3. **Docstring**: LLM이 이해할 수 있는 Tool 설명
4. **반환값**: Tool 실행 결과

---

## 2. 단순 Tool 만들기

### 📝 기본 문법
```python
from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """
    수학 계산을 수행합니다.
    
    Args:
        expression: 계산할 수식 (예: "2 + 3 * 4")
    
    Returns:
        계산 결과
    """
    try:
        result = eval(expression)
        return f"계산 결과: {expression} = {result}"
    except Exception as e:
        return f"❌ 계산 오류: {str(e)}"
```

### 🕒 현재 시간 Tool
```python
from datetime import datetime

@tool
def current_time() -> str:
    """
    현재 한국 시간을 반환합니다.
    
    Returns:
        현재 날짜와 시간
    """
    now = datetime.now()
    return f"현재 시간: {now.strftime('%Y년 %m월 %d일 %H시 %M분 %S초')}"
```

### 🎲 랜덤 데이터 Tool
```python
import random

@tool
def korean_name_generator(count: int = 1) -> str:
    """
    한국어 이름을 생성합니다.
    
    Args:
        count: 생성할 이름 개수 (기본값: 1)
    
    Returns:
        생성된 한국어 이름들
    """
    surnames = ["김", "이", "박", "최", "정"]
    given_names = ["민준", "서연", "지우", "하은", "도윤"]
    
    names = []
    for _ in range(min(count, 10)):
        surname = random.choice(surnames)
        given_name = random.choice(given_names)
        names.append(f"{surname}{given_name}")
    
    return f"생성된 이름: {', '.join(names)}"
```

---

## 3. 복잡한 Tool 만들기

### 📊 텍스트 분석 Tool
```python
@tool
def text_analyzer(text: str) -> str:
    """
    텍스트를 분석하여 통계 정보를 제공합니다.
    
    Args:
        text: 분석할 텍스트
    
    Returns:
        텍스트 분석 결과
    """
    if not text:
        return "❌ 분석할 텍스트가 없습니다."
    
    # 기본 통계
    char_count = len(text)
    word_count = len(text.split())
    sentence_count = text.count('.') + text.count('!') + text.count('?')
    
    # 한글 문자 수
    korean_chars = sum(1 for c in text if '가' <= c <= '힣')
    
    analysis = {
        "전체 문자 수": char_count,
        "단어 수": word_count,
        "문장 수": sentence_count,
        "한글 문자 수": korean_chars,
        "평균 단어 길이": round(char_count / word_count, 2) if word_count > 0 else 0
    }
    
    result = "📊 텍스트 분석 결과:\n"
    for key, value in analysis.items():
        result += f"- {key}: {value}\n"
    
    return result
```

### 🌦️ 모의 날씨 Tool
```python
@tool
def weather_mood(city: str) -> str:
    """
    도시 이름을 받아 가상의 날씨 기분을 생성합니다.
    
    Args:
        city: 도시 이름
    
    Returns:
        해당 도시의 가상 날씨 기분
    """
    weather_conditions = ["맑음", "흐림", "비", "눈", "안개", "바람"]
    moods = ["상쾌함", "차분함", "우울함", "로맨틱함", "신비로움", "활기찬"]
    
    weather = random.choice(weather_conditions)
    mood = random.choice(moods)
    temp = random.randint(-10, 35)
    
    return f"🌤️ {city}의 오늘 날씨: {weather}, 기온 {temp}°C\n기분: {mood}한 하루가 될 것 같습니다!"
```

---

## 4. 비동기 Tool 만들기

### ⚡ 기본 비동기 Tool
```python
import asyncio

@tool
async def async_calculator(expression: str) -> str:
    """
    비동기로 수학 계산을 수행합니다.
    
    Args:
        expression: 계산할 수식
    
    Returns:
        계산 결과
    """
    # 비동기 처리 시뮬레이션
    await asyncio.sleep(0.1)
    
    try:
        result = eval(expression)
        return f"비동기 계산 결과: {expression} = {result}"
    except Exception as e:
        return f"❌ 계산 오류: {str(e)}"
```

### 🌐 비동기 웹 요청 Tool
```python
import aiohttp

@tool
async def fetch_url(url: str) -> str:
    """
    URL에서 데이터를 비동기로 가져옵니다.
    
    Args:
        url: 요청할 URL
    
    Returns:
        응답 내용
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    return f"✅ URL 내용 (첫 100자): {content[:100]}..."
                else:
                    return f"❌ HTTP 오류: {response.status}"
    except Exception as e:
        return f"❌ 요청 오류: {str(e)}"
```

---

## 5. 병렬 처리 Tool 만들기

### 🔢 병렬 계산 Tool
```python
@tool
async def parallel_calculations(expressions: str) -> str:
    """
    여러 수식을 병렬로 계산합니다.
    
    Args:
        expressions: 쉼표로 구분된 수식들 (예: "2+3, 4*5, 10/2")
    
    Returns:
        모든 계산 결과
    """
    async def calculate_single(expr: str) -> str:
        """단일 계산을 비동기로 수행"""
        await asyncio.sleep(0.1)  # 처리 시뮬레이션
        try:
            expr = expr.strip()
            result = eval(expr)
            return f"{expr} = {result}"
        except Exception as e:
            return f"{expr} = 오류 ({str(e)})"
    
    # 수식들을 파싱
    expr_list = [expr.strip() for expr in expressions.split(',')]
    
    # 병렬 계산 실행
    start_time = datetime.now()
    results = await asyncio.gather(*[calculate_single(expr) for expr in expr_list])
    end_time = datetime.now()
    
    duration = (end_time - start_time).total_seconds()
    
    result_text = "🔢 병렬 계산 결과:\n"
    for result in results:
        result_text += f"- {result}\n"
    result_text += f"⏱️ 총 처리 시간: {duration:.2f}초"
    
    return result_text
```

### 🏙️ 다중 도시 날씨 Tool
```python
@tool
async def multi_city_weather(cities: str) -> str:
    """
    여러 도시의 날씨를 병렬로 조회합니다.
    
    Args:
        cities: 쉼표로 구분된 도시들 (예: "서울,부산,대구")
    
    Returns:
        모든 도시의 날씨 정보
    """
    async def get_city_weather(city: str) -> str:
        """단일 도시 날씨를 비동기로 조회"""
        await asyncio.sleep(0.2)  # API 호출 시뮬레이션
        
        weather_conditions = ["맑음", "흐림", "비", "눈"]
        weather = random.choice(weather_conditions)
        temp = random.randint(-10, 35)
        
        return f"{city}: {weather}, {temp}°C"
    
    # 도시들을 파싱
    city_list = [city.strip() for city in cities.split(',')]
    
    # 병렬 날씨 조회
    start_time = datetime.now()
    weather_results = await asyncio.gather(*[get_city_weather(city) for city in city_list])
    end_time = datetime.now()
    
    duration = (end_time - start_time).total_seconds()
    
    result_text = "🌤️ 다중 도시 날씨 결과:\n"
    for weather in weather_results:
        result_text += f"- {weather}\n"
    result_text += f"⏱️ 처리 시간: {duration:.2f}초 ({len(city_list)}개 도시 동시 조회)"
    
    return result_text
```

### 📝 배치 텍스트 분석 Tool
```python
from typing import Dict

@tool
async def batch_text_analysis(texts: str) -> str:
    """
    여러 텍스트를 병렬로 분석합니다.
    
    Args:
        texts: 파이프(|)로 구분된 텍스트들
    
    Returns:
        모든 텍스트의 분석 결과
    """
    async def analyze_single_text(text: str, index: int) -> Dict:
        """단일 텍스트를 비동기로 분석"""
        await asyncio.sleep(0.1)
        
        if not text:
            return {"index": index, "text": "빈 텍스트", "analysis": "분석 불가"}
        
        char_count = len(text)
        word_count = len(text.split())
        korean_chars = sum(1 for c in text if '가' <= c <= '힣')
        
        return {
            "index": index,
            "text": text[:20] + "..." if len(text) > 20 else text,
            "char_count": char_count,
            "word_count": word_count,
            "korean_chars": korean_chars
        }
    
    # 텍스트들을 파싱
    text_list = [text.strip() for text in texts.split('|')]
    
    # 병렬 분석 실행
    start_time = datetime.now()
    analysis_results = await asyncio.gather(*[
        analyze_single_text(text, i) for i, text in enumerate(text_list, 1)
    ])
    end_time = datetime.now()
    
    duration = (end_time - start_time).total_seconds()
    
    result_text = "📊 배치 텍스트 분석 결과:\n"
    for result in analysis_results:
        result_text += f"- 텍스트 {result['index']}: '{result['text']}'\n"
        result_text += f"  문자: {result['char_count']}, 단어: {result['word_count']}, 한글: {result['korean_chars']}\n"
    
    result_text += f"⏱️ 처리 시간: {duration:.2f}초"
    
    return result_text
```

---

## 6. Tool 매개변수와 타입

### 🎯 매개변수 타입 정의
```python
from typing import List, Dict, Optional, Union

@tool
def advanced_calculator(
    expression: str,
    precision: int = 2,
    return_format: str = "text"
) -> str:
    """
    고급 계산기 with 다양한 매개변수 타입
    
    Args:
        expression: 계산할 수식 (필수)
        precision: 소수점 자릿수 (기본값: 2)
        return_format: 반환 형식 ("text" 또는 "json")
    
    Returns:
        계산 결과
    """
    try:
        result = eval(expression)
        
        if return_format == "json":
            return json.dumps({
                "expression": expression,
                "result": round(result, precision),
                "precision": precision
            })
        else:
            return f"계산 결과: {expression} = {round(result, precision)}"
    except Exception as e:
        return f"❌ 오류: {str(e)}"
```

### 📊 복잡한 데이터 구조 Tool
```python
@tool
def process_data_list(data: str) -> str:
    """
    JSON 형태의 데이터를 처리합니다.
    
    Args:
        data: JSON 문자열 형태의 데이터
    
    Returns:
        처리된 데이터 결과
    """
    try:
        # JSON 파싱
        parsed_data = json.loads(data)
        
        if isinstance(parsed_data, list):
            total = sum(parsed_data)
            avg = total / len(parsed_data)
            return f"📊 리스트 분석: 총합={total}, 평균={avg:.2f}, 개수={len(parsed_data)}"
        
        elif isinstance(parsed_data, dict):
            keys = list(parsed_data.keys())
            values = list(parsed_data.values())
            return f"📋 딕셔너리 분석: 키={keys}, 값={values}"
        
        else:
            return f"🔍 데이터 타입: {type(parsed_data)}, 값: {parsed_data}"
    
    except json.JSONDecodeError:
        return "❌ 잘못된 JSON 형식입니다."
    except Exception as e:
        return f"❌ 처리 오류: {str(e)}"
```

---

## 7. 에러 처리

### 🛡️ 기본 에러 처리
```python
@tool
def safe_division(a: float, b: float) -> str:
    """
    안전한 나눗셈 계산
    
    Args:
        a: 피제수
        b: 제수
    
    Returns:
        나눗셈 결과 또는 에러 메시지
    """
    try:
        if b == 0:
            return "❌ 0으로 나눌 수 없습니다."
        
        result = a / b
        return f"✅ {a} ÷ {b} = {result}"
    
    except TypeError:
        return "❌ 숫자가 아닌 값이 입력되었습니다."
    except Exception as e:
        return f"❌ 예상치 못한 오류: {str(e)}"
```

### 🔒 입력 검증과 에러 처리
```python
@tool
def validate_and_process(text: str, min_length: int = 1) -> str:
    """
    입력 검증과 함께 텍스트를 처리합니다.
    
    Args:
        text: 처리할 텍스트
        min_length: 최소 길이 (기본값: 1)
    
    Returns:
        처리 결과
    """
    # 입력 검증
    if not text:
        return "❌ 빈 텍스트는 처리할 수 없습니다."
    
    if len(text) < min_length:
        return f"❌ 텍스트 길이가 최소 {min_length}자 이상이어야 합니다."
    
    if len(text) > 1000:
        return "❌ 텍스트가 너무 깁니다. (최대 1000자)"
    
    try:
        # 실제 처리
        word_count = len(text.split())
        char_count = len(text)
        
        return f"✅ 처리 완료: {char_count}자, {word_count}단어"
    
    except Exception as e:
        return f"❌ 처리 중 오류: {str(e)}"
```

---

## 8. 실제 예제들

### 🔧 Tool 등록과 사용
```python
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model

# 1. Tool 정의
@tool
def my_calculator(expression: str) -> str:
    """계산기 도구"""
    try:
        result = eval(expression)
        return f"결과: {result}"
    except Exception as e:
        return f"오류: {e}"

# 2. 모델과 Tool 설정
model = init_chat_model("openai", model="gpt-4o-mini")
tools = [my_calculator]

# 3. Agent 생성
agent = create_react_agent(model, tools)

# 4. 실행
result = agent.invoke({"messages": [("user", "2 + 3을 계산해주세요")]})
print(result["messages"][-1].content)
```

### 🎯 다중 Tool 시스템
```python
# 여러 Tool을 조합한 시스템
tools = [
    calculator,           # 계산기
    current_time,         # 시간 조회
    korean_name_generator, # 이름 생성
    text_analyzer,        # 텍스트 분석
    weather_mood,         # 날씨 기분
    parallel_calculations, # 병렬 계산
    multi_city_weather,   # 다중 도시 날씨
    batch_text_analysis,  # 배치 텍스트 분석
]

# Agent 생성
agent = create_react_agent(model, tools)

# 복합 작업 실행
query = "현재 시간을 확인하고, 10+20과 30*4를 병렬로 계산해주세요."
result = agent.invoke({"messages": [("user", query)]})
```

---

## 9. 베스트 프랙티스

### ✅ 좋은 Tool 작성법

#### 1. **명확한 Docstring**
```python
@tool
def good_tool(param: str) -> str:
    """
    Tool의 목적을 명확히 설명합니다.
    
    이 도구는 무엇을 하는지, 언제 사용하는지 설명합니다.
    
    Args:
        param: 매개변수의 목적과 형식을 설명
    
    Returns:
        반환값의 형식과 의미를 설명
    """
    pass
```

#### 2. **적절한 에러 처리**
```python
@tool
def robust_tool(data: str) -> str:
    """견고한 에러 처리를 가진 도구"""
    try:
        # 입력 검증
        if not data:
            return "❌ 입력 데이터가 필요합니다."
        
        # 실제 처리
        result = process_data(data)
        return f"✅ 처리 완료: {result}"
    
    except SpecificError as e:
        return f"❌ 특정 오류: {str(e)}"
    except Exception as e:
        return f"❌ 예상치 못한 오류: {str(e)}"
```

#### 3. **성능 최적화**
```python
@tool
async def optimized_tool(items: str) -> str:
    """성능 최적화된 도구"""
    # 병렬 처리로 성능 향상
    async def process_item(item):
        # 개별 처리
        await asyncio.sleep(0.1)
        return f"처리됨: {item}"
    
    item_list = items.split(',')
    results = await asyncio.gather(*[process_item(item) for item in item_list])
    
    return '\n'.join(results)
```

#### 4. **타입 힌트 활용**
```python
from typing import List, Dict, Optional

@tool
def typed_tool(
    text: str,
    options: Optional[str] = None,
    count: int = 1
) -> str:
    """타입 힌트를 활용한 도구"""
    pass
```

### ❌ 피해야 할 안티패턴

#### 1. **애매한 설명**
```python
@tool
def bad_tool(x: str) -> str:
    """뭔가 한다"""  # ❌ 너무 애매함
    pass
```

#### 2. **에러 처리 누락**
```python
@tool
def unsafe_tool(data: str) -> str:
    """에러 처리 없는 도구"""
    return eval(data)  # ❌ 위험한 코드
```

#### 3. **너무 복잡한 Tool**
```python
@tool
def complex_tool(data: str) -> str:
    """너무 많은 기능을 하나에 몰아넣음"""
    # 10개 이상의 다른 기능들...
    pass  # ❌ 단일 책임 원칙 위반
```

---

## 10. 트러블슈팅

### 🔍 자주 발생하는 문제들

#### 1. **Tool이 호출되지 않음**
```python
# 문제: 설명이 불분명
@tool
def unclear_tool(x: str) -> str:
    """도구"""  # ❌ 설명 부족
    return x

# 해결: 명확한 설명 추가
@tool
def clear_tool(text: str) -> str:
    """
    텍스트를 그대로 반환하는 에코 도구입니다.
    
    Args:
        text: 반환할 텍스트
    
    Returns:
        입력받은 텍스트
    """
    return text
```

#### 2. **비동기 Tool 오류**
```python
# 문제: 비동기 함수에서 동기 함수 호출
@tool
async def async_problem(data: str) -> str:
    result = blocking_operation(data)  # ❌ 블로킹 호출
    return result

# 해결: 적절한 비동기 처리
@tool
async def async_solution(data: str) -> str:
    result = await asyncio.get_event_loop().run_in_executor(
        None, blocking_operation, data
    )
    return result
```

#### 3. **매개변수 파싱 오류**
```python
# 문제: 복잡한 매개변수 구조
@tool
def complex_params(data: dict) -> str:  # ❌ 딕셔너리 직접 전달
    return str(data)

# 해결: 문자열로 받아서 파싱
@tool
def simple_params(data: str) -> str:
    try:
        parsed = json.loads(data)
        return str(parsed)
    except json.JSONDecodeError:
        return "❌ JSON 형식이 아닙니다."
```

### 🧪 디버깅 팁

#### 1. **Tool 동작 확인**
```python
@tool
def debug_tool(input_data: str) -> str:
    """디버깅을 위한 도구"""
    print(f"입력 데이터: {input_data}")
    print(f"데이터 타입: {type(input_data)}")
    
    try:
        result = process_data(input_data)
        print(f"처리 결과: {result}")
        return result
    except Exception as e:
        print(f"오류 발생: {e}")
        return f"❌ 오류: {e}"
```

#### 2. **Tool 테스트**
```python
# Tool 단독 테스트
def test_my_tool():
    result = my_tool("테스트 입력")
    print(f"결과: {result}")
    assert "예상 결과" in result

# 실행
test_my_tool()
```

---

## 📋 요약

### 🎯 Tool 개발 체크리스트

- [ ] **명확한 Docstring** 작성
- [ ] **적절한 타입 힌트** 사용
- [ ] **에러 처리** 구현
- [ ] **입력 검증** 추가
- [ ] **성능 최적화** 고려 (비동기/병렬)
- [ ] **테스트** 작성
- [ ] **보안** 고려 (eval 사용 시 주의)

### 🚀 다음 단계

1. **간단한 Tool부터 시작**
2. **점진적으로 복잡한 기능 추가**
3. **비동기/병렬 처리 도입**
4. **실제 API 연동**
5. **에러 처리 강화**

---

*이 가이드는 `agents.py`의 실제 코드를 기반으로 작성되었습니다. 더 자세한 예제는 해당 파일을 참고하세요.* 