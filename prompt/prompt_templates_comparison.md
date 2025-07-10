# PromptTemplate vs ChatPromptTemplate

LangChain에서는 프롬프트 템플릿을 다루는 두 가지 주요 클래스로 `PromptTemplate`과 `ChatPromptTemplate`을 제공합니다.  
두 클래스는 목적과 사용 사례가 명확히 다릅니다.

## TL;DR

- **`ChatPromptTemplate`을 사용하세요.** 현대적인 챗봇 모델(GPT-4, Gemini 등)은 대부분 채팅 형식의 입력을 받도록 최적화되어 있어, 역할(`System`, `Human`, `AI`)을 구분해서 프롬프트를 구성하는 것이 훨씬 효과적입니다.
- **`PromptTemplate`**은 구형 텍스트 완성 모델(Text Completion Model)을 위한 것이며, 현재는 거의 사용되지 않습니다.

---

## 핵심 차이점

| 구분 | `PromptTemplate` | `ChatPromptTemplate` |
| --- | --- | --- |
| **결과물** | 단순 텍스트 문자열 (`string`) | 역할이 지정된 메시지 리스트 (`List[BaseMessage]`) |
| **대상 모델** | 구형 LLM (e.g., `text-davinci-003`) | **현대적인 채팅 모델** (e.g., `gpt-4`, `gemini`) |
| **목적** | 변수를 조합하여 하나의 완성된 텍스트 생성 | 시스템, 사용자, AI 등 역할을 구분하여 대화의 맥락을 구조적으로 전달 |
| **주요 장점** | 간단함 | 모델의 역할 이해도 및 지시사항 준수 능력 향상, 대화형 애플리케이션에 필수적 |

## `PromptTemplate` (구형 방식)

`PromptTemplate`은 모든 변수를 조합하여 최종적으로 하나의 긴 텍스트 문자열을 만듭니다.

**예시:**
```python
from langchain_core.prompts import PromptTemplate

template = "다음 주제에 대해 {length}로 설명해주세요: {topic}"
prompt = PromptTemplate.from_template(template)
formatted_prompt = prompt.format(topic="인공지능", length="3문장")

# 결과물 (string):
# "다음 주제에 대해 3문장로 설명해주세요: 인공지능"
```

이 방식은 모델에게 지시사항과 질문의 구분이 명확하지 않아 성능이 떨어질 수 있습니다.

## `ChatPromptTemplate` (현대적인 방식)

`ChatPromptTemplate`은 대화의 각 부분을 역할에 맞는 메시지 객체(`SystemMessage`, `HumanMessage` 등)로 만듭니다.  
이는 모델이 대화의 맥락과 각 메시지의 의도를 더 잘 파악하게 돕습니다.

**예시:**
```python
from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages([
    ("system", "당신은 {expertise} 전문가입니다."),
    ("human", "{question}")
])

formatted_messages = chat_template.format_messages(
    expertise="파이썬 프로그래밍",
    question="리스트 컴프리헨션 사용법을 알려주세요."
)

# 결과물 (List[BaseMessage]):
# [
#   SystemMessage(content="당신은 파이썬 프로그래밍 전문가입니다."),
#   HumanMessage(content="리스트 컴프리헨션 사용법을 알려주세요.")
# ]
```
이 구조는 모델이 시스템의 지시사항을 더 잘 따르고, 사용자의 질문에 더 정확하게 답변하도록 유도합니다.

## 결론

특별한 이유가 없다면 항상 **`ChatPromptTemplate`**을 사용하는 것이 더 보편적이고 좋습니다.  
이는 현대적인 LLM의 성능을 최대한 활용하고, 더 복잡하고 정교한 애플리케이션을 구축하기 위한 표준적인 방법입니다.
