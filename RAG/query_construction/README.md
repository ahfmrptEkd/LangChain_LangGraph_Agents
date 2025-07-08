# 🚀 Query Construction

## 파일 구조 (File Structure)
```
query_construction/
├── construction.py       # Query Construction 구현체
└── README.md             # 현재 문서
```

---

## 📖 목차
1. [개요 (Overview)](#1-개요-overview)
2. [Query Construction이 필요한 이유](#2-query-construction이-필요한-이유)
3. [핵심 개념 및 작동 방식](#3-핵심-개념-및-작동-방식)
4. [실제 활용 사례](#4-실제-활용-사례)
5. [결론 및 핵심 인사이트](#5-결론-및-핵심-인사이트)

---

## 1. 개요 (Overview)

Query Construction은 사용자의 자연어 질문을 **의미적 검색어(Semantic Query)**와 **구조화된 메타데이터 필터(Structured Filter)**로 변환하는 RAG의 핵심 기술입니다. 이를 통해 단순 키워드 검색을 넘어, 데이터의 속성을 정확하게 필터링하는 정밀 검색을 가능하게 합니다.

---

## 2. Query Construction이 필요한 이유

표준 RAG는 주로 벡터 유사도에만 의존하기 때문에, 다음과 같은 명확한 한계를 가집니다.

-   **⚠️ 메타데이터 무시 (Metadata Neglect)**: "2023년에 게시된 5분 이하 영상"과 같은 질문에서 `연도`나 `길이` 같은 중요한 필터 조건을 검색에 반영하지 못합니다.
-   **⚠️ 부정확한 결과 (Imprecise Results)**: 필터링이 불가능하므로, 관련 없는 수많은 결과를 반환하고 사용자가 직접 원하는 정보를 찾아야 하는 불편함이 발생합니다.
-   **⚠️ 비효율적인 검색 (Inefficient Search)**: 전체 데이터베이스를 대상으로 의미 검색을 수행한 후, 나중에 필터링(Post-filtering)해야 하므로 매우 비효율적입니다.

**Query Construction은 LLM을 사용해 사용자의 의도를 파악하고, 이를 데이터베이스가 이해할 수 있는 구조화된 쿼리로 변환하여 위 문제들을 해결합니다.**

---

## 3. 핵심 개념 및 작동 방식

Query Construction은 자연어 질문을 분석하여 구조화된 쿼리로 변환하는 과정을 따릅니다.

-   **핵심 아이디어**: LLM의 함수 호출(Function Calling) 기능을 사용하여, 사용자의 질문에서 검색어와 필터 조건을 추출하고 이를 Pydantic과 같은 데이터 모델에 매핑합니다.
-   **작동 방식**:
    ```python
    # 1. Pydantic으로 검색 필드를 정의한 데이터 모델 생성
    class TutorialSearch(BaseModel):
        content_search: str
        min_view_count: Optional[int]
        earliest_publish_date: Optional[datetime.date]
        max_length_sec: Optional[int]

    # 2. LLM이 사용자 질문을 분석하여 이 모델에 맞는 구조화된 출력 생성
    # "how to use multi-modal models in an agent, only videos under 5 minutes"
    structured_llm = llm.with_structured_output(TutorialSearch)
    query_analyzer = prompt | structured_llm
    result = query_analyzer.invoke({"question": question})

    # 3. 생성된 구조화된 쿼리 (Pydantic 객체)
    # -> content_search: "multi-modal models in an agent"
    # -> max_length_sec: 300
    ```
-   **장점**: 자연어의 유연함과 구조화된 데이터베이스 검색의 정확성을 결합하여, 사용자 경험과 검색 효율을 동시에 향상시킵니다.
-   **단점**: LLM이 사용자의 의도를 잘못 해석하거나, 정의된 스키마에 없는 필터를 요청할 경우 쿼리 생성에 실패할 수 있습니다.

---

## 4. 실제 활용 사례

| 플랫폼 | 사용자 질문 | → | 구조화된 쿼리 (필터) |
| :--- | :--- | :--- | :--- |
| **교육 플랫폼** | "초급자를 위한 한국어 파이썬 강의 중 3시간 이하인 것" | → | `language=ko`, `level=beginner`, `topic=python`, `duration<=10800` |
| **동영상 플랫폼** | "조회수 1만 이상인 최신 요리 영상 중 15분 이하" | → | `category=cooking`, `views>=10000`, `duration<=900`, `published>=...` |
| **뉴스 검색** | "지난 주 기술 뉴스 중 AI 관련 기사" | → | `category=tech`, `topic=AI`, `date>=last_week` |
| **전자상거래** | "5만원 이하 블루투스 헤드폰 중 평점 4점 이상" | → | `category=headphone`, `price<=50000`, `rating>=4`, `connectivity=bluetooth` |

---

## 5. 결론 및 핵심 인사이트

> **"Query Construction은 RAG 시스템의 '귀'와 '뇌' 역할을 합니다. 사용자의 말을 정확히 알아듣고(의도 파악), 이를 시스템이 실행할 수 있는 명확한 명령으로 변환하는 지능입니다."**

-   **💡 자연어와 시스템의 다리**: 이 기술은 사용자의 자연스러운 언어와 컴퓨터의 구조화된 데이터 처리 방식 사이의 간극을 메우는 핵심적인 다리 역할을 합니다.
-   **💡 사전 필터링(Pre-filtering)의 힘**: 검색 전에 데이터의 범위를 좁히는 사전 필터링은, 검색 후에 필터링하는 것보다 월등히 효율적이고 정확합니다.
-   **💡 LLM의 새로운 활용**: Query Construction은 LLM을 단순히 콘텐츠 생성기가 아닌, 사용자의 의도를 파악하고 시스템을 제어하는 **'지능형 라우터'**로 활용하는 좋은 예시입니다.