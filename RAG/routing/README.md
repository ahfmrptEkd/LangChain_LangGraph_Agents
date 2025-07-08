# 🚀 RAG Routing

## 파일 구조 (File Structure)
```
routing/
├── logical_routing.py     # 논리적 라우팅 시스템 구현
├── sementic_routing.py    # 의미론적 라우팅 시스템 구현
└── README.md              # 현재 문서
```

---

## 📖 목차
1. [개요 (Overview)](#1-개요-overview)
2. [Routing이 필요한 이유](#2-routing이-필요한-이유)
3. [구현된 기법들](#3-구현된-기법들)
    - [Logical Routing (논리적 라우팅)](#-logical-routing-논리적-라우팅)
    - [Semantic Routing (의미론적 라우팅)](#-semantic-routing-의미론적-라우팅)
4. [기법별 비교 테이블](#4-기법별-비교-테이블)
5. [선택 가이드 (Use Cases)](#5-선택-가이드-use-cases)
6. [결론 및 핵심 인사이트](#6-결론-및-핵심-인사이트)

---

## 1. 개요 (Overview)

Routing은 사용자의 질문을 분석하여, **가장 적절한 처리 경로(Route)나 데이터 소스(DataSource)로 안내하는 지능형 교통 경찰**과 같은 역할을 합니다. RAG 시스템이 여러 도구나 데이터베이스를 다룰 때, Routing은 가장 효율적이고 정확한 답변을 생성할 수 있는 경로를 동적으로 결정합니다.

---

## 2. Routing이 필요한 이유

단일 RAG 파이프라인은 모든 유형의 질문에 최적화되어 있지 않습니다. Routing이 없다면 다음과 같은 문제에 직면합니다.

-   **⚠️ 비효율적인 자원 사용 (Inefficient Resource Usage)**: 간단한 질문에 답하기 위해 불필요하게 복잡한 체인을 실행하거나, 모든 질문에 대해 웹 검색을 수행하는 등 자원을 낭비하게 됩니다.
-   **⚠️ 부정확한 컨텍스트 (Incorrect Context)**: 질문의 성격과 맞지 않는 데이터 소스(예: Python 질문에 JavaScript 문서 검색)를 사용하여 잘못된 컨텍스트를 생성할 수 있습니다.
-   **⚠️ 확장성 부족 (Lack of Scalability)**: 새로운 도구나 데이터 소스를 추가할 때마다 전체 시스템 로직을 복잡하게 수정해야 합니다.

**Routing은 질문의 의도를 파악하여 적절한 전문가(체인, 데이터 소스)에게 작업을 할당함으로써, RAG 시스템 전체의 효율성, 정확성, 확장성을 크게 향상시킵니다.**

---

## 3. 구현된 기법들

### 🎯 Logical Routing (논리적 라우팅)
-   **핵심 아이디어**: LLM의 함수 호출(Function Calling) 기능을 사용하여, 사용자의 질문을 미리 정의된 **명확한 카테고리** 중 하나로 분류합니다.
-   **작동 방식**:
    ```python
    # 1. Pydantic으로 라우팅할 카테고리(datasource)를 정의
    class RouteQuery(BaseModel):
        datasource: Literal["python_docs", "js_docs", "golang_docs"]

    # 2. LLM이 질문을 분석하여 가장 적합한 카테고리를 구조화된 형태로 반환
    # "Why doesn't the following python code work?..."
    router = prompt | llm.with_structured_output(RouteQuery)
    result = router.invoke({"question": question})
    # -> result.datasource: "python_docs"

    # 3. 반환된 카테고리에 따라 적절한 체인을 선택하여 실행
    full_chain = router | RunnableLambda(choose_route)
    ```
-   **장점**: 카테고리가 명확하게 구분될 때 매우 높은 정확도를 보이며, 복잡한 조건부 로직을 구현하기에 용이합니다.
-   **단점**: 미리 정의된 카테고리 외의 질문에는 대응하기 어렵고, 카테고리 구분이 모호한 경우 성능이 저하될 수 있습니다.

### 🎯 Semantic Routing (의미론적 라우팅)
-   **핵심 아이디어**: 질문을 벡터로 임베딩한 후, 미리 정의된 여러 **전문가 프롬프트(Prompt)**들의 임베딩과 **코사인 유사도**를 비교하여 가장 유사한 프롬프트를 선택합니다.
-   **작동 방식**:
    ```python
    # 1. 각 전문가(물리학, 수학)의 역할을 정의하는 프롬프트 템플릿 준비
    physics_template = "You are a very smart physics professor..."
    math_template = "You are a very good mathematician..."

    # 2. 질문과 각 프롬프트 템플릿을 임베딩
    query_embedding = embeddings.embed_query(input["query"])
    prompt_embeddings = embeddings.embed_documents(prompt_templates)

    # 3. 코사인 유사도가 가장 높은 프롬프트를 선택하여 체인 구성
    similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
    most_similar = prompt_templates[similarity.argmax()]
    ```
-   **장점**: 의미적 유사성에 기반하므로 유연성이 높고, 새로운 전문가 프롬프트를 쉽게 추가할 수 있습니다. 처리 속도가 빠릅니다.
-   **단점**: 라우팅의 기준이 되는 프롬프트의 품질에 성능이 크게 의존하며, 미묘한 의미 차이를 구분하는 데 한계가 있을 수 있습니다.

---

## 4. 기법별 비교 테이블

| 구분 | Logical Routing | Semantic Routing |
| :--- | :--- | :--- |
| **핵심 기술** | LLM Function Calling + Pydantic | Vector Embeddings + Cosine Similarity |
| **판단 기준** | 명시적 카테고리 분류 | 의미적 유사도 |
| **정확도** | 높음 (카테고리가 명확할 때) | 중간 (의미에 따라 유연) |
| **유연성** | 낮음 (정의된 카테리 내) | 높음 (새로운 프롬프트 추가 용이) |
| **속도** | LLM 호출로 인해 상대적으로 느림 | 임베딩 비교로 매우 빠름 |

---

## 5. 선택 가이드 (Use Cases)

| 상황/목표 | 추천 기법 | 이유 |
| :--- | :--- | :--- |
| **질문의 범주가 명확하게 나뉠 때** | `Logical Routing` | "이 질문은 파이썬 관련인가, 자바스크립트 관련인가?" 와 같이 명확한 분류가 필요할 때 가장 정확합니다. |
| **질문의 '의도'나 '주제'가 중요할 때** | `Semantic Routing` | "이 질문은 물리학자에게 물어봐야 할까, 수학자에게 물어봐야 할까?" 와 같이 의미적 맥락에 따라 전문가를 선택해야 할 때 효과적입니다. |
| **여러 데이터 소스를 정확히 선택해야 할 때** | `Logical Routing` | 특정 문서, API, 데이터베이스 등 명시적인 대상을 선택해야 할 때 적합합니다. |
| **빠른 응답 속도가 중요할 때** | `Semantic Routing` | LLM 호출 없이 임베딩 계산만으로 라우팅이 가능하여 속도가 빠릅니다. |

---

## 6. 결론 및 핵심 인사이트

> **"Routing은 RAG 시스템에 '눈'을 달아주는 것과 같습니다. 질문의 본질을 꿰뚫어 보고, 가장 적합한 전문가에게 안내하는 능력은 시스템 전체의 지능을 한 단계 끌어올립니다."**

-   **💡 명시적 vs. 암묵적**: Logical Routing은 명시적으로 정의된 규칙에 따라 작동하는 반면, Semantic Routing은 데이터의 암묵적인 의미에 따라 작동합니다. 문제의 성격에 따라 적절한 방식을 선택해야 합니다.
-   **💡 조합의 가능성**: 두 라우팅은 조합하여 사용할 수 있습니다. 예를 들어, Logical Routing으로 큰 카테고리(예: "기술 문서")를 먼저 정하고, 그 안에서 Semantic Routing으로 세부 전문가(예: "네트워크 전문가", "보안 전문가")를 선택하는 계층적 라우팅도 가능합니다.
-   **💡 라우팅은 시작일 뿐**: 라우팅은 단지 올바른 길을 안내할 뿐, 각 경로(Chain)의 성능이 최종 결과의 품질을 결정합니다. 효과적인 라우팅과 잘 설계된 전문 에이전트가 결합될 때 최고의 시너지를 낼 수 있습니다.