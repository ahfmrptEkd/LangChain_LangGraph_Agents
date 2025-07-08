# 🚀 RAG Retrieval

## 파일 구조 (File Structure)
```
retrieval/
├── adaptive.py             # Adaptive RAG 구현
├── crag.py                 # CRAG (Corrective RAG) 구현
├── langchain_reranker.py   # LangChain Reranker 구현
├── rag_fusion.py           # RAG-Fusion 구현
├── self_rag.py             # Self-RAG 구현
└── README.md               # 현재 문서
```

---

## 📖 목차
1. [개요 (Overview)](#1-개요-overview)
2. [Retrieval 기법이 필요한 이유](#2-retrieval-기법이-필요한-이유)
3. [구현된 기법들](#3-구현된-기법들)
    - [RAG-Fusion](#-rag-fusion)
    - [Reranker](#-reranker)
    - [Adaptive RAG](#-adaptive-rag)
    - [CRAG (Corrective RAG)](#-crag-corrective-rag)
    - [Self-RAG](#-self-rag)
4. [기법별 비교 테이블](#4-기법별-비교-테이블)
5. [선택 가이드 (Use Cases)](#5-선택-가이드-use-cases)
6. [결론 및 핵심 인사이트](#6-결론-및-핵심-인사이트)

---

## 1. 개요 (Overview)

Retrieval 단계는 RAG의 성능을 좌우하는 가장 중요한 과정입니다. 이 폴더의 기법들은 단순히 문서를 가져오는 것을 넘어, **검색된 정보의 품질을 평가하고, 개선하며, 질문에 맞춰 최적의 전략을 선택**하는 고급 기능을 제공합니다. 이를 통해 최종 답변의 정확성과 신뢰성을 극대화하는 것을 목표로 합니다.

---

## 2. Retrieval 기법이 필요한 이유

표준 RAG의 Retrieval은 한 번의 벡터 검색으로 모든 것을 해결하려고 하므로, 다음과 같은 한계를 가집니다.

-   **⚠️ 관련성 낮은 문서 (Irrelevant Documents)**: 벡터 검색은 의미적으로 유사하지만, 실제로는 질문과 관련 없는 문서를 가져올 수 있습니다.
-   **⚠️ 검색 실패 (Retrieval Failure)**: 데이터베이스에 정보가 없거나, 질문이 모호하여 관련 문서를 전혀 찾지 못하는 경우가 발생합니다.
-   **⚠️ 단일 소스 의존성 (Single Source Dependency)**: 오직 내부 벡터 저장소에만 의존하므로, 최신 정보나 데이터베이스에 없는 정보에 답변할 수 없습니다.
-   **⚠️ 품질 평가 부재 (Lack of Quality Assessment)**: 검색된 문서의 품질이나 생성된 답변의 사실 여부를 스스로 평가하고 개선할 능력이 없습니다.

**고급 Retrieval 기법들은 검색 결과를 재정렬하고, 외부 소스를 활용하며, 스스로를 평가하고 교정하는 메커니즘을 도입하여 이러한 문제들을 해결합니다.**

---

## 3. 구현된 기법들

### 🎯 RAG-Fusion
-   **핵심 아이디어**: 여러 개의 변형된 질문으로 검색을 수행한 뒤, Reciprocal Rank Fusion (RRF) 알고리즘을 사용해 검색 결과들을 지능적으로 융합하고 재정렬합니다.
-   **작동 방식**:
    ```python
    # 1. 원본 질문을 여러 개의 유사 질문으로 변환
    # "What is agent memory?" -> ["Types of agent memory", ...]
    generate_queries = (prompt | llm | ...)

    # 2. 각 질문으로 병렬 검색 후, RRF로 점수화 및 재정렬
    retrieval_chain = generate_queries | retriever.map() | reciprocal_rank_fusion
    reranked_docs = retrieval_chain.invoke({"question": question})
    ```
-   **장점**: 다양한 관점에서 문서를 검색하여 재현율(Recall)을 높이고, RRF를 통해 가장 관련성 높은 문서를 상위에 배치하여 정확도(Precision)를 향상시킵니다.
-   **단점**: 여러 번의 검색으로 인해 비용과 지연 시간이 증가할 수 있습니다.

### 🎯 Reranker
-   **핵심 아이디어**: 1차적으로 검색된 문서들을 더 정교한 모델(Cross-Encoder 등)을 사용하여 질문과의 관련성을 다시 평가하고 순위를 재조정합니다.
-   **작동 방식**:
    ```python
    # 1. 기본 검색기로 넓은 범위의 후보 문서 검색 (k=15)
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 15})

    # 2. Cohere Rerank 또는 Cross-Encoder 같은 압축기를 사용해 재정렬
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=cohere_reranker, # or cross_encoder_reranker
        base_retriever=base_retriever
    )
    reranked_docs = compression_retriever.invoke(query)
    ```
-   **장점**: 초기 검색 결과의 노이즈를 효과적으로 제거하고, 최종적으로 LLM에 전달되는 컨텍스트의 질을 크게 향상시킵니다.
-   **단점**: 재정렬을 위한 추가적인 계산 비용이 발생합니다.

### 🎯 Adaptive RAG
-   **핵심 아이디어**: 사용자의 질문을 분석하여, 가장 적절한 검색 전략(벡터 검색, 웹 검색 등)을 동적으로 선택하는 라우팅 기반 접근법입니다.
-   **작동 방식**:
    ```python
    # 1. LLM 라우터가 질문의 유형을 판단
    # "agent memory 종류는?" -> "vectorstore"
    # "오늘 날씨는?" -> "web_search"
    source = question_router.invoke({"question": question})

    # 2. 결정된 경로에 따라 워크플로우 실행
    if source.datasource == "web_search":
        # 웹 검색 수행
    else:
        # 벡터 저장소 검색 수행
    ```
-   **장점**: 질문의 성격에 맞는 최적의 도구를 사용하여, 효율성과 답변의 정확성을 모두 높일 수 있습니다.
-   **단점**: 라우팅의 정확도에 전체 시스템의 성능이 크게 의존합니다.

### 🎯 CRAG (Corrective RAG)
-   **핵심 아이디어**: 검색된 문서의 품질을 LLM 평가기로 평가하고, 품질이 낮을 경우 웹 검색을 통해 정보를 보강하거나 질문을 변환하여 검색을 교정합니다.
-   **작동 방식**:
    ```python
    # 1. 문서 검색 후, LLM 평가기로 관련성 점수 매기기
    grade = retrieval_grader.invoke(...)

    # 2. 점수가 낮으면 (grade=="no"), 웹 검색 수행
    if web_search == "Yes":
        # 질문 변환 후 웹 검색
        web_results = web_search_tool.invoke(...)
        documents.append(web_results)
    ```
-   **장점**: 검색 실패 시 자동으로 대처하는 능력이 있어 시스템의 견고성이 높고, 잘못된 정보에 기반한 답변 생성을 방지합니다.
-   **단점**: 평가 및 교정 과정이 추가되어 워크플로우가 복잡해지고, 웹 검색으로 인한 비용이 발생할 수 있습니다.

### 🎯 Self-RAG
-   **핵심 아이디어**: 생성된 답변을 LLM 평가기를 통해 스스로 여러 기준(사실성, 관련성, 완전성)으로 평가하고, 기준 미달 시 만족할 때까지 답변 생성을 반복적으로 개선합니다.
-   **작동 방식**:
    ```python
    # 1. 답변 생성 후, 여러 평가기로 스스로 점검
    factual_result = factual_grader.invoke(...)
    relevance_result = relevance_grader.invoke(...)

    # 2. 평가 점수가 낮으면, 개선된 프롬프트로 재생성 시도
    if should_retry:
        generation = enhanced_chain.invoke(...)
    ```
-   **장점**: 답변의 품질을 자율적으로 관리하고 신뢰도 점수를 제공하여, 매우 높은 수준의 정확성과 신뢰성을 보장합니다.
-   **단점**: 반복적인 평가와 생성으로 인해 계산 비용이 매우 높고, 구현이 복잡합니다.

---

## 4. 기법별 비교 테이블

| 기법 | 핵심 초점 | 주요 메커니즘 | 장점 |
| :--- | :--- | :--- | :--- |
| **RAG-Fusion** | 검색 확장 및 융합 | 다중 쿼리 생성 + RRF | 재현율과 정확도 동시 향상 |
| **Reranker** | 검색 결과 재정렬 | Cross-Encoder / API | 노이즈 제거, 정확도 극대화 |
| **Adaptive RAG** | 동적 전략 선택 | 질문 라우팅 (LLM Router) | 효율성, 다양한 질문 유형 대응 |
| **CRAG** | 검색 결과 교정 | LLM 평가기 + 웹 검색 | 견고성, 검색 실패 시 자동 보정 |
| **Self-RAG** | 생성 답변 자가 개선 | LLM 평가기 (다중 기준) + 반복 | 신뢰성, 자동 품질 관리 |

---

## 5. 선택 가이드 (Use Cases)

| 상황/목표 | 추천 기법 | 이유 |
| :--- | :--- | :--- |
| **간단하게 검색 정확도 개선** | `Reranker` | 기존 시스템에 쉽게 추가할 수 있고, 비용 대비 효과가 가장 확실합니다. |
| **포괄적이면서도 정확한 답변** | `RAG-Fusion` | 다양한 관점의 검색으로 놓치는 정보를 줄이고, RRF로 핵심 정보를 강조합니다. |
| **최신 정보나 다양한 유형의 질문 처리** | `Adaptive RAG` | 질문에 맞는 최적의 도구(DB, 웹)를 선택하여 유연하게 대응합니다. |
| **답변의 사실성과 신뢰성이 매우 중요** | `CRAG` 또는 `Self-RAG` | 잘못된 정보나 환각을 최소화해야 하는 프로덕션 환경에 적합합니다. (CRAG는 검색, Self-RAG는 생성에 더 집중) |

---

## 6. 결론 및 핵심 인사이트

> **"고급 Retrieval 기법들은 RAG 시스템을 단순한 '검색-생성' 기계에서, 스스로 생각하고, 평가하며, 교정하는 '지능형 에이전트'로 발전시킵니다."**

-   **💡 평가의 중요성**: 검색된 문서와 생성된 답변의 품질을 '평가'하는 단계가 추가되는 것이 고급 RAG의 핵심입니다.
-   **💡 동적 워크플로우**: 정적인 파이프라인에서 벗어나, LangGraph 등을 활용하여 조건에 따라 분기하고 반복하는 동적 워크플로우가 중요해집니다.
-   **💡 비용과 성능의 균형**: Self-RAG처럼 강력한 기법일수록 더 많은 LLM 호출을 필요로 합니다. 실제 적용 시에는 비용과 성능 사이의 현실적인 균형점을 찾아야 합니다.
-   **💡 조합 가능성**: 이 기법들은 상호 배타적이지 않습니다. 예를 들어, Adaptive RAG의 각 경로에 Reranker를 적용하거나, RAG-Fusion과 Reranker를 함께 사용하는 등 필요에 따라 조합하여 더 강력한 시스템을 구축할 수 있습니다.
