# 🚀 Query Translation for RAG

## 파일 구조 (File Structure)
```
query_translation/
├── README.md                           # 현재 파일
├── multi_query.py                      # Multi-Query RAG 구현
├── rag_fusion.py                       # RAG-Fusion 구현
├── decomposition.py                    # Query Decomposition 구현
├── step_back.py                        # Step-back Prompting 구현
├── HyDE.py                             # HyDE 구현
└── LangChain_Core_Components.md        # 핵심 컴포넌트 가이드
```

---

## 📖 목차
1. [개요 (Overview)](#1-개요-overview)
2. [Query Translation이 필요한 이유](#2-query-translation이-필요한-이유)
3. [구현된 기법들](#3-구현된-기법들)
    - [Multi-Query RAG](#-multi-query-rag)
    - [RAG-Fusion](#-rag-fusion)
    - [Query Decomposition](#-query-decomposition)
    - [Step-back Prompting](#-step-back-prompting)
    - [HyDE (Hypothetical Document Embeddings)](#-hyde-hypothetical-document-embeddings)
4. [기법별 비교 테이블](#4-기법별-비교-테이블)
5. [선택 가이드 (Use Cases)](#5-선택-가이드-use-cases)
6. [결론 및 핵심 인사이트](#6-결론-및-핵심-인사이트)

---

## 1. 개요 (Overview)

Query Translation은 사용자의 질문을 **다각도로 변환**하여 RAG 시스템의 검색 성능을 향상시키는 핵심 전략입니다. 이를 통해 **숨겨진 관련 정보**를 찾아내고, 답변의 정확성과 풍부함을 극대화합니다.

---

## 2. Query Translation이 필요한 이유

표준 RAG는 사용자의 질문을 그대로 사용하므로, 다음과 같은 본질적인 한계에 부딪힙니다.

-   **⚠️ 어휘 불일치 (Vocabulary Mismatch)**: 사용자가 사용하는 단어와 문서의 용어가 다를 때 (예: "AI 비서" vs "LLM Agent") 검색 성능이 저하됩니다.
-   **⚠️ 질문의 모호성 (Ambiguity of Query)**: 질문이 너무 광범위하거나 여러 주제를 포함할 때, 핵심 의도를 파악하기 어렵습니다.
-   **⚠️ 컨텍스트 부족 (Lack of Context)**: 질문에 명시적으로 드러나지 않은 배경지식이나 상위 개념을 놓치기 쉽습니다.

**Query Translation은 원본 질문을 더 효과적인 여러 검색어로 변환하여 이러한 한계를 극복하고, 검색의 정확도(Precision)와 재현율(Recall)을 모두 높입니다.**

---

## 3. 구현된 기법들

### 🎯 Multi-Query RAG
-   **핵심 아이디어**: 하나의 질문을 LLM을 통해 여러 개의 다른 관점을 가진 질문으로 변환하여 동시에 검색합니다.
-   **작동 방식**:
    ```python
    # 1. 원본 질문을 받아 여러 개의 변형된 질문 생성
    question = "What is task decomposition for LLM agents?"
    generate_queries = (prompt_perspectives | llm | ...)
    # -> ["How do LLM agents break down tasks?", "...", ...]

    # 2. 각 질문으로 병렬 검색 후, 고유한 문서들을 통합
    retrieval_chain = generate_queries | retriever.map() | get_unique_union
    unique_docs = retrieval_chain.invoke({"question": question})
    ```
-   **장점**: 다양한 관점에서 정보를 수집하여 재현율(Recall)을 높이고, 놓칠 수 있는 문서를 포착합니다.
-   **단점**: 생성된 질문들의 품질이 LLM에 의존적이며, 관련 없는 질문이 생성될 경우 노이즈가 발생할 수 있습니다.

### 🎯 RAG-Fusion
-   **핵심 아이디어**: Multi-Query와 유사하게 여러 질문을 생성하지만, 검색된 결과들을 Reciprocal Rank Fusion (RRF) 알고리즘으로 재정렬하여 가장 관련성 높은 문서를 상위로 올립니다.
-   **작동 방식**:
    ```python
    # 1. 여러 질문 생성 및 병렬 검색
    retrieval_chain = generate_queries | retriever.map()

    # 2. RRF 알고리즘으로 결과 재정렬
    # score = 1 / (rank + k)
    reranked_results = reciprocal_rank_fusion(retrieved_results)
    ```
-   **장점**: 여러 검색 결과에서 공통적으로 상위에 랭크된 문서에 가중치를 부여하여 검색 정확도(Precision)를 크게 향상시킵니다.
-   **단점**: RRF 계산을 위한 추가적인 단계가 필요합니다.

### 🎯 Query Decomposition
-   **핵심 아이디어**: 여러 단계의 추론이 필요한 복잡한 질문을 여러 개의 간단한 하위 질문으로 분해합니다.
-   **작동 방식**:
    ```python
    # 1. 복잡한 질문을 하위 질문들로 분해
    question = "What are the main components of an LLM-powered agent system?"
    sub_questions = generate_queries_decomposition.invoke({"question": question})
    # -> ["What is a LLM?", "What is an agent system?", ...]

    # 2. 각 하위 질문에 대해 RAG를 수행하고 답변을 종합
    # (접근법 1: 순차적 답변 / 접근법 2: 독립적 답변 후 종합)
    final_answer = synthesis_chain.invoke({"context": ..., "question": question})
    ```
-   **장점**: 복잡하고 다면적인 질문을 체계적으로 처리할 수 있으며, 각 단계별로 명확한 답변을 얻을 수 있습니다.
-   **단점**: 질문을 분해하는 과정 자체가 복잡하며, 분해된 질문들이 원래 질문의 의도를 모두 포함하지 못할 위험이 있습니다.

### 🎯 Step-back Prompting
-   **핵심 아이디어**: 구체적인 질문에서 한 걸음 물러나, 더 넓은 맥락을 포괄하는 일반적인 질문을 생성합니다. 이 두 가지 질문(원본+일반)을 모두 사용하여 문서를 검색합니다.
-   **작동 방식**:
    ```python
    # 1. Few-shot 프롬프트를 사용해 일반적인 질문 생성
    original_question = "What is task decomposition for LLM agents?"
    step_back_question = generate_queries_step_back.invoke({"question": original_question})
    # -> "What are the general approaches to task management in AI systems?"

    # 2. 원본 질문과 일반 질문의 검색 결과를 모두 사용해 최종 답변 생성
    final_answer = chain.invoke({"question": original_question})
    ```
-   **장점**: 원본 질문만으로는 찾기 어려운 상위 개념이나 배경지식을 함께 검색하여 더 풍부하고 깊이 있는 답변을 제공합니다.
-   **단점**: 일반적인 질문이 너무 광범위할 경우, 관련 없는 정보가 검색될 수 있습니다.

### 🎯 HyDE (Hypothetical Document Embeddings)
-   **핵심 아이디어**: 사용자의 질문에 대해 가상의 답변 문서(Hypothetical Document)를 먼저 생성하고, 이 가상 문서를 임베딩하여 실제 문서와 비교 검색합니다.
-   **작동 방식**:
    ```python
    # 1. 질문에 대한 가상의 답변 문서 생성
    question = "What is task decomposition for LLM agents?"
    hypothetical_doc = generate_docs_for_retrieval.invoke({"question": question})

    # 2. 가상 문서를 사용해 실제 문서 검색
    retrieved_docs = retriever.invoke(hypothetical_doc)
    ```
-   **장점**: 사용자의 질문과 실제 문서 간의 어휘 불일치(semantic gap)를 효과적으로 해결합니다. 특히 전문 용어가 많은 도메인에서 강력합니다.
-   **단점**: 생성된 가상 문서의 품질이 최종 검색 결과에 큰 영향을 미칩니다.

---

## 4. 기법별 비교 테이블

| 기법 | 질문 변환 방식 | 주요 장점 | 해결하는 RAG 한계점 |
| :--- | :--- | :--- | :--- |
| **Multi-Query** | 1→N 다양한 관점 | 포괄적 정보 수집 (Recall 향상) | 질문의 모호성 |
| **RAG-Fusion** | 1→N + RRF 순위화 | 높은 검색 정확도 (Precision 향상) | 검색 결과의 관련성 부족 |
| **Decomposition** | 1→N 하위 질문 | 복잡한 질문의 논리적 추론 | 다단계 질문 처리의 어려움 |
| **Step-back** | 1→2 (구체적→일반적) | 풍부한 배경지식 및 컨텍스트 제공 | 컨텍스트 부족 |
| **HyDE** | 1→가상문서→검색 | 어휘 불일치(Semantic Gap) 해결 | 사용자와 문서 간의 용어 차이 |

---

## 5. 선택 가이드 (Use Cases)

| 질문 유형 | 추천 기법 | 예시 질문 |
| :--- | :--- | :--- |
| **일반적이고 포괄적인 정보** | `Multi-Query` | "머신러닝이란 무엇인가?" |
| **정확하고 신뢰도 높은 답변** | `RAG-Fusion` | "신경망에서 경사 하강법은 어떻게 작동하는가?" |
| **여러 질문이 섞인 복잡한 문제** | `Decomposition` | "RAG 시스템을 구축하고 평가하는 방법은?" |
| **배경지식이나 넓은 맥락 필요** | `Step-back` | "트랜스포머 아키텍처의 최신 발전 동향은?" |
| **전문 용어가 많은 도메인** | `HyDE` | "NLP에서 어텐션 메커니즘의 중요성을 설명해줘." |

---

## 6. 결론 및 핵심 인사이트

> **"Query Translation은 단순한 검색 기술이 아니라, 사용자의 의도와 데이터베이스의 정보 사이의 간극을 메우는 지능적인 다리입니다."**

-   **💡 No Silver Bullet**: 완벽한 단일 기법은 없습니다. 질문의 유형과 목표에 따라 적절한 기법을 선택하거나 조합해야 합니다.
-   **💡 Recall vs. Precision**: `Multi-Query`는 재현율(Recall)을, `RAG-Fusion`은 정확도(Precision)를 높이는 데 더 효과적입니다.
-   **💡 Prompt Engineering is Key**: 대부분의 기법은 LLM을 사용하므로, 원하는 질문 변환을 유도하는 프롬프트 엔지니어링이 성능을 좌우합니다.
-   **💡 Guaranteed Improvement**: 어떤 기법을 사용하든, 단일 질문 방식보다 거의 항상 더 나은 결과를 제공합니다. RAG 시스템의 성능을 한 단계 끌어올리고 싶다면 Query Translation 도입은 필수적입니다.
