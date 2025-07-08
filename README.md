# LangChain Advanced RAG and Study Projects

이 저장소는 LangChain을 사용하여 고급 RAG(Retrieval-Augmented Generation) 기법을 구현하고, 관련 기술들을 학습하기 위한 개인 연구 공간입니다. 표준적인 RAG를 넘어, 더 정확하고 신뢰성 높은 LLM 애플리케이션을 구축하기 위한 다양한 방법론을 직접 코드로 구현하고 실험하는 것을 목표로 합니다.

![rag short](src/imgs/rag%20short.png)
(Source: https://www.youtube.com/watch?v=wd7TZ4w1mSw&feature=youtu.be)

---

## 🎯 프로젝트 목표 및 범위

### 1. 현재 초점: 고급 RAG 구현 (Advanced RAG Implementation)

현재 이 프로젝트는 **기초적인 RAG 부터 고급 RAG 기법**을 구현하고 체계적으로 정리하는 데 중점을 두고 있습니다. `RAG/` 디렉토리 내에 각 RAG 단계를 세분화하여, 아래와 같은 다양한 기술들을 구현하고 문서화합니다.

-   **Query Transformation**: Query Construction, Query Translation (Multi-Query, RAG-Fusion, Decomposition, Step-back, HyDE)
-   **Indexing**: Multi-Representation, RAPTOR 등
-   **Routing**: Logical and Semantic Routing
-   **Retrieval**: Reranking, Self-RAG, CRAG, Adaptive RAG

각 기법에 대한 상세한 설명과 소스 코드는 해당 하위 폴더의 `README.md` 파일에서 확인할 수 있습니다.

### 2. 미래 확장 계획 (Future Scope)

이 저장소는 RAG를 넘어 다음과 같은 주제들로 확장될 예정입니다.

-   **CAG (Cache-Augmented Generation)**: 캐시된 정보를 활용하여 응답 속도를 향상시키고 일관성을 유지하는 기법들을 탐구합니다.
-   **KAG (Knowledge-Augmented Generation)**: 외부 지식 그래프(Knowledge Graph)를 활용하여 더 깊이 있는 추론과 답변 생성을 연구합니다.
-   **추가적인 랭체인 프로젝트**: LLM 애플리케이션 개발과 관련된 다양한 소규모 스터디와 실험 결과들을 기록할 예정입니다.

> **Note**: LangGraph를 활용한 복잡한 에이전트 및 워크플로우 구현은, 명확한 초점 유지를 위해 별도의 저장소에서 진행될 예정입니다.

---

## 📂 저장소 구조

-   **/RAG**: 이 프로젝트의 핵심으로, 다양한 고급 RAG 기법들의 구현 코드와 상세한 `README.md` 문서가 포함되어 있습니다.
-   **/src**: 이미지 등 프로젝트에 사용되는 소스 파일들을 포함합니다.