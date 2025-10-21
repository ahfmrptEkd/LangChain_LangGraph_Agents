# 🤖 고급 AI 에이전트 패턴: LangChain & LangGraph 구현 저장소

> LangChain과 LangGraph를 활용한 최신 AI 에이전트 아키텍처(RAG, Multi-Agent, KG-RAG, CRAG) 및 프롬프트 엔지니어링 기법 심층 구현 및 비교 분석

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![LangChain](https://img.shields.io/badge/langchain-latest-green.svg)
![LangGraph](https://img.shields.io/badge/langgraph-latest-purple.svg)

*본 프로젝트는 다양한 고급 에이전트 아키텍처 구현을 목표로 합니다.*

---
<br>


## 🎯 프로젝트 개요

* **S (Situation / 문제)**
  
  기본적인 RAG 및 에이전트 구현은 널리 알려져 있지만, 실제 복잡한 문제 해결에는 **더욱 정교하고 신뢰성 높은 에이전트 아키텍처**가 필요합니다. 개발자들은 Adaptive RAG, Cache-RAG(CRAG), Knowledge Graph RAG (KG-RAG), 다양한 Multi-Agent 협업 패턴 등 최신 기법들을 **실제 코드로 이해하고 비교하며 적용**할 필요가 있습니다.
<br>

* **T (Task / 목표)**
  
  LangChain과 LangGraph 프레임워크를 기반으로, **다양한 고급 RAG 기법, Multi-Agent 아키텍처, 지식 그래프 활용 에이전트(KG-RAG), 캐시 RAG(CRAG), 그리고 효과적인 프롬프트 엔지니어링 전략**들을 직접 구현하고 비교 분석하는 것이 목표였습니다. 최종적으로는 각 패턴별 **재사용 가능한 코드 템플릿**을 제공하여 실용성을 높이고자 했습니다.
<br>

* **A (Action / 해결)**

  1. **고급 RAG 구현 (RAG/)**: Adaptive RAG, Corrective RAG (Self-Correction), RAG Fusion 등 다양한 검색 및 생성 전략을 LangGraph 기반 워크플로우로 구현했습니다. (예: `RAG/retrieval/crag.py`, `RAG/retrieval/adaptive.py`)

  2. **Multi-Agent 아키텍처 탐구 (Multi-agent/)**: 계층적(Hierarchical), 네트워크(Network), 감독자(Supervisor) 패턴 등 다양한 멀티 에이전트 협업 모델을 LangGraph로 구현하고 비교했습니다. (예: `Multi-agent/templates/supervisor.py`)

  3. **지식 그래프 연동 (KAG/)**: Neo4j 지식 그래프와 연동하여 Cypher 쿼리 생성 및 실행을 통해 구조화된 지식을 활용하는 KG-RAG (Knowledge Graph Agent)를 구현했습니다. (예: `KAG/run_example/dynamic_cypher.py`)

  4. **캐시 RAG 특화 (CAG/)**: 정보를 Cache하는 CAG 패턴을 집중적으로 구현하고, LangGraph를 사용한 구현과 그렇지 않은 구현 간의 성능 비교를 수행했습니다. (예: `CAG/true_cag_template.py`, `CAG/performance_comparison.py`)

  5. **프롬프트 엔지니어링 (prompt/)**: CoT (Chain-of-Thought), ToT (Tree-of-Thought), Self-Reflection 등 다양한 프롬프트 전략을 구현하고 비교했습니다. (예: `prompt/templates.py`)

  6. **기본 에이전트 패턴 (agent/)**: 챗봇, 정보 추출, 분류, 요약 등 기본적인 에이전트 활용 사례를 LangChain 기반으로 구현하여 기초를 다졌습니다.
<br>

* **R (Result / 결과)**
  
  * **핵심 산출물**: 다양한 고급 에이전트 패턴(`RAG`, `Multi-agent`, `KAG`, `CAG`)과 프롬프트 기법(`prompt`)을 실제 작동하는 **모듈화된 Python 코드 템플릿**으로 구현했습니다.
  * **기술적 성과**: LangGraph를 활용하여 상태 관리, 조건부 분기, 도구 호출 등 복잡한 에이전트 워크플로우를 효과적으로 제어하는 능력을 입증했습니다. CRAG 패턴 구현 시 **LangGraph 사용 유무에 따른 성능 비교 분석**(`CAG/performance_comparison.py`)을 통해 프레임워크의 장점을 정량적으로 확인했습니다. **Neo4j 연동**을 통해 지식 그래프 기반 에이전트 구축 역량을 확보했습니다.

---
<br>

## 📂 저장소 구조

-   **/RAG**: 이 프로젝트의 핵심으로, 다양한 고급 RAG 기법들의 구현 코드와 상세한 `README.md` 문서가 포함되어 있습니다.
-   **/prompt**: 프롬프트 엔지니어링 기법들(Basic, CoT, ToT, Self-Consistency, Self-Reflection)의 재사용 가능한 템플릿과 실행 예제를 포함합니다.
-   **/agent**: 에이전트 시스템, 대화형 RAG, 텍스트 분류, 문서 요약, 정보 추출 등 고급 NLP 기법들의 구현 코드를 포함합니다.
-   **/src**: 이미지 등 프로젝트에 사용되는 소스 파일들을 포함합니다.
-   **/CAG**: RAG를 기반으로, cache-augmented generation을 구현한 코드와 상세한 `README.md` 문서가 포함되어 있습니다.
-   **/KAG**: RAG를 기반으로, knowledge-augmented generation을 graph database(Neo4j)를 이용하여 구현한 코드와 상세한 `README.md` 문서가 포함되어 있습니다.
-   **/Multi-agent**: LangGraph를 활용한 다양한 멀티 에이전트 아키텍처 패턴(Network, Supervisor, Hierarchical)
-   **/doc**: 랭체인을 공부하며 배웠던 부분들을 문서로 작성하고 정리한 폴더입니다.

---
<br>

## 🛠️ 기술적 구현 (Action): 핵심 모듈 상세

### 1. 고급 RAG 패턴 (RAG/)
![rag short](src/imgs/rag%20short.png)
(Source: https://www.youtube.com/watch?v=wd7TZ4w1mSw&feature=youtu.be) <br>
LangGraph를 사용하여 단순 검색-생성을 넘어선 지능적인 RAG 워크플로우를 구현했습니다.

* **Query Transformation**: 사용자의 질문을 다양한 방식으로 변형하여 검색 성능을 향상시키는 기법들을 구현했습니다.
  * `query_translation/HyDE.py`: 가상의 답변을 생성하여 임베딩 유사도를 높이는 HyDE 기법.
  * `query_translation/multi_query.py`: 하나의 질문을 여러 하위 질문으로 분해.
  * `query_translation/rag_fusion.py`: 질문을 여러 버전으로 생성하고 결과를 RRF(Reciprocal Rank Fusion)로 결합.
  * `query_translation/step_back.py`: 구체적인 질문에서 한 단계 물러나 일반적인 질문을 생성하여 컨텍스트 확보.

* **Intelligent Retrieval & Generation**: 검색된 문서의 관련성을 평가하고 생성 방식을 동적으로 결정합니다.
  * `retrieval/adaptive.py`: 검색된 문서가 질문과 관련 없을 경우 웹 검색으로 전환하는 **Adaptive RAG**.
  * `retrieval/crag.py`: 검색된 문서의 신뢰도를 평가하고, 낮을 경우 웹 검색을 통해 정보를 보강/교정하는 **Corrective RAG (CRAG)**.
  * `retrieval/self_rag.py`: LLM 스스로 검색 필요 여부, 검색 결과 활용 여부, 생성 품질을 판단하는 **Self-RAG**.
  * `retrieval/langchain_reranker.py`: 검색된 문서를 `Cohere Rerank` 등을 이용해 재정렬하여 관련성 높은 문서를 우선순위화.

* **Routing**: 질문의 유형에 따라 다른 처리 경로(예: 요약 vs 벡터 검색)로 안내하는 로직을 구현했습니다.
  * `routing/logical_routing.py`: LLM의 논리적 판단에 기반한 라우팅.
  * `routing/sementic_routing.py`: 임베딩 유사도 기반의 시맨틱 라우팅.
<br>

### 2. Multi-Agent 아키텍처 (Multi-agent/)

LangGraph를 사용하여 여러 에이전트가 협업하는 다양한 방식을 구현했습니다. 각 에이전트는 특정 역할을 수행하며, `AgentState`를 통해 정보를 공유합니다.

* `templates/supervisor.py`: **감독자(Supervisor)** 패턴. 중앙의 감독자 에이전트가 사용자 요청을 분석하여 적절한 전문가 에이전트에게 작업을 할당하고 최종 응답을 종합합니다. 도구 사용 여부를 결정하는 라우팅 로직 포함 (`supervisor_tool.py`).
* `templates/hierarchical.py`: **계층적(Hierarchical)** 패턴. 여러 에이전트 팀이 계층 구조를 이루어 작업을 분담하고 결과를 상위 에이전트에게 보고하는 방식입니다.
* `templates/network.py`: **네트워크(Network)** 패턴. 특정 구조 없이 에이전트들이 자유롭게 소통하며 작업을 처리하는 방식입니다. (구현 예제 포함)
<br>

### 3. Knowledge Graph Agent (KAG/)
![kag](./src/imgs/kag.jpg) <br>
(Source:https://papooo-dev.github.io/posts/CAG_vs_KAG/)

구조화된 지식 그래프를 활용하여 RAG 성능을 향상시키는 KG-RAG를 구현했습니다.

* **Neo4j 연동**: `docker-compose.yml`을 통해 Neo4j 데이터베이스 환경을 설정합니다.
* **Dynamic Cypher Generation**: `run_example/dynamic_cypher.py`에서 LangChain의 `GraphCypherQAChain`을 사용하여 자연어 질문을 Cypher 쿼리로 변환하고 Neo4j에서 직접 답변을 검색합니다. LLM이 그래프 스키마를 이해하고 적절한 쿼리를 생성하는 것이 핵심입니다.
* **Semantic Layer**: `run_example/semantic_layer.py` (LangChain 문서 기반 아이디어)에서는 자연어 질문과 그래프 스키마 요소 간의 의미적 유사성을 기반으로 관련 노드/관계를 식별하여 보다 정확한 Cypher 쿼리 생성을 유도하는 접근 방식을 탐구합니다. (구현은 초기 단계)
<br>

### 4. Corrective RAG (CAG/)
![cag](./src/imgs/cag2.svg) <br>
(Source:https://papooo-dev.github.io/posts/cag-openai/)

CAG 패턴을 독립적으로 구현하고 성능을 비교 분석했습니다.

* `true_cag_template.py`: LangGraph를 사용하여 CAG 워크플로우
* `cag_template.py`: LangGraph 없이 일반적인 함수 호출 방식으로 CRAG 로직을 구현했습니다.
* `performance_comparison.py`: 위 두 방식의 실행 시간, 토큰 사용량 등을 비교하여 LangGraph 도입의 오버헤드와 장점을 평가했습니다.
* `semantic_cache_template.py`: 유사한 질문에 대한 이전 답변을 재사용하여 효율성을 높이는 시맨틱 캐싱 기법을 CRAG에 적용하는 예제를 구현했습니다.
<br>

### 5. Prompt Engineering Techniques (prompt/)

LLM의 추론 능력을 극대화하기 위한 다양한 프롬프트 전략을 구현하고 비교했습니다.

* `templates.py` 및 `run_examples/`:
  * **Zero-shot/Few-shot**: 기본적인 프롬프팅 기법.
  * **CoT (Chain-of-Thought)**: 단계별 추론 과정을 유도하여 복잡한 문제 해결 능력 향상.
  * **Self-Consistency**: 여러 번의 CoT 추론 결과 중 다수결로 가장 일관된 답변 선택.
  * **ToT (Tree-of-Thought)**: 여러 추론 경로를 트리 형태로 탐색하고 평가하여 최적의 해결책 도출.
  * **Self-Reflection**: 생성된 답변을 LLM 스스로 비판하고 개선하도록 유도.
<br>

---
<br>

## 📊 결과 및 성과 (Result)

* **모듈화된 코드 템플릿**: 각 고급 에이전트 패턴(Adaptive RAG, CRAG, KG-RAG, Supervisor Multi-agent 등)과 프롬프트 기법에 대한 **재사용 가능한 LangChain/LangGraph 코드 템플릿**을 제공합니다.
* **LangGraph 기반 워크플로우 제어**: 복잡한 조건부 로직과 상태 관리가 필요한 에이전트 워크플로우(특히 CRAG, Adaptive RAG)를 LangGraph를 통해 효과적으로 구현하고 제어할 수 있음을 입증했습니다.
* **성능 비교 분석**: CAG 구현 시 LangGraph 사용 유무에 따른 성능(실행 시간, 토큰 사용량)을 비교하여, 워크플로우 복잡도와 성능 간의 트레이드오프를 정량적으로 분석했습니다 (`CAG/performance_comparison.py`).
* **이론과 실습의 결합**: `doc/` 디렉토리에 각 에이전트 패턴과 LangChain/LangGraph 핵심 개념에 대한 상세한 이론적 설명과 가이드를 제공하여, 코드 구현과 함께 깊이 있는 학습이 가능하도록 구성했습니다.

---
<br>

## 💡 주요 학습 포인트 (Learnings)

* **LangGraph의 강점**: 복잡한 에이전트 워크플로우, 특히 상태 추적과 조건부 분기가 중요한 경우(예: CRAG, Adaptive RAG) LangGraph가 코드의 명확성과 제어 가능성을 크게 향상시킨다는 것을 확인했습니다.
* **고급 RAG의 필요성**: 단순 RAG는 부정확하거나 관련 없는 정보를 반환할 수 있으며, CRAG, Adaptive RAG, Self-RAG 등 상황에 맞게 검색 전략을 동적으로 조정하고 결과를 검증/교정하는 메커니즘이 필수적임을 깨달았습니다.
* **Multi-Agent 협업 패턴**: 작업의 성격에 따라 Supervisor, Hierarchical 등 적합한 협업 패턴을 선택하는 것이 중요하며, LangGraph는 이러한 패턴을 구현하는 데 유연한 구조를 제공함을 학습했습니다.
* **KG-RAG의 잠재력**: 구조화된 지식 그래프를 활용하면 특정 도메인에 대한 정확하고 효율적인 정보 검색이 가능하며, 자연어-Cypher 변환이 핵심 기술임을 파악했습니다.
* **프롬프트 엔지니어링의 중요성**: 동일한 LLM이라도 CoT, ToT, Self-Reflection 등 프롬프트 전략에 따라 문제 해결 능력이 크게 달라지며, 태스크에 맞는 전략 선택이 필수적임을 확인했습니다.

---
<br>

## 🚀 기술 스택

| 구분 | 기술 | 설명 |
|:-----|:-----|:-----|
| **Core Frameworks** | `LangChain`, `LangGraph` | AI 에이전트 및 워크플로우 구축 |
| **LLMs** | `ChatOpenAI`, `ChatGroq`, `ChatOllama` (설정 가능) | 언어 모델 인터페이스 |
| **Embeddings** | `OpenAIEmbeddings`, `HuggingFaceEmbeddings` (설정 가능) | 텍스트 임베딩 모델 |
| **Vector Stores** | `Chroma`, `FAISS` | 벡터 데이터베이스 |
| **Knowledge Graph** | `Neo4j`, `langchain_community.graphs.Neo4jGraph` | 지식 그래프 데이터베이스 및 연동 |
| **Tools & Utilities** | `Tavily Search`, `DuckDuckGo Search`, `Pydantic`, `BeautifulSoup4` | 웹 검색, 데이터 유효성 검사, 웹 스크래핑 등 |
| **Prompting** | `PromptTemplate`, `ChatPromptTemplate` | 다양한 프롬프트 엔지니어링 기법 구현 |
| **Development** | `Python 3.11+`, `Jupyter Notebook` (실험용), `.env` (환경 변수), `Docker` (Neo4j용) | 개발 환경 |
| **Performance** | `GPUtil`, `psutil`, `tiktoken` | 성능 측정 (GPU, 메모리, 토큰 사용량) |

*상세 의존성은 [`requirments.txt`](requirments.txt) 파일을 참조하세요.*

---
<br>

## 🚀 시작하기 (Getting Started)

### 전제 조건

* Python 3.11 이상
* Git
* (선택) OpenAI, Groq, Tavily 등 API 키
* (KAG 모듈) Docker 및 Docker Compose (Neo4j 실행용)
<br>

### 1. 환경 설정
```bash
# 저장소 클론
git clone https://github.com/ahfmrptekd/langchain_langgraph_agents.git
cd langchain_langgraph_agents

# 가상환경 생성 및 활성화
python3 -m venv .venv
source .venv/bin/activate

# 의존성 설치
pip install -r requirments.txt
```
<br>

### 2. API 키 설정

각 모듈(`agent`, `RAG`, `CAG`, `KAG`, `Multi-agent`, `prompt`) 디렉토리 내의 `.env.example` 파일을 복사하여 `.env` 파일을 만들고, 필요한 API 키를 입력합니다. 최소한 LLM (OpenAI, Groq 등) 및 임베딩 모델 관련 키가 필요합니다.
```bash
# 예시: RAG 모듈 설정
cp RAG/.env.example RAG/.env
# nano RAG/.env # 파일 열어서 API 키 입력
```
<br>

### 3. 모듈별 예제 실행

각 모듈 디렉토리 내의 `README.md` 파일과 `run_example/` 또는 최상위 스크립트 파일(`*.py`)을 참조하여 예제를 실행합니다.

**예시: CRAG (LangGraph 버전) 실행**
```bash
cd CAG
# .env 파일에 필요한 API 키 (LLM, Tavily 등) 설정

# LangGraph 기반 CRAG 실행
python run_examples/run_true_cag_example.py --question "LangGraph가 뭐지?"
```

**예시: Multi-Agent (Supervisor) 실행**
```bash
cd Multi-agent
# .env 파일에 필요한 API 키 (LLM, Tavily 등) 설정

# Supervisor 패턴 실행
python run_example/supervisor_run.py --question "미국 서부 날씨 알려주고 관련 블로그 글 초안 작성해줘"
```

**예시: Knowledge Graph Agent (KAG) 실행**
```bash
cd KAG
# .env 파일에 Neo4j 접속 정보 및 LLM API 키 설정

# Neo4j Docker 컨테이너 실행 (최초 1회 또는 필요시)
make start

# (필요시) 샘플 데이터 로딩 (Neo4j Browser 또는 Cypher shell 사용)
# 예: CREATE (:Movie {title:'The Matrix', released:1999}), (:Person {name:'Keanu Reeves'}) ...

# Dynamic Cypher QA 실행
python run_example/dynamic_cypher.py --question "키아누 리브스가 출연한 영화 알려줘"

# Neo4j Docker 컨테이너 종료
make stop
```

---
<br>

## 🧪 테스트 및 문서

* **실행 예제**: 각 모듈의 `run_example/` 디렉토리 또는 최상위 `*.py` 스크립트를 통해 핵심 기능을 직접 실행하고 확인할 수 있습니다.
* **이론 문서**: `doc/` 디렉토리에 LangChain/LangGraph 핵심 개념, RAG 기법, 에이전트 아키텍처 등에 대한 상세한 설명 문서가 포함되어 있습니다.
* **모듈별 README**: 각 모듈 디렉토리(`agent/`, `RAG/`, `CAG/`, `KAG/`, `Multi-agent/`, `prompt/`) 내의 `README.md` 파일은 해당 모듈의 목표, 구현 내용, 실행 방법을 구체적으로 안내합니다.
---
