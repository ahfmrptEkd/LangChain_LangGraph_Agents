# LangChain Agent 및 고급 기법 라이브러리

이 디렉토리는 **LangChain과 LangGraph를 활용한 다양한 에이전트 및 고급 NLP 기법**의 Python 구현 예제를 포함하고 있습니다. 단순한 체인부터 복잡한 에이전트 시스템까지, 실제 프로덕션 환경에서 사용할 수 있는 고급 기법들을 체계적으로 정리하고 구현한 것을 목표로 한다.

각 Python 파일은 특정 기능(예: `chatbot`, `classification`, `summarization`)에 해당하며, 관련 마크다운 가이드 파일과 함께 상세한 설명과 사용법을 제공한다.

---

## 📚 목차

- [프로젝트 개요](#-프로젝트-개요)
- [디렉토리 구조](#-디렉토리-구조)
- [핵심 컨셉](#-핵심-컨셉)
- [실행 환경 설정](#-실행-환경-설정)
- [구현된 기법들](#-구현된-기법들)
- [사용 방법](#-사용-방법)
- [Agent vs Chain: 언제 무엇을 사용할까?](#agent-vs-chain-언제-무엇을-사용할까)
- [참고 자료](#-참고-자료)

---

## 🎯 프로젝트 개요

이 라이브러리는 다음과 같은 고급 AI 기법들을 실제 사용 가능한 형태로 구현합니다:

- **🤖 Agent 시스템**: 도구를 사용하여 복잡한 작업을 수행하는 지능형 에이전트
- **💬 대화형 RAG**: 메모리와 검색 기능을 갖춘 대화형 생성 시스템
- **🔍 텍스트 분류**: 다양한 분류 방법론을 통한 텍스트 카테고리화
- **📝 요약 시스템**: 맵-리듀스 패턴을 활용한 대규모 문서 요약
- **🏷️ 정보 추출**: 구조화된 데이터 추출 및 스키마 검증
- **🔗 상태 관리**: LangGraph를 활용한 복잡한 워크플로우 상태 관리

---

## 📁 디렉토리 구조

```
agent/
├── README.md                      # 현재 문서
│
├── # 🤖 Agent 시스템
├── agents.py                      # 커스텀 도구와 React 에이전트 구현
├── chatbot.py                     # 메모리 기능이 있는 기본 챗봇
├── conversationalRAG.py           # 대화형 RAG 시스템 (체인 + 에이전트)
│
├── # 📊 NLP 고급 기법
├── classification.py              # 텍스트 분류 (Basic/Enum/Literal 방법)
├── summarization.py               # 문서 요약 (맵-리듀스 패턴)
├── extraction.py                  # 구조화된 데이터 추출
```

---

## ✨ 핵심 컨셉

### 🔧 모듈화 설계
- **재사용 가능한 컴포넌트**: 각 기법을 독립적인 클래스로 구현하여 다른 프로젝트에서 쉽게 재사용
- **설정 기반 초기화**: 환경 변수와 설정 클래스를 통한 유연한 구성
- **에러 처리**: 프로덕션 환경을 고려한 포괄적인 예외 처리

### 🧠 지능형 에이전트
- **도구 기반 추론**: 계산기, 검색, 날씨 등 다양한 도구를 활용하는 에이전트
- **메모리 관리**: 대화 컨텍스트를 유지하고 관리하는 메모리 시스템
- **상태 추적**: LangGraph를 활용한 복잡한 워크플로우 상태 관리

### 🎨 다양한 접근 방식
- **체인 vs 에이전트**: 각 상황에 맞는 최적의 접근 방식 제공
- **동기 vs 비동기**: 성능 최적화를 위한 비동기 처리 지원
- **구조화된 출력**: Pydantic 모델을 활용한 타입 안전한 데이터 처리

---

## 🛠️ 구현된 기법들

| 분류 | 파일명 | 주요 기능 | 사용 기술 |
|------|--------|-----------|-----------|
| **🤖 Agent 시스템** | `agents.py` | 커스텀 도구 + React 에이전트 | LangGraph, 커스텀 도구 |
| **💬 대화형 AI** | `chatbot.py` | 메모리 기반 챗봇 | LangGraph, MemorySaver |
| **🔍 검색 생성** | `conversationalRAG.py` | 대화형 RAG (체인 + 에이전트) | 벡터 스토어, 검색 도구 |
| **📊 텍스트 분류** | `classification.py` | 다중 분류 방법론 | Pydantic, Literal 타입 |
| **📝 문서 요약** | `summarization.py` | 맵-리듀스 요약 | LangGraph, 병렬 처리 |
| **🏷️ 정보 추출** | `extraction.py` | 구조화된 데이터 추출 | Function Calling, 스키마 검증 |

### 🎯 Agent vs Chain 비교

| 구분 | Chain | Agent |
|------|--------|--------|
| **복잡성** | 단순, 예측 가능 | 복잡, 동적 |
| **실행 흐름** | 순차적, 고정됨 | 도구 기반, 적응적 |
| **사용 사례** | 정해진 작업 파이프라인 | 문제 해결, 멀티스텝 추론 |
| **성능** | 빠름, 효율적 | 느림, 자원 소모 |
| **디버깅** | 쉬움 | 어려움 |
| **적합한 시나리오** | 번역, 요약, 분류 | 검색+분석, 계산+추론 |

---

## 🤔 Agent vs Chain: 언제 무엇을 사용할까?

### 🔗 Chain을 사용하는 경우
```python
# ✅ 좋은 예: 정해진 순서의 작업
# 텍스트 → 번역 → 요약 → 분류
text_processing_chain = (
    translation_prompt | llm |
    summarization_prompt | llm |
    classification_prompt | llm
)
```

**Chain 선택 기준:**
- 작업 순서가 명확하고 고정적
- 빠른 응답 시간이 중요
- 비용을 최소화하려는 경우
- 디버깅과 모니터링이 중요

### 🤖 Agent를 사용하는 경우
```python
# ✅ 좋은 예: 복잡한 문제 해결
# 사용자 질문 → 검색 필요 여부 판단 → 적절한 도구 선택 → 결과 종합
agent = create_react_agent(
    model=llm,
    tools=[search_tool, calculator_tool, weather_tool]
)
```

**Agent 선택 기준:**
- 문제 해결 과정이 동적이고 복잡
- 여러 도구를 조합해야 하는 경우
- 상황에 따라 다른 접근이 필요
- 사용자 상호작용이 중요

---

## 📖 참고 자료

### 🔧 LangChain & LangGraph 공식 문서
- [LangChain 공식 문서](https://docs.langchain.com/)
- [LangGraph 공식 문서](https://langchain-ai.github.io/langgraph/)
- [LangChain Agent 가이드](https://python.langchain.com/docs/tutorials/agents)
- [LangChain classify](https://python.langchain.com/docs/tutorials/classification/)
- [LangChain summarization with map-reduce](https://python.langchain.com/docs/tutorials/summarization/)

### 🤖 Agent 시스템 관련
- [ReAct 논문](https://arxiv.org/abs/2210.03629) - Reasoning and Acting in Language Models
- [Tool Learning Survey](https://arxiv.org/abs/2304.08354) - Tool Learning with Foundation Models
- [LangChain Agent 비교](https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/) - 다양한 에이전트 타입 비교

### 🔍 RAG 및 검색 기법
- [RAG 논문](https://arxiv.org/abs/2005.11401) - Retrieval-Augmented Generation
- [Conversational RAG](https://python.langchain.com/docs/tutorials/qa_chat_history/#chains) - 대화형 검색 생성
- [Vector Store 비교](https://python.langchain.com/docs/integrations/vectorstores/) - 벡터 스토어 선택 가이드

### 📊 구조화된 출력 및 추출
- [Pydantic 공식 문서](https://docs.pydantic.dev/) - 데이터 검증 및 스키마
- [Function Calling 가이드](https://docs.langchain.com/docs/use-cases/extraction) - 구조화된 데이터 추출
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling) - OpenAI 함수 호출

### 🔄 비동기 및 병렬 처리
- [LangChain 비동기 가이드](https://python.langchain.com/docs/how_to/custom_tools/) - 비동기 툴 처리 패턴
- [Python asyncio 공식 문서](https://docs.python.org/3/library/asyncio.html) - 비동기 프로그래밍

---

## 🎓 학습 순서 추천

1. **🔰 기초**: `chatbot.py` → 기본 대화형 AI 구현
2. **🔍 검색**: `conversationalRAG.py` → 검색 기반 생성 시스템
3. **🤖 에이전트**: `agents.py` → 도구 기반 에이전트 시스템
4. **📊 분류**: `classification.py` → 텍스트 분류 방법론
5. **📝 요약**: `summarization.py` → 대규모 문서 요약
6. **🏷️ 추출**: `extraction.py` → 구조화된 정보 추출

각 단계마다 관련 가이드 문서(`*.md`)를 함께 참고하시면 더 깊이 있는 이해에 더 도움된다!
