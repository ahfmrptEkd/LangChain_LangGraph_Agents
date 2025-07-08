# Query Translation for RAG

## 🎯 개요

Query Translation은 사용자의 원본 질문을 다양한 형태로 변환하여 RAG 시스템의 검색 성능을 향상시키는 기법입니다. 단일 질문으로는 놓칠 수 있는 관련 정보들을 다각도로 접근하여 더 풍부하고 정확한 답변을 생성할 수 있습니다.

## 📚 구현된 기법들

### 1. 🔄 Multi-Query RAG
**파일**: `multi_query.py`

**핵심 아이디어**: 하나의 질문을 5개의 다른 관점으로 변환하여 검색

```python
# 원본 질문
"What is task decomposition for LLM agents?"

# 생성된 다중 질문들
1. "How do LLM agents break down complex tasks?"
2. "What are the steps involved in task decomposition for AI agents?"
3. "How do large language models divide tasks into subtasks?"
4. "What methods do LLM agents use to decompose tasks?"
5. "How is task splitting implemented in LLM-based agents?"
```

**장점**:
- 다양한 관점에서 정보 수집
- 간단한 구현
- 7개의 고유 문서 검색으로 풍부한 컨텍스트 제공

**사용 사례**: 포괄적인 정보가 필요한 일반적인 질문

---

### 2. 🎯 RAG-Fusion
**파일**: `rag_fusion.py`

**핵심 아이디어**: Reciprocal Rank Fusion (RRF) 알고리즘을 사용한 정교한 문서 순위 결정

```python
# RRF 공식
score = 1 / (rank + k)  # k=60이 기본값
```

**특징**:
- 여러 쿼리 결과를 점수 기반으로 재순위화
- 6개 문서로 더 정교한 관련성 기반 정렬
- Multi-Query보다 정밀한 문서 선별

**장점**:
- 높은 정확도
- 관련성 기반 문서 정렬
- 중복 문서의 가중치 증가

**사용 사례**: 정확하고 정밀한 답변이 필요한 전문적인 질문

---

### 3. 🔧 Query Decomposition
**파일**: `decomposition.py`

**핵심 아이디어**: 복잡한 질문을 3개의 하위 질문으로 분해하여 단계별 답변

**두 가지 접근법**:

#### Recursive Approach (순차적)
```python
# 이전 답변을 다음 질문의 컨텍스트로 활용
Q1 → A1 → Q2 (with A1 context) → A2 → Q3 (with A1+A2 context) → A3
```

#### Individual Approach (독립적)
```python
# 각 질문을 독립적으로 답변 후 종합
Q1 → A1
Q2 → A2  
Q3 → A3
→ Synthesis(A1, A2, A3)
```

**장점**:
- 복잡한 질문의 체계적 분해
- 단계별 논리적 추론
- 두 가지 접근법 비교 가능

**사용 사례**: 다단계 추론이 필요한 복잡한 질문

---

### 4. 🔄 Step-back Prompting
**파일**: `step_back.py`

**핵심 아이디어**: 구체적인 질문을 더 일반적이고 광범위한 질문으로 변환

```python
# Few-shot 예시 활용
원본: "Could the members of The Police perform lawful arrests?"
Step-back: "what can the members of The Police do?"

원본: "What is task decomposition for LLM agents?"
Step-back: "What are the general approaches to task management in AI systems?"
```

**특징**:
- 이중 컨텍스트 활용 (구체적 + 일반적)
- Few-shot learning을 통한 패턴 학습
- 컨텍스트 중복도 분석

**장점**:
- 포괄적인 배경 정보 제공
- 구체적 질문의 맥락 확장
- 더 풍부한 답변 생성

**사용 사례**: 배경 지식이 필요한 구체적인 질문

---

### 5. 🔮 HyDE (Hypothetical Document Embeddings)
**파일**: `HyDE.py`

**핵심 아이디어**: 가상의 답변 문서를 생성하여 검색에 활용

```python
# 프로세스
질문 → 가상 문서 생성 → 가상 문서로 검색 → 실제 문서 검색 → 최종 답변
```

**특징**:
- 학술적 스타일의 가상 문서 생성
- 어휘 격차 해결
- 의미적 유사성 기반 검색

**장점**:
- 질문과 문서 간 용어 차이 극복
- 도메인 특화 언어 활용
- 더 관련성 높은 문서 검색

**사용 사례**: 전문 용어가 많거나 질문 표현이 문서와 다른 경우

---

## 🏗️ 파일 구조

```
query_translation/
├── README.md                           # 이 파일
├── multi_query.py                      # Multi-Query RAG 구현
├── rag_fusion.py                       # RAG-Fusion 구현
├── decomposition.py                    # Query Decomposition 구현
├── step_back.py                        # Step-back Prompting 구현
├── HyDE.py                            # HyDE 구현
└── LangChain_Core_Components.md       # 핵심 컴포넌트 가이드
```

## 🚀 실행 방법

### 환경 설정
```bash
# 가상환경 활성화
source .venv/bin/activate

# 환경변수 설정 (.env 파일에 OPENAI_API_KEY 필요)
```

### 개별 기법 실행
```bash
# Multi-Query RAG
python multi_query.py

# RAG-Fusion
python rag_fusion.py

# Query Decomposition
python decomposition.py

# Step-back Prompting
python step_back.py

# HyDE
python HyDE.py
```

## 📊 기법별 비교

| 기법 | 질문 변환 방식 | 검색 문서 수 | 주요 장점 | 적용 사례 |
|------|----------------|--------------|-----------|-----------|
| **Multi-Query** | 1→5 다양한 관점 | 7개 (고유) | 포괄적 정보 수집 | 일반적 질문 |
| **RAG-Fusion** | 1→5 + RRF 순위화 | 6개 (정밀) | 높은 정확도 | 전문적 질문 |
| **Decomposition** | 1→3 하위 질문 | 가변적 | 논리적 추론 | 복잡한 질문 |
| **Step-back** | 1→2 (구체→일반) | 이중 컨텍스트 | 배경 지식 제공 | 맥락 확장 |
| **HyDE** | 1→가상문서→검색 | 표준 검색 | 어휘 격차 해결 | 전문 용어 질문 |

## 💡 선택 가이드

### 질문 유형별 추천

**📋 일반적인 정보 수집**
- **Multi-Query**: 광범위한 정보 수집이 필요한 경우
- 예: "What is machine learning?"

**🎯 정확한 답변이 필요한 경우**
- **RAG-Fusion**: 정밀하고 관련성 높은 답변
- 예: "How does gradient descent work in neural networks?"

**🔧 복잡한 문제 해결**
- **Decomposition**: 단계별 접근이 필요한 경우
- 예: "How to implement a complete RAG system?"

**🌍 배경 지식이 필요한 경우**
- **Step-back**: 구체적 질문에 맥락 제공
- 예: "What are the latest developments in transformer architecture?"

**📚 전문 용어 질문**
- **HyDE**: 질문과 문서 간 용어 차이 극복
- 예: "Explain the implications of attention mechanisms in NLP"

## 🔧 핵심 컴포넌트

### dumps & loads
```python
# Document 객체 직렬화/역직렬화
from langchain.load import dumps, loads

# 중복 제거에 활용
unique_docs = list(set([dumps(doc) for doc in all_docs]))
restored_docs = [loads(doc) for doc in unique_docs]
```

### itemgetter vs RunnablePassthrough
```python
# 복잡한 딕셔너리 구조 → itemgetter 사용
{"context": retrieval_chain, "question": itemgetter("question")}

# 단순한 문자열 전달 → RunnablePassthrough 사용
{"context": retriever, "question": RunnablePassthrough()}
```

### Few-Shot Prompting
```python
# FewShotChatMessagePromptTemplate 활용
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)
```

## 🎓 학습 포인트

### 1. **자동검색 vs 수동검색**
- **자동검색**: 파이프라인에 retriever 포함
- **수동검색**: 개발자가 직접 `retriever.invoke()` 호출

### 2. **중복 제거의 중요성**
- Document 객체는 직접 비교 불가
- `dumps`로 문자열 변환 후 `set()`으로 중복 제거 필요

### 3. **프롬프트 엔지니어링**
- Few-shot learning으로 패턴 학습
- 구체적 → 일반적 변환 패턴
- 다양한 관점 생성 기법

### 4. **검색 성능 최적화**
- 다중 쿼리를 통한 recall 향상
- RRF를 통한 precision 향상
- 가상 문서를 통한 semantic gap 해결

## 🔍 참고 자료

- [LangChain Core Components Guide](./LangChain_Core_Components.md)
- [Multi-Query RAG 논문](https://arxiv.org/abs/2305.14283)
- [RAG-Fusion 개념](https://github.com/Raudaschl/rag-fusion)
- [HyDE 논문](https://arxiv.org/abs/2212.10496)

---

**💡 Tip**: 각 기법은 독립적으로 사용 가능하며, 필요에 따라 조합하여 사용할 수도 있습니다. 질문의 복잡도와 도메인에 따라 적절한 기법을 선택하세요!
