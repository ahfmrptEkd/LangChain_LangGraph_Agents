# Knowledge-Augmented Generation (KAG)

이 디렉토리에는 **Neo4j 그래프 데이터베이스**를 활용한 Knowledge-Augmented Generation (KAG) 시스템의 실제 구현 예제가 포함되어 있다.

![kag overview](../src/imgs/kag2.png)

KAG는 기존 RAG(Retrieval-Augmented Generation)를 발전시킨 형태로, 구조화된 지식 그래프를 통해 더욱 정확하고 맥락적인 정보 검색 및 생성이 가능한 고급 AI 시스템입니다.

## 📚 목차

- [📂 프로젝트 구조](#-프로젝트-구조)
- [🎯 핵심 개념](#-핵심-개념)
- [🛠️ 실행 환경 설정](#️-실행-환경-설정)
- [구현된 접근 방식](#구현된-접근-방식)
- [KAG vs RAG: 핵심 차이점](#kag-vs-rag-핵심-차이점)
- [향후 확장 계획](#향후-확장-계획)
- [참고 자료](#-참고-자료)

## 📂 프로젝트 구조

```
KAG/
├── README.md                    # 프로젝트 설명
├── docker-compose.yml           # Neo4j Docker 환경 설정
├── Makefile                     # Docker 관리 명령어
├── run_example/                 # 실행 예제 디렉토리
│   ├── dynamic_cypher.py        # 동적 Cypher 쿼리 생성 시스템
│   └── semantic_layer.py        # semantic layer 기반 검색 시스템
└── .env.example                 # 환경 변수 설정 파일
```

## 🎯 핵심 개념

### Knowledge-Augmented Generation (KAG)

KAG는 **구조화된 지식 그래프**를 활용하여 정보 검색과 생성을 수행하는 고급 AI 기법:

- **🔗 그래프 기반 지식 표현**: 엔티티 간의 복잡한 관계를 그래프로 모델링
- **🧠 맥락적 추론**: 연결된 정보들을 통한 깊이 있는 추론
- **⚡ 동적 쿼리 생성**: 자연어 질문을 그래프 쿼리로 자동 변환
- **🎯 정확한 검색**: 벡터 유사도가 아닌 명확한 관계 기반 검색

### Neo4j 그래프 데이터베이스

```cypher
# 영화 도메인 그래프 구조
(Person)-[:ACTED_IN]->(Movie)-[:IN_GENRE]->(Genre)
(Person)-[:DIRECTED]->(Movie)
```

- **노드**: Movie, Person, Genre
- **관계**: ACTED_IN, DIRECTED, IN_GENRE
- **속성**: title, name, released, imdbRating

## 🛠️ 실행 환경 설정

### 1. Docker 환경 준비

```bash
# Neo4j 컨테이너 시작
make start

# 컨테이너 상태 확인
make status

# 로그 확인
make logs
```

### 2. 환경 변수 설정

```env
# .env 파일 생성
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Neo4j 브라우저 접속

- **URL**: http://localhost:7474
- **사용자명**: neo4j
- **비밀번호**: password

## 구현된 접근 방식

### 🔄 Dynamic Cypher 방식 (`dynamic_cypher.py`)

**복잡한 워크플로우**를 통한 자연어 → Cypher 쿼리 변환 시스템 (Text2Cypher).

```python
# 워크플로우 단계
1. 가드레일 체크 → 영화 관련 질문인지 판단
2. 쿼리 생성 → Text2Cypher 변환 (Few-shot 학습)
3. 쿼리 검증 → 문법 및 스키마 검증
4. 쿼리 수정 → 오류 발견 시 자동 수정
5. 쿼리 실행 → Neo4j 데이터베이스 실행
6. 결과 생성 → 자연어 답변 생성
```

**주요 특징**:
- **🛡️ 가드레일**: 영화와 무관한 질문 필터링
- **🔍 시맨틱 유사도 기반 예제 선택**: 가장 유사한 Cypher 예제 자동 선택
- **🔧 자동 오류 수정**: 문법 오류 및 스키마 불일치 자동 수정
- **📊 구조화된 출력**: Pydantic 모델 기반 타입 안전성

### 🎭 Semantic Layer 방식 (`semantic_layer.py`)

**도구 기반 접근**을 통한 직관적인 영화 정보 시스템입니다.

```python
# 사용 가능한 도구들
- InformationTool: 영화/배우 정보 검색
- RecommendGenreTool: 장르 기반 영화 추천  
- RecommendActorTool: 배우 기반 영화 추천
- RecommendPersonalizedTool: 개인화 추천 시스템
```

**주요 특징**:
- **🎯 직관적 도구**: 명확한 기능별 도구 분리
- **🤖 ReAct 패턴**: 추론-행동-관찰 사이클
- **📈 다양한 추천**: 장르, 배우, 개인화 추천 지원
- **🔗 자연스러운 대화**: 멀티턴 대화 지원

### 📊 Neo4j 브라우저 활용

```cypher
-- 데이터 확인
MATCH (n) RETURN count(n)

-- 영화 검색
MATCH (m:Movie {title: 'Casino'}) 
RETURN m

-- 관계 탐색
MATCH (p:Person)-[r:ACTED_IN]->(m:Movie) 
RETURN p.name, m.title 
LIMIT 10
```

## KAG vs RAG: 핵심 차이점

| 구분 | RAG (검색 증강 생성) | KAG (지식 증강 생성) |
|------|---------------------|---------------------|
| **데이터 구조** | 비구조화된 문서 | 구조화된 지식 그래프 |
| **검색 방식** | 벡터 유사도 기반 | 관계 기반 정확 검색 |
| **추론 능력** | 단순 유사도 매칭 | 복잡한 관계 추론 |
| **정확성** | 유사도 기반 부정확 | 명확한 관계 기반 정확 |
| **복잡한 질의** | 제한적 | 뛰어남 (다중 홉 쿼리) |
| **도메인 적응** | 어려움 | 스키마 기반 쉬운 확장 |

### 🎯 **KAG의 장점**

1. **정확한 관계 추론**: "Tom Hanks와 함께 출연한 배우들 중 Comedy 장르 영화에 3편 이상 출연한 사람은?"
2. **복잡한 조건 처리**: "감독이자 배우로도 활동한 사람들의 영화 목록"
3. **도메인 특화**: 명확한 스키마 기반 정확한 답변
4. **확장 가능성**: 새로운 노드/관계 타입 쉬운 추가

### ⚠️ **KAG의 한계**

1. **구조화 비용**: 그래프 구축에 많은 시간과 노력 필요
2. **도메인 제한**: 사전 정의된 스키마 범위 내에서만 동작  
3. **자연어 처리**: 복잡한 Text2Cypher 변환 과정 필요
4. **확장성**: 대규모 일반 도메인 적용 어려움

## 향후 확장 가능성

### ⚡ **성능 최적화**

```python
# 가능한 개선 방향들
- 쿼리 캐싱 시스템
- 병렬 처리 지원
- 실시간 스키마 업데이트
- 멀티모달 지식 그래프 (이미지, 텍스트 통합)
```

## 📚 참고 자료

- [Neo4j 공식 문서](https://neo4j.com/docs/)
- [Cypher 쿼리 언어 가이드](https://neo4j.com/developer/cypher/)
- [LangChain Neo4j 통합](https://python.langchain.com/docs/integrations/graphs/neo4j_cypher/)
- [KAG 논문](https://arxiv.org/pdf/2409.13731)
- [KAG Graph + Multimodal RAG + LLM Agents = Powerful AI Reasoning](https://pub.towardsai.net/kag-graph-multimodal-rag-llm-agents-powerful-ai-reasoning-b3da38d31358)
- [KAG(Knowledge Augmented Generation): LLM과 지식그래프의 강력한 조합!](https://papooo-dev.github.io/posts/KAG/#%EB%A7%88%EB%AC%B4%EB%A6%AC)
- [LangChain - Building QA application over graph DB](https://python.langchain.com/docs/tutorials/graph/)
- [LangChain - Techniques for implementing semantic layers](https://python.langchain.com/docs/how_to/graph_semantic/)
- [LangChain - Techniques for constructing knowledge graphs](https://python.langchain.com/docs/how_to/graph_constructing/)

