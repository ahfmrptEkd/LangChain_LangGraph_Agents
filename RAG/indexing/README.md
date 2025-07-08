# RAG Indexing Methods 📚

이 디렉토리는 **세 가지 고급 RAG 인덱싱 방법론**을 구현한 Python 코드들을 포함합니다. 각 방법론은 서로 다른 접근법으로 문서 검색의 성능을 향상시킵니다.

## 📖 목차

1. [Multi-Representation RAG](#1-multi-representation-rag)
2. [RAPTOR (Recursive Abstractive Processing)](#2-raptor-recursive-abstractive-processing)
3. [ColBERT RAG](#3-colbert-rag)
4. [방법론 비교](#4-방법론-비교)
5. [사용 가이드](#5-사용-가이드)

---

## 1. Multi-Representation RAG

### 📋 개요
문서의 **요약본을 벡터화하여 검색**하지만, **원본 문서를 반환**하는 방식입니다. 이는 검색 정확도를 높이면서도 완전한 정보를 제공합니다.

### 🎯 핵심 특징

- **이중 저장소 구조**: 
  - 벡터스토어: 문서 요약본 저장
  - 바이트스토어: 원본 문서 저장
- **LLM 기반 요약**: GPT-4o-mini를 사용한 지능형 요약
- **MultiVectorRetriever**: LangChain의 고급 검색기 사용
- **웹 문서 로더**: 실시간 웹 페이지 로딩 기능

### 🔧 작동 방식

```python
# 1. 문서 로드 및 요약 생성
docs = load_documents()  # 웹에서 문서 로드
summaries = create_summaries(docs)  # LLM으로 요약 생성

# 2. 이중 저장소 설정
vectorstore = Chroma()  # 요약본 벡터화
store = InMemoryByteStore()  # 원본 문서 저장

# 3. 검색 시 프로세스
# 쿼리 → 요약본 벡터 검색 → 원본 문서 반환
```

### 💡 장점
- 요약본으로 검색하여 **의미적 정확도 향상**
- 원본 문서 반환으로 **정보 손실 없음**
- 검색 속도와 정확도의 균형

### 📊 적용 사례
- 긴 문서의 효율적 검색
- 기술 문서 및 연구 논문
- 뉴스 아티클 검색 시스템

---

## 2. RAPTOR (Recursive Abstractive Processing)

### 📋 개요
문서들을 **계층적으로 클러스터링**하고 **각 레벨에서 요약**하여 트리 구조로 조직화하는 방법입니다. 다양한 추상화 레벨에서 검색이 가능합니다.

### 🎯 핵심 특징

- **계층적 클러스터링**: 문서를 다단계로 그룹화
- **GMM 클러스터링**: Gaussian Mixture Model로 최적 클러스터 수 결정
- **UMAP 차원축소**: 고차원 임베딩을 효율적으로 축소
- **재귀적 요약**: 각 클러스터를 상위 레벨로 요약
- **트리 구조**: 문서 계층을 트리로 구조화

### 🔧 작동 방식

```python
# 1. 임베딩 및 차원축소
embeddings = embed_texts(texts)
reduced_embeddings = global_cluster_embeddings(embeddings)

# 2. 계층적 클러스터링
clusters = perform_clustering(embeddings, dim, threshold)

# 3. 재귀적 요약
level_1_summaries = summarize_clusters(clusters)
level_2_summaries = summarize_clusters(level_1_summaries)
# ... 계속 반복

# 4. 트리 구조 생성
raptor_tree = build_raptor_tree(all_levels)
```

### 💡 장점
- **다단계 검색**: 세부사항부터 전체 개요까지
- **의미적 그룹화**: 관련 문서들의 자동 클러스터링
- **확장성**: 대량 문서 처리에 효과적
- **계층적 정보 액세스**: 필요한 추상화 레벨 선택 가능

### 📊 적용 사례
- 대규모 문서 컬렉션 구조화
- 연구 논문 데이터베이스
- 법률 문서 시스템
- 기업 지식 관리 시스템

---

## 3. ColBERT RAG

### 📋 개요
**토큰 레벨 상호작용**을 기반으로 한 고성능 검색 모델입니다. 각 토큰을 독립적으로 임베딩하여 더 정교한 검색을 제공합니다.

### 🎯 핵심 특징

- **토큰 레벨 임베딩**: 각 토큰을 개별적으로 벡터화
- **지연 상호작용**: 검색 시점에 토큰 간 상호작용 계산
- **RAGatouille 통합**: ColBERT 모델의 쉬운 사용
- **의존성 오류 처리**: 강건한 에러 핸들링
- **위키피디아 API**: 실시간 문서 로딩

### 🔧 작동 방식

```python
# 1. ColBERT 모델 초기화
retriever = ColBERTRetriever(model_name="colbert-ir/colbertv2.0")

# 2. 문서 인덱싱
documents = chunk_document(wiki_content)
retriever.index_documents(documents, "wiki_index")

# 3. 토큰 레벨 검색
# 쿼리 토큰 ↔ 문서 토큰 상호작용 계산
results = retriever.search(query, k=5)
```

### 💡 장점
- **높은 검색 정확도**: 토큰 레벨 정밀 매칭
- **효율적인 인덱싱**: 지연 상호작용으로 저장 공간 효율성
- **다국어 지원**: 다양한 언어에서 안정적 성능
- **실시간 검색**: 빠른 응답 시간

### 📊 적용 사례
- 정밀 검색이 필요한 시스템
- 다국어 문서 검색
- 학술 연구 플랫폼
- 전문 용어 검색 시스템

---

## 4. 방법론 비교

| 특징 | Multi-Representation | RAPTOR | ColBERT |
|------|---------------------|---------|---------|
| **검색 방식** | 요약본 → 원본 반환 | 계층적 클러스터 검색 | 토큰 레벨 상호작용 |
| **처리 속도** | 빠름 | 중간 | 빠름 |
| **정확도** | 높음 | 매우 높음 | 매우 높음 |
| **메모리 사용** | 중간 | 높음 | 중간 |
| **복잡성** | 낮음 | 높음 | 중간 |
| **확장성** | 중간 | 높음 | 높음 |
| **적용 분야** | 일반적 검색 | 대규모 문서 구조화 | 정밀 검색 |

### 🎯 선택 가이드

- **Multi-Representation**: 빠른 구현과 안정적 성능이 필요한 경우
- **RAPTOR**: 대량 문서의 계층적 구조화가 필요한 경우  
- **ColBERT**: 최고 정확도의 검색이 필요한 경우

---

### ⚠️ 주의사항

- **API 키**: OpenAI API 키가 필요합니다
- **메모리**: RAPTOR는 대량 메모리를 사용할 수 있습니다
- **의존성**: ColBERT는 특정 버전의 transformers가 필요합니다

---

## 📚 추가 자료

- [chunking_methods.md](./chunking_methods.md): 5가지 청킹 방법론 상세 가이드
- [langchain_key_component.md](./langchain_key_component.md): LangChain 핵심 컴포넌트 비교
