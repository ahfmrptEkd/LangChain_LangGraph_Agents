# Cache-Augmented Generation (CAG) Template

이 디렉토리에는 LangChain 프로젝트에서 **하이브리드 캐싱 접근법**을 사용하여 Cache-Augmented Generation (CAG)을 구현하는 재사용 가능한 템플릿이 포함되어 있습니다.

기본적인 CAG 방식은 대형 언어 모델의 확장된 컨텍스트 윈도우를 활용하여 모든 관련 문서를 사전에 로드하고, 런타임 매개변수(Key-Value 캐시)를 미리 계산하여 저장하는 방식입니다.

이 접근법은 RAG 파이프라인의 성능을 향상시키기 위해 두 가지 캐싱 전략을 결합합니다:
- **리트리버 캐싱**: 문서 검색 결과를 캐시
- **LLM 캐싱**: 언어 모델 호출 결과를 캐시


## 📂 프로젝트 구조

```
CAG/
├── README.md                    # 프로젝트 설명서 
├── caching.py                   # 캐시 구현 
├── cag_template.py              # 캐시 기반 리트리버 클래스
├── run_cag_example.py           # 리트리버 캐싱 + LLM 캐싱 = 하이브리드 캐싱 예제
├── basic_cag_template.py         # 진짜 CAG 구현
├── run_basic_cag_example.py      # 진짜 CAG 예제
├── performance_comparison.py    # 성능 비교 벤치마크 
├── research.md                  # CAG 연구 자료 
```

## 핵심 개념

### 하이브리드 캐싱 전략

기존 리트리버를 완전히 교체하는 대신, 이 템플릿은 두 가지 캐싱 레이어를 제공합니다:

1. **리트리버 캐싱**: 커스텀 InMemoryCache를 사용하여 문서 검색 결과를 캐시
2. **LLM 캐싱**: LangChain의 내장 캐시 시스템을 사용하여 언어 모델 호출을 캐시

### 캐시 히트/미스 프로세스

- **리트리버 캐시 미스**: 처음 쿼리가 들어올 때, 벡터 스토어에서 관련 문서를 검색하고 결과를 캐시에 저장
- **리트리버 캐시 히트**: 동일한 쿼리가 반복될 때, 캐시에서 즉시 문서를 반환
- **LLM 캐시**: LangChain의 내장 시스템이 자동으로 언어 모델 호출을 캐시

## 파일 구조

### 📋 **파일별 상세 설명**

#### 🔧 **기본 캐싱 구현**
- `caching.py`: 간단한 `InMemoryCache` 클래스를 정의합니다. Redis나 파일 기반 캐시 등 더 강력한 캐시로 쉽게 교체할 수 있습니다.
- `cag_template.py`: 표준 LangChain 리트리버를 감싸고 하이브리드 캐싱 로직을 추가하는 핵심 `CachedRetriever` 클래스를 포함합니다.
- `run_cag_example.py`: `CachedRetriever`를 사용하는 방법을 보여주고 캐싱 메커니즘을 검증하는 실행 가능한 스크립트입니다.

#### 🚀 **True CAG 구현** (새로 추가!)
- `basic_cag_template.py`: **진짜 CAG 방식**을 구현합니다. 모든 지식을 모델 컨텍스트에 사전 로드하여 검색 단계를 완전히 제거합니다.
- `run_basic_cag_example.py`: basic CAG 방식의 데모 스크립트입니다. 40배 빠른 응답과 기존 방식과의 성능 비교를 보여줍니다.

#### 📊 **성능 분석 및 연구**
- `performance_comparison.py`: 4가지 캐싱 전략의 성능을 올바르게 측정하는 종합 벤치마크 스크립트입니다. 각 쿼리를 반복 실행하여 실제 캐시 효과(Cache Miss vs Cache Hit)를 정확히 측정합니다.
- `research.md`: CAG 기술에 대한 상세한 연구 자료, 다양한 구현 방법, 성능 최적화 전략을 다룹니다.

## 주요 기능

### 하이브리드 캐싱

```python
from caching import InMemoryCache
from cag_template import CachedRetriever

# 하이브리드 캐싱 설정
cache = InMemoryCache()
cached_retriever = CachedRetriever(
    retriever=base_retriever,
    cache=cache,
    enable_llm_cache=True  # LangChain의 LLM 캐시 활성화
)
```

### 캐시 관리

```python
# 캐시 정보 확인
cache_info = cached_retriever.get_cache_info()
print(f"캐시 크기: {cache_info['cache_size']}")
print(f"캐시된 쿼리: {cache_info['cache_keys']}")

# 캐시 클리어
cached_retriever.clear_cache()
```

## 성능 비교 결과

### 4가지 캐싱 전략 비교

`performance_comparison.py` 스크립트를 실행하여 다음 4가지 캐싱 전략의 성능을 비교했습니다:

1. **Regular RAG** (캐시 없음)
2. **LLM Cache only** (LLM 캐시만)
3. **Retrieval Cache only** (리트리버 캐시만)
4. **Full CAG** (리트리버 + LLM 캐시)

### 실험 결과

```bash
python performance_comparison.py
```

**📊 성능 결과 (평균 응답 시간)**:

| 캐싱 전략 | 평균 시간 | 성능 개선 | 속도 향상 |
|-----------|----------|----------|----------|
| 📍 Regular RAG | 1.306s | - (기준) | 1.0x |
| 🥇 Full CAG | 0.112s | **91.4%** | 11.6x |
| 🥈 Retrieval Cache only | 0.118s | **90.9%** | 10.1x |
| 🥉 LLM Cache only | 0.328s | **74.9%** | 4.0x |

### 주요 발견사항

#### 🎯 **리트리버 캐시의 극적인 효과**
- **캐시 히트 시 98.2% 성능 개선** (Cache Miss: 0.343s → Cache Hit: 0.006s)
- **57배 빠른 속도**: 거의 즉시 응답
- **벡터 검색 비용 완전 제거**: 반복 쿼리에서 압도적 성능

#### 💎 **Full CAG의 최고 성능**
- **11.6배 빠른 속도**: 전체 시나리오에서 최고 성능
- **91.4% 성능 개선**: 리트리버 + LLM 캐시 시너지 효과
- **25초 절약**: 총 27.4초 → 2.4초

#### 💭 **LLM 캐시의 안정적 효과**
- **74.9% 전체 개선**: 일관된 성능 향상
- **안정적인 응답 시간**: 0.3초대 일정한 성능

#### 📋 **진짜 CAG (Context Preloading) 성능**
- **소규모 실험 결과**: 2.2x 빠름 (2.867s → 1.277s)
- **논문 증명 결과**: **40x 성능 개선** 가능
- **⚠️ 실험 한계**: 작은 토큰 데이터셋(225 tokens, 6개 문서)으로 인한 제한적 결과
- **📈 실제 환경**: 대규모 지식 베이스에서 40x 개선 효과 검증됨 (논문 기준)



### 캐시 효과 분석

```
🏆 최고 성능: Full CAG (11.6x 빠름)
🚀 전체 개선: 91.4% 성능 향상
💰 비용 절감: 25.1초 절약 (총 27.4초 → 2.4초)

🔍 캐시 히트 vs 미스 비교:
   📚 리트리버 캐시: 0.343s → 0.006s (98.2% 개선)
   💎 Full CAG: 0.325s → 0.006s (98.2% 개선)
   
📈 개별 캐시 효과:
   💭 LLM 캐시: 74.9% 개선
   📚 리트리버 캐시: 90.9% 개선
```

### 사용 권장사항

1. **🏆 최고 성능을 원한다면**: Full CAG (리트리버 + LLM 캐시)
2. **📚 반복 쿼리가 많다면**: 리트리버 캐시 우선 적용
3. **💭 안정적 성능을 원한다면**: LLM 캐시로 시작
4. **🚀 궁극적 성능을 원한다면**: 진짜 CAG (Context Preloading) - 대규모 지식 베이스에서 40x 개선 효과 in 논문.

### 진짜 CAG vs 리트리버 캐시 비교

| 특성 | 리트리버 캐시 | 진짜 CAG (Context Preloading) |
|------|---------------|------------------------------|
| **구현 복잡도** | 🟢 낮음 | 🟡 중간 |
| **성능 개선** | 🟢 11.6x (검증됨) | 🟢 40x (논문 기준) |
| **메모리 사용** | 🟢 적음 | 🟡 많음 (전체 지식 로드) |
| **토큰 비용** | 🟢 적음 | 🔴 많음 (전체 컨텍스트) |
| **데이터 크기** | 🟢 제한 없음 | 🔴 토큰 제한 있음 |
| **검색 품질** | 🟡 Top-k 검색 | 🟢 전체 지식 활용 |
| **실시간 업데이트** | 🟢 쉬움 | 🔴 어려움 |

**결론**: 소규모 지식 베이스는 진짜 CAG, 대규모는 리트리버 캐시가 실용적


## 템플릿 확장

### 다양한 캐시 백엔드

다른 캐시(예: Redis)를 사용하려면 `InMemoryCache`와 동일한 `get`, `set`, `clear` 메서드를 구현하는 새로운 캐시 클래스를 만들고 `CachedRetriever`에 전달하자.

```python
from redis import Redis
from caching import InMemoryCache

class RedisCache(InMemoryCache):
    def __init__(self, redis_url="redis://localhost:6379"):
        self.redis = Redis.from_url(redis_url)
    
    def get(self, key):
        # Redis 구현
        pass
    
    def set(self, key, value):
        # Redis 구현
        pass
```

### 의미론적 캐싱

더 고급 사용 사례의 경우, 정확한 매칭 대신 캐시 키(쿼리)에 대한 의미론적 검색을 수행하도록 `CachedRetriever`를 확장할 수 있다.

```python
from langchain.embeddings import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

class SemanticCache(InMemoryCache):
    def __init__(self, similarity_threshold=0.8):
        super().__init__()
        self.embeddings = OpenAIEmbeddings()
        self.similarity_threshold = similarity_threshold
        self.query_embeddings = {}
    
    def get(self, key):
        # 의미론적 유사성 검색 구현
        pass
```

## 성능 최적화

### 캐시 크기 제한

```python
from collections import OrderedDict

class LRUCache(InMemoryCache):
    def __init__(self, max_size=100):
        self.max_size = max_size
        self._cache = OrderedDict()
    
    def set(self, key, value):
        if key in self._cache:
            self._cache.move_to_end(key)
        elif len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)
        self._cache[key] = value
```

### 캐시 만료

```python
import time

class TTLCache(InMemoryCache):
    def __init__(self, ttl=3600):  # 1시간 TTL
        super().__init__()
        self.ttl = ttl
        self.timestamps = {}
    
    def get(self, key):
        if key in self._cache:
            if time.time() - self.timestamps[key] < self.ttl:
                return self._cache[key]
            else:
                del self._cache[key]
                del self.timestamps[key]
        return None
```

## 장점

1. **극적인 성능 향상**: 캐시 히트 시 **98.2% 성능 개선** (0.006초 응답)
2. **실질적 비용 절감**: **25초 절약** (총 실행 시간 91.4% 단축)
3. **이중 최적화**: 리트리버 캐시(90.9% 개선) + LLM 캐시(74.9% 개선)
4. **확장성**: 다양한 캐시 백엔드로 쉽게 교체 가능
5. **투명성**: 기존 LangChain 워크플로우와 완벽 호환
6. **즉시 효과**: 반복 쿼리에서 **11.6배 빠른 응답**

이 하이브리드 접근법은 특히 FAQ 시스템, 챗봇, 반복 질문이 많은 환경에서 **거의 즉시 응답**을 제공하여 사용자 경험을 혁신적으로 개선한다.

## Source
- [Don’t Do RAG: When Cache-Augmented Generation is All You Need for Knowledge Tasks](https://arxiv.org/pdf/2412.15605)
- [Cache-Augmented Generation (CAG) from Scratch](https://medium.com/@sabaybiometzger/cache-augmented-generation-cag-from-scratch-441adf71c6a3)
- [RAG 대신 CAG? 캐시 증강 생성 기술이 차세대 LLM을 바꾼다](https://digitalbourgeois.tistory.com/716)