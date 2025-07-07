# 🚀 RAG Routing Systems

이 디렉토리에는 LangChain을 사용한 두 가지 다른 라우팅 시스템이 포함되어 있습니다.

## 📁 파일 구조

```
routing/
├── logical_routing.py     # 논리적 라우팅 시스템
├── sementic_routing.py    # 의미적 라우팅 시스템
└── README.md             # 이 파일
```

## 🎯 라우팅 시스템 비교

### 1. **Logical Routing (논리적 라우팅)**
- **파일**: `logical_routing.py`
- **방식**: 구조화된 출력을 사용한 LLM 기반 분류
- **기술**: Pydantic 모델 + Function Calling
- **장점**: 명확한 카테고리 분류, 높은 정확도
- **용도**: 명확한 카테고리가 있는 경우 (예: 프로그래밍 언어별 문서)

### 2. **Semantic Routing (의미적 라우팅)**
- **파일**: `sementic_routing.py`
- **방식**: 임베딩 기반 코사인 유사도 계산
- **기술**: OpenAI Embeddings + Cosine Similarity
- **장점**: 의미적 유사성 기반, 빠른 처리
- **용도**: 의미적 유사성이 중요한 경우 (예: 주제별 전문가 시스템)

## 🔧 설치 및 설정

### 필요한 패키지

```bash
pip install langchain langchain-openai langchain-core python-dotenv
```

### 환경 변수 설정

프로젝트 루트에 `.env` 파일을 생성하고 다음을 추가하세요:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

## 🎨 주요 특징

### Logical Routing 특징
- 🎯 **RouteQuery 클래스**: Pydantic 기반 구조화된 데이터 모델
- 🎯 **Function Calling**: GPT-4o-mini의 구조화된 출력 기능 활용
- 🎯 **명확한 분류**: python_docs, js_docs, golang_docs 중 선택

### Semantic Routing 특징
- 🧠 **임베딩 기반**: OpenAI Embeddings로 의미적 유사성 계산
- 🧠 **코사인 유사도**: 수학적 정확성을 통한 최적 템플릿 선택
- 🧠 **전문가 시스템**: Physics Professor vs Mathematics Expert

## 🤔 언제 어떤 시스템을 사용할까?

### Logical Routing을 사용하는 경우:
- 명확한 카테고리가 있을 때
- 정확한 분류가 중요할 때
- 복잡한 조건부 로직이 필요할 때

### Semantic Routing을 사용하는 경우:
- 의미적 유사성이 중요할 때
- 빠른 응답이 필요할 때
- 전문가 시스템을 구축할 때

## 🔧 커스터마이징

### Logical Routing 커스터마이징
```python
# 새로운 데이터소스 추가
datasource: Literal["python_docs", "js_docs", "golang_docs", "new_docs"]

# choose_route 함수에 새로운 로직 추가
def choose_route(result):
    if "new_docs" in result.datasource.lower():
        return "chain for new_docs"
    # ... 기존 로직
```

### Semantic Routing 커스터마이징
```python
# 새로운 전문가 템플릿 추가
new_expert_template = """You are an expert in...
Here is a question:
{query}
"""

# 템플릿 목록에 추가
prompt_templates = [physics_template, math_template, new_expert_template]
```

### 성능 최적화
- **임베딩 캐싱**: 반복 사용시 임베딩 결과 캐싱 고려
- **배치 처리**: 대량 쿼리 처리시 배치 처리 구현
- **모델 선택**: 용도에 따라 적절한 OpenAI 모델 선택

## 📚 추가 학습 자료

- [LangChain 공식 문서](https://python.langchain.com/docs/get_started/introduction)
- [OpenAI API 문서](https://platform.openai.com/docs/api-reference)
- [Pydantic 문서](https://docs.pydantic.dev/)

---

💡 **Tip**: 실제 프로덕션 환경에서는 두 시스템을 조합하여 사용하는 것도 좋은 방법입니다! 