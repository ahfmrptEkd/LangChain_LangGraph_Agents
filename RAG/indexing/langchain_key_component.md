# LangChain 핵심 컴포넌트: MultiVectorRetriever 메소드 비교

## 🔍 메소드 비교 분석

### 1. **get_relevant_documents** (기존 방식)
```python
# 매개변수 직접 전달
retrieved_docs = retriever.get_relevant_documents(query, n_results=1)
```

### 2. **invoke** (새로운 방식)
```python
# 설정 단계에서 search_kwargs로 매개변수 지정
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=id_key,
    search_kwargs={"k": 1}  # 여기서 매개변수 설정
)

# 쿼리 문자열만 전달
retrieved_docs = retriever.invoke(query)
```

## 🔄 주요 차이점

| 구분 | get_relevant_documents | invoke |
|------|----------------------|---------|
| **매개변수 전달 방식** | 호출 시 직접 전달 | 설정 단계에서 미리 지정 |
| **쿼리 형식** | 문자열 + 매개변수 | 문자열만 |
| **설정 시점** | 매번 호출 시 | 한 번만 설정 |
| **유연성** | 호출마다 다른 매개변수 가능 | 고정된 매개변수 사용 |

## 📊 내부 작동 원리

두 방식 모두 동일한 내부 프로세스를 따릅니다:

```
1. 쿼리 → vectorstore.similarity_search(query, **search_kwargs)
2. 요약 문서의 metadata에서 doc_ids 추출
3. 원본 문서 검색: docstore.get(doc_ids)
4. 원본 문서 반환
```

## 🎯 사용 예시

### 기존 방식 (get_relevant_documents)
```python
# 각 호출마다 다른 매개변수 사용 가능
docs1 = retriever.get_relevant_documents("쿼리1", n_results=1)
docs2 = retriever.get_relevant_documents("쿼리2", n_results=3)
```

### 새로운 방식 (invoke)
```python
# 한 번만 설정
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key="doc_id",
    search_kwargs={"k": 1}
)

# 여러 번 사용
docs1 = retriever.invoke("쿼리1")
docs2 = retriever.invoke("쿼리2")  # 모두 k=1로 동일하게 적용
```

## 💡 사용 권장사항

### **invoke 방식을 사용하는 경우** ✅
- 일관된 동작이 필요할 때
- 코드가 더 깔끔해짐
- LangChain 파이프라인과 잘 통합됨
- 미래 호환성 보장

### **get_relevant_documents 방식을 사용하는 경우** ⚠️
- 쿼리마다 다른 매개변수가 필요할 때
- 기존 레거시 코드와 호환성이 필요할 때

## 🔧 실제 적용 예시

```python
# 설정 단계
def setup_retriever():
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=store,
        id_key="doc_id",
        search_kwargs={"k": 1}  # 검색 결과 1개로 제한
    )
    return retriever

# 사용 단계
retriever = setup_retriever()
results = retriever.invoke("에이전트의 메모리 구조")
```

## 🎬 결론

**invoke 방식**이 LangChain의 현대적인 접근 방식이며, 대부분의 경우 이 방식을 사용하는 것이 좋습니다. 설정과 사용이 분리되어 있어 코드 관리가 용이하고, 다른 LangChain 컴포넌트와의 통합성도 뛰어납니다. 