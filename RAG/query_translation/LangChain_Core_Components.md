# LangChain Core Components Guide

## Overview
Query Translation과 RAG 구현에서 자주 사용되는 LangChain의 핵심 컴포넌트들에 대한 종합 가이드입니다. 실제 사용 예시와 함께 각 컴포넌트의 역할과 활용법을 다룹니다.

## 🔄 Document Serialization: `dumps` & `loads`

### 기본 개념
Document 객체를 문자열로 변환(직렬화)하고 다시 복원(역직렬화)하는 기능입니다.

```python
from langchain.load import dumps, loads
from langchain.schema import Document

# Document 객체 생성
doc = Document(
    page_content="This is the content of the document", 
    metadata={"source": "web", "page": 1}
)

# dumps: Document → 문자열
serialized = dumps(doc)
print(type(serialized))  # <class 'str'>

# loads: 문자열 → Document  
restored_doc = loads(serialized)
print(type(restored_doc))  # <class 'langchain.schema.Document'>
print(restored_doc.page_content)  # "This is the content of the document"
```

### Multi-Query RAG에서의 활용

#### 중복 제거 과정
```python
def get_unique_union(documents: list[list]):
    """
    여러 쿼리 결과에서 중복 문서 제거
    
    Input: [
        [Doc1, Doc2, Doc3],  # Query 1 결과
        [Doc2, Doc4, Doc5],  # Query 2 결과  
        [Doc1, Doc3, Doc6],  # Query 3 결과
    ]
    Output: [Doc1, Doc2, Doc3, Doc4, Doc5, Doc6]  # 중복 제거됨
    """
    
    # 1. 모든 Document를 문자열로 변환 (비교 가능하게 만들기)
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    
    # 2. set()으로 중복 제거 (문자열이므로 비교 가능)
    unique_docs = list(set(flattened_docs))
    
    # 3. 다시 Document 객체로 변환
    return [loads(doc) for doc in unique_docs]
```

#### 왜 이렇게 복잡하게?
```python
# ❌ Document 객체 직접 비교는 불가능
doc1 = Document(page_content="Same content", metadata={"source": "web"})
doc2 = Document(page_content="Same content", metadata={"source": "web"})
print(doc1 == doc2)  # False! (객체 참조가 다름)

# ✅ dumps로 문자열 변환 후 비교 가능
str1 = dumps(doc1)
str2 = dumps(doc2)
print(str1 == str2)  # True! (내용이 같으면 같은 문자열)
```

### 주요 사용 사례
- **중복 제거**: Multi-Query RAG에서 동일한 문서 제거
- **저장/로드**: 문서를 파일이나 데이터베이스에 저장
- **네트워크 전송**: API를 통해 문서 데이터 전송
- **캐싱**: 검색 결과를 캐시로 저장

## 🔗 Data Passing: `RunnablePassthrough`

### 기본 개념
입력 데이터를 변경하지 않고 그대로 출력으로 전달하는 컴포넌트입니다.

```python
from langchain_core.runnables import RunnablePassthrough

# 기본 사용법
passthrough = RunnablePassthrough()
result = passthrough.invoke("Hello World")
print(result)  # "Hello World" (그대로 출력)

# 파이프라인에서 사용
chain = (
    {"input": RunnablePassthrough(), "processed": some_processor}
    | next_step
)
```

### RAG 파이프라인에서의 활용

#### 단순한 RAG 체인
```python
# 문자열을 직접 전달하는 구조
simple_rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 사용 시
result = simple_rag_chain.invoke("What is task decomposition?")
```

#### 내부 동작 과정
```python
# 1. 입력: "What is task decomposition?"

# 2. 딕셔너리 구성:
{
    "context": retriever | format_docs,  # 자동으로 관련 문서 검색+포맷팅
    "question": RunnablePassthrough()    # 질문을 그대로 전달
}

# 3. 실제 실행 시:
{
    "context": "retrieved and formatted documents...",
    "question": "What is task decomposition?"  # 그대로 전달됨
}

# 4. 프롬프트 템플릿에 전달
```

### 제한사항과 대안

#### ❌ 복잡한 딕셔너리 구조에서는 사용 불가
```python
# Multi-Query RAG에서 문제 발생
final_rag_chain = (
    {"context": retrieval_chain, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)

# 호출 시
result = final_rag_chain.invoke({"question": "What is task decomposition?"})

# 문제: RunnablePassthrough가 전체 딕셔너리를 전달
# 프롬프트: "Question: {'question': 'What is task decomposition?'}" ❌
```

#### ✅ itemgetter 사용
```python
from operator import itemgetter

final_rag_chain = (
    {"context": retrieval_chain, "question": itemgetter("question")}
    | prompt | llm | StrOutputParser()
)

# 올바른 결과
# 프롬프트: "Question: What is task decomposition?" ✅
```

### 실제 생성되는 프롬프트 구조
```
System: You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:

# 딕셔너리에서 사용
data = {"question": "What is AI?", "context": "AI is..."}
getter = itemgetter("question")
result = getter(data)  # "What is AI?"

# 리스트에서 사용
my_list = ["a", "b", "c", "d"]
list_getter = itemgetter(0, 2)
result = list_getter(my_list)  # ("a", "c")

# 여러 키 동시 추출
multi_getter = itemgetter("question", "context")
result = multi_getter(data)  # ("What is AI?", "AI is...")
```

### vs dict.get() 비교

| 기능 | itemgetter | dict.get() |
|------|------------|------------|
| **기본 추출** | ✅ `itemgetter("key")(data)` | ✅ `data.get("key")` |
| **존재하지 않는 키** | ❌ KeyError 발생 | ✅ None 반환 (기본값 가능) |
| **여러 키 동시 추출** | ✅ `itemgetter("a", "b")` | ❌ 불가능 |
| **다양한 데이터 타입** | ✅ 딕셔너리, 리스트, 튜플 | ❌ 딕셔너리만 |
| **파이프라인 사용** | ✅ 직접 사용 가능 | ❌ lambda로 감싸야 함 |
| **성능** | 약간 느림 | 약간 빠름 |

### LangChain에서 선호하는 이유

#### 파이프라인 호환성
```python
# ✅ itemgetter (권장)
chain = (
    {"context": retrieval_chain, "question": itemgetter("question")}
    | prompt | llm
)

# 🔄 .get() (가능하지만 번거로움)
chain = (
    {"context": retrieval_chain, "question": lambda x: x.get("question")}
    | prompt | llm
)
```

## 📊 Component Comparison Table

| 컴포넌트 | 주요 기능 | 입력 타입 | 출력 타입 | 주요 사용처 |
|----------|-----------|-----------|-----------|-------------|
| **dumps** | Document → 문자열 | Document | str | 중복 제거, 저장 |
| **loads** | 문자열 → Document | str | Document | 복원, 로딩 |
| **RunnablePassthrough** | 데이터 그대로 전달 | Any | Same | 단순 파이프라인 |
| **itemgetter** | 키/인덱스로 값 추출 | dict/list/tuple | Any | 복잡한 파이프라인 |

## 🎯 Few-Shot Prompting: `FewShotChatMessagePromptTemplate`

### 기본 개념
Few-Shot Prompting은 AI 모델에게 원하는 출력 형태를 몇 개의 예시로 보여주는 기법입니다. LangChain에서는 `FewShotChatMessagePromptTemplate`을 사용해 체계적으로 구현할 수 있습니다.

### 구조 이해

#### 1. 기본 구성 요소
```python
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

# 1. 개별 예시 템플릿 정의
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}"),
])

# 2. 예시 데이터 준비
examples = [
    {"input": "질문1", "output": "답변1"},
    {"input": "질문2", "output": "답변2"},
]

# 3. Few-Shot 템플릿 생성
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)
```

#### 2. Step-back Prompting 실제 예시
```python
# 예시 데이터: 구체적 질문 → 일반적 질문
examples = [
    {
        "input": "Could the members of The Police perform lawful arrests?",
        "output": "what can the members of The Police do?",
    },
    {
        "input": "Jan Sindel's was born in what country?",
        "output": "what is Jan Sindel's personal history?",
    },
]

# 개별 예시 템플릿
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}"),
])

# Few-Shot 템플릿 생성
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

# 최종 프롬프트 구성
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:"),
    few_shot_prompt,  # 예시들이 여기에 삽입됨
    ("user", "{question}"),
])
```

## 🎯 실전 활용 가이드

### When to Use Each Component

#### `dumps` & `loads`
```python
# ✅ 사용하는 경우
- Multi-Query RAG에서 중복 문서 제거
- 문서를 파일이나 DB에 저장/로드
- API를 통한 문서 데이터 전송
- 검색 결과 캐싱

# ❌ 불필요한 경우  
- 단순한 RAG에서 문서 처리
- 실시간 처리만 하는 경우
```

#### `RunnablePassthrough`
```python
# ✅ 사용하는 경우
simple_chain = (
    {"input": RunnablePassthrough(), "processed": processor}
    | prompt | llm
)
simple_chain.invoke("직접 문자열 전달")

# ❌ 사용하면 안 되는 경우
complex_chain = (
    {"context": retrieval_chain, "question": RunnablePassthrough()}
    | prompt | llm
)
complex_chain.invoke({"question": "딕셔너리 전달"})  # 문제 발생!
```

#### `itemgetter`
```python
# ✅ 사용하는 경우
- 딕셔너리 기반 파이프라인
- 여러 값을 동시에 추출해야 할 때
- LangChain 표준 패턴 따를 때

# 🔄 대안 (.get())
- 에러 안전성이 필요할 때
- 기본값이 필요할 때
```

## 📝 Best Practices

### 1. **Multi-Query RAG Pattern**
```python
def get_unique_union(documents: list[list]):
    # dumps/loads로 중복 제거
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    unique_docs = list(set(flattened_docs))
    return [loads(doc) for doc in unique_docs]

final_rag_chain = (
    {"context": retrieval_chain | format_docs, 
     "question": itemgetter("question")}  # itemgetter 사용
    | prompt | llm | StrOutputParser()
)
```

### 2. **Simple RAG Pattern**
```python
simple_rag_chain = (
    {"context": retriever | format_docs, 
     "question": RunnablePassthrough()}  # RunnablePassthrough 사용
    | prompt | llm | StrOutputParser()
)
```

### 3. **Error-Safe Pattern**
```python
# 에러 안전성이 중요한 경우
safe_chain = (
    {"context": retrieval_chain | format_docs,
     "question": lambda x: x.get("question", "기본 질문")}
    | prompt | llm | StrOutputParser()
)
```

## 🔍 Debugging Tips

### Document Serialization Issues
```python
# 문제: Document 비교가 안 됨
if doc1 == doc2:  # ❌ 항상 False
    print("같은 문서")

# 해결: dumps로 비교
if dumps(doc1) == dumps(doc2):  # ✅ 내용 비교
    print("같은 문서")
```

### Pipeline Data Flow Issues
```python
# 문제: 딕셔너리 전체가 전달됨
{"question": RunnablePassthrough()}
# 결과: {"question": {"question": "actual question"}}

# 해결: itemgetter로 값만 추출
{"question": itemgetter("question")}
# 결과: {"question": "actual question"}
```

## 📚 Summary

### Key Takeaways

1. **`dumps`/`loads`**: Document 객체의 직렬화/역직렬화로 중복 제거와 저장에 필수
2. **`RunnablePassthrough`**: 단순한 파이프라인에서 데이터를 그대로 전달
3. **`itemgetter`**: 복잡한 딕셔너리 구조에서 값 추출, LangChain 표준 패턴

### 선택 가이드

```python
# 단순한 구조 → RunnablePassthrough
chain.invoke("문자열")

# 복잡한 구조 → itemgetter  
chain.invoke({"key": "value"})

# 중복 제거 → dumps/loads
unique_docs = get_unique_union(document_lists)
```

### 일반적인 실수

1. **RunnablePassthrough를 딕셔너리 구조에서 사용**
2. **Document 객체를 직접 비교**
3. **itemgetter 대신 복잡한 lambda 사용**

이 컴포넌트들을 올바르게 이해하고 사용하면 효율적이고 안정적인 RAG 시스템을 구축할 수 있습니다. 