# AsyncIO와 LangChain을 이용한 병렬 처리 가이드

## 개요

이 문서는 Python의 asyncio와 LangChain을 결합하여 효율적인 병렬 처리를 구현하는 방법을 다룹니다. 대용량 문서 처리, 다중 API 호출, 체인 병렬 실행 등의 시나리오에서 성능을 최적화하는 방법을 설명합니다.

## 목차

1. [병렬 처리 패턴](#병렬-처리-패턴)
2. [LangChain과 AsyncIO 통합](#langchain과-asyncio-통합)
3. [LangGraph 병렬 처리](#langgraph-병렬-처리)
4. [성능 최적화 팁](#성능-최적화-팁)
5. [결론](#결론)

## 병렬 처리 패턴

### 1. asyncio.gather() - 모든 작업 완료 대기

```python
async def gather_pattern(documents: List[Document]) -> List[str]:
    """모든 문서가 처리될 때까지 기다리는 패턴"""
    processor = AsyncLLMProcessor()
    tasks = [processor.process_document(doc) for doc in documents]
    
    # 모든 작업이 완료될 때까지 대기
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results
```

### 2. asyncio.as_completed() - 완료되는 대로 처리

```python
async def as_completed_pattern(documents: List[Document]) -> List[str]:
    """완료되는 대로 결과를 처리하는 패턴"""
    processor = AsyncLLMProcessor()
    tasks = [processor.process_document(doc) for doc in documents]
    results = []
    
    # 완료되는 대로 결과 수집
    for coro in asyncio.as_completed(tasks):
        try:
            result = await coro
            results.append(result)
            print(f"완료된 작업 수: {len(results)}/{len(documents)}")
        except Exception as e:
            results.append(f"에러: {str(e)}")
    
    return results
```

### 3. asyncio.Semaphore - 동시 실행 제한

```python
import asyncio
from typing import List

class RateLimitedProcessor:
    def __init__(self, max_concurrent: int = 5):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.llm = OpenAI()
    
    async def process_with_limit(self, document: Document) -> str:
        """세마포어를 사용하여 동시 실행 수를 제한"""
        async with self.semaphore:
            try:
                result = await self.llm.ainvoke(
                    f"요약해주세요: {document.page_content}"
                )
                return result
            except Exception as e:
                return f"처리 실패: {str(e)}"
    
    async def process_documents_with_limit(self, documents: List[Document]) -> List[str]:
        """제한된 동시 실행으로 문서들을 처리"""
        tasks = [self.process_with_limit(doc) for doc in documents]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
```

## LangChain과 AsyncIO 통합

### 비동기 LLM 호출

```python
import asyncio
from langchain.llms import OpenAI
from langchain.schema import Document
from typing import List

class AsyncLLMProcessor:
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.llm = OpenAI(model_name=model_name)
    
    async def process_document(self, document: Document) -> str:
        """단일 문서를 비동기적으로 처리"""
        try:
            # LangChain의 비동기 메서드 사용
            result = await self.llm.ainvoke(
                f"요약해주세요: {document.page_content}"
            )
            return result
        except Exception as e:
            return f"처리 실패: {str(e)}"
    
    async def process_documents_parallel(self, documents: List[Document]) -> List[str]:
        """여러 문서를 병렬로 처리"""
        tasks = [self.process_document(doc) for doc in documents]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
```

### 비동기 체인 실행

```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class AsyncChainProcessor:
    def __init__(self):
        self.summarize_chain = LLMChain(
            llm=OpenAI(),
            prompt=PromptTemplate(
                input_variables=["text"],
                template="다음 텍스트를 요약해주세요: {text}"
            )
        )
        
        self.translate_chain = LLMChain(
            llm=OpenAI(),
            prompt=PromptTemplate(
                input_variables=["text"],
                template="다음 텍스트를 영어로 번역해주세요: {text}"
            )
        )
    
    async def process_text_pipeline(self, text: str) -> dict:
        """텍스트를 요약과 번역으로 병렬 처리"""
        summarize_task = self.summarize_chain.ainvoke({"text": text})
        translate_task = self.translate_chain.ainvoke({"text": text})
        
        # 두 작업을 병렬로 실행
        summary, translation = await asyncio.gather(
            summarize_task,
            translate_task
        )
        
        return {
            "summary": summary["text"],
            "translation": translation["text"]
        }
```

## LangGraph 병렬 처리

LangGraph는 워크플로우 기반의 병렬 처리를 제공하며, `Send()` 함수를 사용한 동적 노드 생성과 조건부 에지를 통한 복잡한 병렬 처리 패턴을 구현할 수 있습니다.

### Send() 함수를 사용한 동적 병렬 처리

```python
import asyncio
from typing import List, TypedDict, Annotated
import operator
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langgraph.constants import Send
from langgraph.graph import StateGraph, END, START

class OverallState(TypedDict):
    """전체 상태 관리"""
    contents: List[str]
    summaries: Annotated[List[str], operator.add]
    final_result: str

class IndividualTask(TypedDict):
    """개별 작업 상태"""
    content: str

class LangGraphParallelProcessor:
    """LangGraph를 사용한 병렬 처리 시스템"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.app = None
    
    async def _process_individual_task(self, state: IndividualTask) -> dict:
        """개별 작업을 처리하는 노드"""
        try:
            print(f"🔄 개별 작업 처리 중: {state['content'][:50]}...")
            
            prompt = f"다음 내용을 요약해주세요:\n\n{state['content']}"
            response = await self.llm.ainvoke(prompt)
            
            return {"summaries": [response.content]}
        except Exception as e:
            print(f"❌ 개별 작업 처리 실패: {e}")
            return {"summaries": [f"처리 실패: {str(e)}"]}
    
    def _map_tasks(self, state: OverallState) -> List[Send]:
        """작업들을 동적으로 매핑하여 병렬 처리"""
        print(f"🚀 {len(state['contents'])}개 작업을 병렬로 매핑")
        
        return [
            Send("process_individual_task", {"content": content})
            for content in state["contents"]
        ]
    
    def _collect_results(self, state: OverallState) -> dict:
        """병렬 처리된 결과들을 수집"""
        print(f"📦 {len(state['summaries'])}개 결과 수집 완료")
        return {"collected": True}
    
    async def _generate_final_result(self, state: OverallState) -> dict:
        """최종 결과 생성"""
        try:
            combined_summaries = "\n\n".join(state["summaries"])
            prompt = f"""
            다음 요약들을 하나의 통합된 최종 요약으로 정리해주세요:
            
            {combined_summaries}
            """
            
            response = await self.llm.ainvoke(prompt)
            return {"final_result": response.content}
        except Exception as e:
            return {"final_result": f"최종 결과 생성 실패: {str(e)}"}
    
    def build_graph(self) -> StateGraph:
        """LangGraph 워크플로우 구축"""
        graph = StateGraph(OverallState)
        
        # 노드 추가
        graph.add_node("process_individual_task", self._process_individual_task)
        graph.add_node("collect_results", self._collect_results)
        graph.add_node("generate_final_result", self._generate_final_result)
        
        # 에지 추가
        graph.add_conditional_edges(
            START,
            self._map_tasks,  # 동적으로 병렬 작업 생성
            ["process_individual_task"]
        )
        graph.add_edge("process_individual_task", "collect_results")
        graph.add_edge("collect_results", "generate_final_result")
        graph.add_edge("generate_final_result", END)
        
        return graph
    
    async def process_parallel(self, contents: List[str]) -> str:
        """병렬 처리 실행"""
        if not self.app:
            graph = self.build_graph()
            self.app = graph.compile()
        
        final_result = ""
        async for step in self.app.astream(
            {"contents": contents},
            {"recursion_limit": 20}
        ):
            step_names = list(step.keys())
            print(f"⚡ 실행 단계: {step_names}")
            
            if "generate_final_result" in step:
                if "final_result" in step["generate_final_result"]:
                    final_result = step["generate_final_result"]["final_result"]
        
        return final_result
```

## 성능 최적화 팁

### 1. 메모리 사용량 모니터링

```python
import asyncio
import psutil
import gc
from typing import List

class MemoryOptimizedProcessor:
    def __init__(self, memory_threshold: float = 0.8):
        self.memory_threshold = memory_threshold
    
    def check_memory_usage(self) -> float:
        """현재 메모리 사용률 확인"""
        return psutil.virtual_memory().percent / 100
    
    async def process_with_memory_check(self, documents: List[Document]) -> List[str]:
        """메모리 사용률을 확인하면서 처리"""
        results = []
        
        for i, doc in enumerate(documents):
            # 메모리 사용률 확인
            if self.check_memory_usage() > self.memory_threshold:
                print(f"메모리 사용률 높음 ({self.check_memory_usage():.1%}), 가비지 컬렉션 실행")
                gc.collect()
                await asyncio.sleep(0.1)  # 잠시 대기
            
            # 문서 처리
            result = await self.process_document(doc)
            results.append(result)
            
            if i % 10 == 0:
                print(f"진행률: {i}/{len(documents)} ({i/len(documents):.1%})")
        
        return results
```

### 2. 연결 풀 최적화

```python
import asyncio
import aiohttp
from typing import Optional

class OptimizedAsyncProcessor:
    def __init__(self, max_connections: int = 100, max_connections_per_host: int = 10):
        self.connector = aiohttp.TCPConnector(
            limit=max_connections,
            limit_per_host=max_connections_per_host,
            ttl_dns_cache=300,  # DNS 캐시 TTL
            use_dns_cache=True,
        )
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            connector=self.connector,
            timeout=aiohttp.ClientTimeout(total=30, connect=10)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
        await self.connector.close()
```

### 3. 프로그레스 바 통합

```python
import asyncio
from tqdm.asyncio import tqdm
from typing import List

async def process_with_progress(documents: List[Document]) -> List[str]:
    """프로그레스 바와 함께 문서 처리"""
    processor = AsyncLLMProcessor()
    
    # tqdm을 사용한 비동기 프로그레스 바
    tasks = [processor.process_document(doc) for doc in documents]
    results = []
    
    for coro in tqdm.as_completed(tasks, desc="문서 처리 중"):
        result = await coro
        results.append(result)
    
    return results
```

## 결론

AsyncIO, LangChain, 그리고 LangGraph를 함께 사용하면 다음과 같은 이점을 얻을 수 있습니다:

### 🚀 성능 및 효율성
1. **병렬 처리**: 여러 작업을 동시에 실행하여 처리 시간 단축
2. **리소스 최적화**: 메모리와 네트워크 리소스 효율적 사용
3. **확장성**: 대용량 데이터 처리 가능
4. **비동기 처리**: I/O 바운드 작업에서 높은 성능 발휘

### 🛠️ 기술별 장단점

#### AsyncIO + LangChain
- **장점**: 간단한 구현, 직관적인 코드, 빠른 개발
- **단점**: 복잡한 워크플로우 관리 어려움, 상태 관리 복잡성
- **적용 시나리오**: 간단한 병렬 처리, API 호출 최적화

#### LangGraph + Send()
- **장점**: 동적 워크플로우, 복잡한 상태 관리, 조건부 처리
- **단점**: 학습 곡선 존재, 초기 설정 복잡성
- **적용 시나리오**: Map-Reduce 패턴, 조건부 워크플로우, 복잡한 문서 처리

### 📊 선택 기준

| 사용 사례 | 권장 방식 | 이유 |
|-----------|-----------|------|
| 간단한 병렬 처리 | AsyncIO + LangChain | 구현 단순성, 빠른 개발 |
| 대용량 문서 요약 | LangGraph Map-Reduce | 토큰 제한 관리, 단계별 처리 |
| 조건부 워크플로우 | LangGraph 조건부 에지 | 복잡한 로직 처리, 상태 관리 |
| API 호출 최적화 | AsyncIO Semaphore | 레이트 리밋 관리, 동시성 제어 |
| 실시간 처리 | AsyncIO as_completed | 즉시 결과 처리, 사용자 피드백 |

### 🔧 실제 구현 가이드

1. **프로젝트 초기 단계**: AsyncIO + LangChain 또는 LangGraph의 Send() 이용
2. **성능 최적화**: 메모리 모니터링, 연결 풀 관리
3. **프로덕션 배포**: 에러 처리, 로깅, 모니터링 강화
