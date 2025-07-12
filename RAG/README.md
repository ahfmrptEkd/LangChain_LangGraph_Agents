![rag overall](../src/imgs/rag.png)
(https://github.com/langchain-ai/rag-from-scratch)

## 🎯 About This Directory

이 디렉토리는 **고급 RAG(Retrieval-Augmented Generation) 기법**들의 Python 구현 예제를 포함하고 있습니다. 표준 RAG 파이프라인의 각 단계를 개선하고, 더 정확하고 신뢰성 높은 답변을 생성하기 위한 다양한 최신 방법론들을 체계적으로 정리하는 것을 목표로 한다.

각 하위 폴더는 RAG의 특정 단계(예: `Indexing`, `Query Translation`, `Retrieval`)에 해당하며, 해당 단계에서 사용할 수 있는 여러 기법들의 코드와 상세한 설명을 담은 `README.md` 파일을 포함되어있다.

---

## 📁 Directory Structure

```
RAG/
├── README.md                   # 현재 문서
|
├── indexing/                   # 문서를 구조화하고 표현하는 기법
│   ├── README.md
│   ├── multi_representation.py
│   └── raptor.py
|
├── query_construction/         # 자연어 질문을 구조화된 쿼리로 변환하는 기법
│   ├── README.md
│   └── consturction.py
|
├── query_translation/          # 사용자의 질문을 다각도로 변환하는 기법
│   ├── README.md
│   └── ... (5 files)
|
├── retrieval/                  # 검색 결과를 평가하고 개선하는 기법
│   ├── README.md
│   └── ... (5 files)
|
└── routing/                    # 질문에 맞는 최적의 경로를 선택하는 기법
    ├── README.md
    └── ... (2 files)
```

---

## 📚 Sources and Further Reading

**source from https://github.com/langchain-ai/rag-from-scratch & https://github.com/langchain-ai/langgraph/tree/main/examples/rag**

# query-translation
- 1. RAG fusion : https://github.com/langchain-ai/langchain/blob/master/cookbook/rag_fusion.ipynb?ref=blog.langchain.dev
- 2. RAG fusion : https://towardsdatascience.com/how-to-make-your-llm-more-accurate-with-rag-fine-tuning/
- 3. decomposition - answer recursivly (IQ-CoT)_1 : https://arxiv.org/pdf/2205.10625
- 4. decomposition - answer recursivly (IQ-CoT)_22 : https://arxiv.org/pdf/2212.10509
- 5. step-back : https://arxiv.org/pdf/2310.06117
- 6. HyDE : https://github.com/langchain-ai/langchain/blob/master/cookbook/hypothetical_document_embeddings.ipynb
- 7. HyDE : https://arxiv.org/abs/2212.10496
<br>
<br>

# query-construction 
- 1. Query Construction (blog) : https://blog.langchain.com/query-construction/
- 2. LangChain another blog : https://blog.langchain.com/enhancing-rag-based-applications-accuracy-by-constructing-and-leveraging-knowledge-graphs/
- 3. self-query tutorial(official LC) : https://python.langchain.com/docs/how_to/self_query/
<br>
<br>

# Routing
- 1. logical & sementic route : https://python.langchain.com/docs/how_to/routing/
<br>
<br>

# Indexing
- 1. chunking : https://www.youtube.com/watch?v=8OJC21T2SL4&ab_channel=GregKamradt
- 2. multi representation LC blog (1) : https://blog.langchain.com/semi-structured-multi-modal-rag/
- 3. multi representation LC document (2) : https://python.langchain.com/docs/how_to/multi_vector/
- 4. multi representation scholar paper (3) : https://arxiv.org/abs/2312.06648
- 5. RAPTOR scholar paper : https://arxiv.org/pdf/2401.18059
- 6. ColBERT : https://arxiv.org/abs/2004.12832
- 7. ColBERT repo : https://arxiv.org/abs/2004.12832
<br>
<br>

# retrieval
- 1. re-rank (cohere api) : https://python.langchain.com/docs/integrations/retrievers/cohere-reranker/
- 2. CRAG LC Blog (Corrected RAG) w/ LangGraph : https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_crag/
- 3. CRAG : https://arxiv.org/pdf/2401.15884
- 4. self-rag : https://arxiv.org/pdf/2310.11511
- 5. Adaptive-rag : https://arxiv.org/pdf/2403.14403
- 6. Agentic rag : https://arxiv.org/pdf/2501.09136
