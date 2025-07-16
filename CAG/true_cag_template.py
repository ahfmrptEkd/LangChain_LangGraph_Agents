"""
True Cache-Augmented Generation (CAG) Implementation

This module implements the True CAG approach where all knowledge is preloaded
into the model's context window, eliminating the retrieval step entirely.

Based on the research paper:
"Don't Do RAG: When Cache-Augmented Generation is All You Need for Knowledge Tasks"
"""

from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
import tiktoken
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentPreloader:
    """
    Utility class for preloading documents into the model's context window
    
    This class handles the preprocessing and formatting of documents
    for preloading into the model's context window.
    """
    
    def __init__(self, max_tokens: int = 16000, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize the document preloader.
        
        Args:
            max_tokens: Maximum number of tokens to use for knowledge preloading
            model_name: Name of the model (for token counting)
        """
        self.max_tokens = max_tokens
        self.model_name = model_name
        self.encoding = tiktoken.encoding_for_model(model_name)
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        return len(self.encoding.encode(text))
    
    def preload_documents(self, documents: List[Document]) -> str:
        """
        Merge all documents into a single context for preloading
        
        This method combines all documents into a single context string
        that will be preloaded into the model's context window.
        
        Args:
            documents: List of documents to preload
            
        Returns:
            Formatted context string containing all documents
        """
        logger.info(f"🔄 Preloading {len(documents)} documents into context...")
        
        # 문서들을 컨텍스트 형식으로 변환
        context_sections = []
        total_tokens = 0
        
        for i, doc in enumerate(documents):
            # 문서를 구조화된 형식으로 포맷팅
            section = f"""
                    --- Document {i+1} ---
                    Title: {doc.metadata.get('title', f'Document {i+1}')}
                    Source: {doc.metadata.get('source', 'Unknown')}
                    Content: {doc.page_content}
                    ---
                    """
            
            section_tokens = self.count_tokens(section)
            
            # 토큰 제한 확인
            if total_tokens + section_tokens > self.max_tokens:
                logger.warning(f"⚠️ Token limit reached. Truncating at document {i}")
                break
                
            context_sections.append(section)
            total_tokens += section_tokens
        
        # 모든 문서를 하나의 컨텍스트로 결합
        preloaded_context = "\n".join(context_sections)
        
        logger.info(f"✅ Successfully preloaded {len(context_sections)} documents")
        logger.info(f"📊 Total tokens used: {total_tokens}/{self.max_tokens}")
        
        return preloaded_context


class basicCagGenerator:
    """
    Basic CAG implementation class
    
    This class implements the basic CAG approach where all knowledge is
    preloaded into the model's context window, eliminating retrieval entirely.
    
    Key features:
    - 40x faster than traditional RAG (no retrieval step)
    - All knowledge preloaded in model context
    - Direct generation from preloaded knowledge
    """
    
    def __init__(
        self,
        llm: BaseChatModel,
        documents: List[Document],
        max_context_tokens: int = 16000
    ):
        """
        Initialize the basic CAG generator.
        
        Args:
            llm: The language model to use for generation
            documents: List of documents to preload
            max_context_tokens: Maximum tokens for context preloading
        """
        self.llm = llm
        self.documents = documents
        self.max_context_tokens = max_context_tokens
        
        # 문서 사전 로더 초기화
        self.preloader = DocumentPreloader(
            max_tokens=max_context_tokens,
            model_name=llm.model_name if hasattr(llm, 'model_name') else "gpt-3.5-turbo"
        )
        
        # 지식 사전 로드 (핵심!)
        self.preloaded_knowledge = self._preload_knowledge()
        
        # 프롬프트 템플릿 설정
        self.prompt_template = self._create_prompt_template()
        
        logger.info("🚀 basic CAG Generator initialized with preloaded knowledge")
    
    def _preload_knowledge(self) -> str:
        """
        Preloads all knowledge into the model's context.
        """
        logger.info("🔄 Preloading all knowledge into model context...")
        
        # 모든 문서를 하나의 컨텍스트로 변환
        preloaded_context = self.preloader.preload_documents(self.documents)
        
        logger.info("✅ Knowledge preloading completed!")
        return preloaded_context
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create the prompt template for basic CAG generation."""
        return ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant with access to preloaded knowledge.
            
            IMPORTANT: All necessary knowledge has been preloaded in your context. 
            DO NOT mention that you need to search for information - everything you need is already available.

            Answer questions based ONLY on the preloaded knowledge provided below.
            If the information is not in the preloaded knowledge, say so clearly.

            === PRELOADED KNOWLEDGE ===
            {preloaded_knowledge}
            =========================="""),
                        ("human", "{question}")
                    ])
    
    def generate(self, question: str) -> Dict[str, Any]:
        """
        Generates an answer to the given question using preloaded knowledge.
        
        This method generates answers directly from preloaded knowledge,
        eliminating the retrieval step entirely. It does not perform any
        external document retrieval.
        
        Args:
            question: The question to answer
            
        Returns:
            Dictionary containing the answer and metadata
        """
        logger.info(f"❓ Generating answer for: '{question}'")
        logger.info("🚀 Using preloaded knowledge (NO RETRIEVAL)")
        
        # 사전 로드된 지식을 활용하여 직접 답변 생성
        chain = self.prompt_template | self.llm
        
        # 검색 단계 없이 바로 생성!
        response = chain.invoke({
            "preloaded_knowledge": self.preloaded_knowledge,
            "question": question
        })
        
        result = {
            "answer": response.content,
            "method": "basic CAG (Preloaded Context)",
            "retrieval_time": 0.0,  # 검색 시간 없음!
            "total_documents": len(self.documents),
            "context_tokens": self.preloader.count_tokens(self.preloaded_knowledge)
        }
        
        logger.info("✅ Answer generated from preloaded knowledge")
        return result
    
    def get_knowledge_info(self) -> Dict[str, Any]:
        """
        Returns information about the preloaded knowledge.
        """
        return {
            "total_documents": len(self.documents),
            "context_tokens": self.preloader.count_tokens(self.preloaded_knowledge),
            "max_context_tokens": self.max_context_tokens,
            "token_utilization": f"{self.preloader.count_tokens(self.preloaded_knowledge)}/{self.max_context_tokens}",
            "preloaded": True
        }
    
    def update_knowledge(self, new_documents: List[Document]) -> None:
        """
        Updates the preloaded knowledge with new documents.
        """
        logger.info(f"🔄 Updating preloaded knowledge with {len(new_documents)} new documents")
        
        self.documents = new_documents
        self.preloaded_knowledge = self._preload_knowledge()
        
        logger.info("✅ Knowledge updated successfully") 