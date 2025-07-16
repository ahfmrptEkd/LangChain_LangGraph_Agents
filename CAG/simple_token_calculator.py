"""
LangChain Document Token Calculator

This is a simple function to calculate the number of tokens in a text or a document.
"""

import tiktoken
from langchain_core.documents import Document
from typing import List


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """
    Calculate the number of tokens in a text.
    
    Args:
        text (str): The text to calculate the number of tokens.
        encoding_name (str): The encoding to use for tokenization (default: "cl100k_base")
        
    Returns:
        int: The number of tokens
    """
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


def count_document_tokens(document: Document, encoding_name: str = "cl100k_base") -> int:
    """
    Calculate the number of tokens in a document.
    
    Args:
        document (Document): The document to calculate the number of tokens.
        encoding_name (str): The encoding to use for tokenization (default: "cl100k_base")
        
    Returns:
        int: The number of tokens
    """
    return count_tokens(document.page_content, encoding_name)


def count_documents_tokens(documents: List[Document], encoding_name: str = "cl100k_base") -> dict:
    """
    Calculate the number of tokens in a list of documents.
    
    Args:
        documents (List[Document]): The list of documents to calculate the number of tokens.
        encoding_name (str): The encoding to use for tokenization (default: "cl100k_base")
        
    Returns:
        dict: The result of token analysis
    """
    total_tokens = 0
    document_tokens = []
    
    for i, doc in enumerate(documents):
        tokens = count_document_tokens(doc, encoding_name)
        total_tokens += tokens
        document_tokens.append({
            "index": i,
            "tokens": tokens,
            "content_preview": doc.page_content[:50] + "..." if len(doc.page_content) > 50 else doc.page_content
        })
    
    return {
        "total_documents": len(documents),
        "total_tokens": total_tokens,
        "average_tokens_per_document": total_tokens / len(documents) if documents else 0,
        "document_details": document_tokens
    }


if __name__ == "__main__":
    # 1. Simple text token calculation
    text = "Hello! I am using LangChain to analyze documents."
    tokens = count_tokens(text)
    print(f"텍스트: '{text}'")
    print(f"토큰 수: {tokens}\n")
    
    # 2. Document token calculation
    doc = Document(page_content="LangChain은 대규모 언어 모델을 활용한 애플리케이션 개발을 위한 프레임워크입니다.")
    doc_tokens = count_document_tokens(doc)
    print(f"Document 토큰 수: {doc_tokens}\n")
    
    # 3. Multiple document token calculation
    documents = [
        Document(page_content="첫 번째 문서입니다."),
        Document(page_content="두 번째 문서입니다."),
        Document(page_content="세 번째 문서입니다.")
    ]
    
    analysis = count_documents_tokens(documents)
    print(f"총 문서 수: {analysis['total_documents']}")
    print(f"총 토큰 수: {analysis['total_tokens']}")
    print(f"평균 토큰 수: {analysis['average_tokens_per_document']:.2f}") 