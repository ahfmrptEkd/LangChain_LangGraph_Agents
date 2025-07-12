from dotenv import load_dotenv
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import os

# Load environment variables
load_dotenv()

# LLM 설정
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

# 분류용 프롬프트 템플릿
classification_prompt = ChatPromptTemplate.from_template(
    """
    다음 텍스트에서 요청된 정보를 추출하세요.
    'Classification' 스키마에 정의된 속성만 추출하세요.
    
    한국어 텍스트를 정확하게 분석하여 분류해주세요.

    텍스트:
    {input}
    """
)

# 방법 1: enum 없이 (자유로운 문자열)
class BasicClassification(BaseModel):
    """기본 분류 - 제한 없는 문자열"""
    sentiment: str = Field(description="텍스트의 감정 (긍정적, 부정적, 중립적 등)")
    topic: str = Field(description="텍스트의 주요 주제")
    urgency: int = Field(description="긴급도 (1-10 점수)")

# 방법 2: enum으로 제한된 선택지
class EnumClassification(BaseModel):
    """Enum 분류 - 제한된 선택지"""
    sentiment: str = Field(..., enum=["긍정", "부정", "중립"])
    category: str = Field(
        ..., 
        enum=["리뷰", "질문", "불만", "정보요청", "일상대화"],
        description="텍스트의 카테고리"
    )
    priority: int = Field(
        ...,
        enum=[1, 2, 3, 4, 5],
        description="우선순위 (1=낮음, 5=높음)"
    )

# 방법 3: Literal 타입 (권장 방식)
class LiteralClassification(BaseModel):
    """Literal 분류 - 타입 안전성 보장"""
    sentiment: Literal["긍정", "부정", "중립"] = Field(
        description="텍스트의 감정"
    )
    category: Literal["고객지원", "기술문의", "결제문의", "일반문의", "불만접수"] = Field(
        description="고객 서비스 카테고리"
    )
    language: Literal["한국어", "영어", "중국어", "일본어", "기타"] = Field(
        description="텍스트의 언어"
    )
    confidence: float = Field(
        description="분류 신뢰도 (0.0-1.0)",
        ge=0.0,
        le=1.0
    )

def classify_with_basic(text: str) -> BasicClassification:
    """기본 분류 실행"""
    structured_llm = llm.with_structured_output(BasicClassification)
    prompt = classification_prompt.invoke({"input": text})
    return structured_llm.invoke(prompt)

def classify_with_enum(text: str) -> EnumClassification:
    """Enum 분류 실행"""
    structured_llm = llm.with_structured_output(EnumClassification)
    prompt = classification_prompt.invoke({"input": text})
    return structured_llm.invoke(prompt)

def classify_with_literal(text: str) -> LiteralClassification:
    """Literal 분류 실행 (권장)"""
    structured_llm = llm.with_structured_output(LiteralClassification)
    prompt = classification_prompt.invoke({"input": text})
    return structured_llm.invoke(prompt)

def demo_all_methods():
    """모든 분류 방법 데모"""
    
    print("🔍 LangChain 텍스트 분류 방법 비교")
    print("=" * 60)
    
    # 테스트 텍스트들
    test_texts = [
        "제품이 불량입니다. 즉시 환불해주세요!",
        "이 서비스 사용법을 알고 싶습니다.",
        "정말 만족스러운 구매였어요. 추천합니다!",
        "Hello, I need help with my account",
        "결제가 안 되는데 도와주세요."
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n📄 테스트 {i}: {text}")
        print("-" * 40)
        
        try:
            # 방법 1: 기본 분류
            basic_result = classify_with_basic(text)
            print("🔸 기본 분류:")
            print(f"  감정: {basic_result.sentiment}")
            print(f"  주제: {basic_result.topic}")
            print(f"  긴급도: {basic_result.urgency}/10")
            
            # 방법 2: Enum 분류
            enum_result = classify_with_enum(text)
            print("🔸 Enum 분류:")
            print(f"  감정: {enum_result.sentiment}")
            print(f"  카테고리: {enum_result.category}")
            print(f"  우선순위: {enum_result.priority}/5")
            
            # 방법 3: Literal 분류 (권장)
            literal_result = classify_with_literal(text)
            print("🔸 Literal 분류 (권장):")
            print(f"  감정: {literal_result.sentiment}")
            print(f"  카테고리: {literal_result.category}")
            print(f"  언어: {literal_result.language}")
            print(f"  신뢰도: {literal_result.confidence:.2f}")
            
        except Exception as e:
            print(f"❌ 오류: {e}")


def main():
    """메인 실행 함수"""
    
    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY를 .env 파일에 설정해주세요")
        return
    
    print("🎯 LangChain 분류 시스템")
    
    demo_all_methods()

if __name__ == "__main__":
    main()