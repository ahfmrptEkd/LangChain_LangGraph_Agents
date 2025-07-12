from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, model_validator
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.utils.function_calling import tool_example_to_messages

# Load environment variables
load_dotenv()

## Define the schema for the data extraction
class Person(BaseModel):
    """Information about a person extracted from text."""
    
    # Each field is optional to allow the model to decline extraction if unsure
    name: Optional[str] = Field(
        default=None, 
        description="The full name of the person"
    )
    age: Optional[int] = Field(
        default=None,
        description="The age of the person in years"
    )
    occupation: Optional[str] = Field(
        default=None,
        description="The job or profession of the person"
    )
    location: Optional[str] = Field(
        default=None,
        description="The city, country, or location where the person lives"
    )

class Company(BaseModel):
    """Information about a company extracted from text."""
    
    name: Optional[str] = Field(
        default=None,
        description="The name of the company"
    )
    industry: Optional[str] = Field(
        default=None,
        description="The industry or sector the company operates in"
    )
    location: Optional[str] = Field(
        default=None,
        description="The location of the company headquarters"
    )
    founded_year: Optional[int] = Field(
        default=None,
        description="The year the company was founded"
    )

class ExtractedData(BaseModel):
    """Container for all extracted structured data."""
    
    people: List[Person] = Field(
        default_factory=list,
        description="List of people mentioned in the text"
    )
    companies: List[Company] = Field(
        default_factory=list,
        description="List of companies mentioned in the text"
    )
    summary: Optional[str] = Field(
        default=None,
        description="A brief summary of the main topics discussed in the text"
    )
    
    @classmethod
    @model_validator(mode='before')
    def validate_lists(cls, data):
        """Validate and clean the input data.
        
        This validator ensures that None values are converted to empty lists
        for people and companies fields to prevent validation errors.
        """
        if isinstance(data, dict):
            # Convert None to empty list for people and companies
            if data.get('people') is None:
                data['people'] = []
            if data.get('companies') is None:
                data['companies'] = []
        return data

## Define the extraction chain
class ExtractionChain:
    """LangChain-based extraction chain for structured data extraction."""
    
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.1):
        """Initialize the extraction chain.
        
        Args:
            model_name: Name of the LLM model to use
            temperature: Temperature parameter for the model (0.0 for deterministic)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.llm = None
        self.extraction_runnable = None
        
        # Initialize the model and chain
        self._setup_model()
        self._setup_extraction_chain()
        
    def _setup_model(self):
        """Set up the LLM model for extraction.
        
        This method initializes the chat model with structured output support.
        Currently supports OpenAI models but can be extended for other providers.
        """
        # Check if OpenAI API key is available
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found in environment variables")
            
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature
        )

    def _setup_extraction_chain(self):
        """Set up the extraction chain with prompt template.
        
        This method creates the complete extraction pipeline including
        the prompt template with Korean language support and example placeholders.
        """
        prompt_template = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are an expert extraction algorithm that can extract structured information from unstructured text.
                
                Instructions:
                - Extract relevant information from the text accurately
                - If you don't know the value of an attribute, return null
                - Support both Korean and English text input
                - Be precise and don't make up information
                - For Korean names, preserve the original Korean characters
                
                한국어 지원:
                - 한국어 텍스트에서 정보를 정확하게 추출하세요
                - 한국어 이름은 원본 한글을 유지하세요
                - 확실하지 않은 정보는 null로 반환하세요
                """
            ),
            # Placeholder for few-shot examples
            MessagesPlaceholder("examples"),
            (
                "human", 
                "Extract structured information from the following text:\n\n{text}"
            ),
        ])
        
        # Create the extraction runnable with structured output
        self.extraction_runnable = prompt_template | self.llm.with_structured_output(
            schema=ExtractedData,
            method="function_calling",
            include_raw=False,
        )
        
    def _get_reference_examples(self) -> List[Dict[str, Any]]:
        """Get reference examples for few-shot learning.
        
        Returns:
            List of example dictionaries with input text and expected output
            
        This method provides reference examples to improve extraction accuracy.
        Examples include both Korean and English text with various entity types.
        """
        examples = [
            # Example 1: No entities found
            (
                "오늘 날씨가 정말 좋네요. 하늘이 파랗고 구름이 하얗습니다.",
                ExtractedData(
                    people=[],
                    companies=[],
                    summary="Weather description - clear sky with white clouds"
                )
            ),
            # Example 2: Person extraction
            (
                "김민수는 35세의 소프트웨어 엔지니어로 서울에서 일하고 있습니다.",
                ExtractedData(
                    people=[Person(
                        name="김민수",
                        age=35,
                        occupation="소프트웨어 엔지니어",
                        location="서울"
                    )],
                    companies=[],
                    summary="Information about Kim Min-su, a software engineer from Seoul"
                )
            ),
            # Example 3: Company extraction
            (
                "삼성전자는 1969년에 설립된 대한민국의 대표적인 IT 기업입니다.",
                ExtractedData(
                    people=[],
                    companies=[Company(
                        name="삼성전자",
                        industry="IT",
                        location="대한민국",
                        founded_year=1969
                    )],
                    summary="Information about Samsung Electronics, a major Korean IT company"
                )
            ),
            # Example 4: Mixed content
            (
                "Steve Jobs founded Apple Inc. in 1976. He was a visionary leader in the technology industry.",
                ExtractedData(
                    people=[Person(
                        name="Steve Jobs",
                        age=None,
                        occupation="visionary leader",
                        location=None
                    )],
                    companies=[Company(
                        name="Apple Inc.",
                        industry="technology",
                        location=None,
                        founded_year=1976
                    )],
                    summary="Information about Steve Jobs and Apple Inc. founding"
                )
            )
        ]
        
        return examples
        
    def _convert_examples_to_messages(self, examples: List[tuple]) -> List:
        """Convert examples to message format for the model.
        
        Args:
            examples: List of (input_text, expected_output) tuples
            
        Returns:
            List of messages in the format expected by the model
        """
        messages = []
        
        for text, expected_output in examples:
            # Convert each example to the message format using LangChain utility
            # Based on LangChain documentation: tool_example_to_messages(input_text, tool_calls, ai_response=None)
            if expected_output.people or expected_output.companies:
                ai_response = "Detected entities."
            else:
                ai_response = "No entities detected."
                
            example_messages = tool_example_to_messages(
                text, 
                [expected_output], 
                ai_response=ai_response
            )
            messages.extend(example_messages)
            
        return messages
        
    def extract(self, text: str, use_examples: bool = True) -> ExtractedData:
        """Extract structured data from unstructured text.
        
        Args:
            text: Input text to extract information from
            use_examples: Whether to use few-shot examples for better performance
            
        Returns:
            ExtractedData object containing extracted information
            
        This method performs the actual extraction using the configured model
        and prompt template. It supports both Korean and English text input.
        """
        if not self.extraction_runnable:
            raise ValueError("Extraction chain not properly initialized")
            
        # Prepare examples if requested
        examples = []
        if use_examples:
            reference_examples = self._get_reference_examples()
            examples = self._convert_examples_to_messages(reference_examples)
            
        # Run extraction
        try:
            result = self.extraction_runnable.invoke({
                "text": text,
                "examples": examples
            })
            return result
            
        except Exception as e:
            raise Exception(f"Extraction failed: {str(e)}")
            
    def batch_extract(self, texts: List[str], use_examples: bool = True) -> List[ExtractedData]:
        """Extract structured data from multiple texts.
        
        Args:
            texts: List of input texts to extract information from
            use_examples: Whether to use few-shot examples for better performance
            
        Returns:
            List of ExtractedData objects containing extracted information
        """
        results = []
        for text in texts:
            result = self.extract(text, use_examples=use_examples)
            results.append(result)
        return results

def main():
    """Main function to demonstrate the extraction functionality.
    
    This function demonstrates how to use the ExtractionChain class
    with various Korean and English text examples. It shows both
    single extraction and batch extraction capabilities.
    """
    
    print("=== LangChain 데이터 추출 예제 ===")
    print("=== LangChain Data Extraction Example ===\n")
    
    try:
        # Initialize extraction chain
        print("📋 추출 체인 초기화 중...")
        extractor = ExtractionChain()
        print("✅ 추출 체인 초기화 완료!\n")
        
        # Test examples in Korean and English
        test_texts = [
            # Korean examples
            "박지성은 42세의 전 축구선수로 현재 서울에서 축구 해설가로 활동하고 있습니다. 그는 맨체스터 유나이티드에서 활약했던 한국의 대표적인 축구선수입니다.",
            
            "네이버는 1999년에 설립된 대한민국의 대표적인 인터넷 기업으로 검색엔진과 온라인 서비스를 제공합니다. 본사는 경기도 분당에 위치하고 있습니다.",
            
            "이재용 삼성전자 부회장은 기술 혁신을 통해 회사를 이끌고 있습니다. 삼성전자는 반도체 산업에서 세계적인 기업으로 성장했습니다.",
            
            # English examples
            "Elon Musk, the 52-year-old entrepreneur, is the CEO of Tesla and SpaceX. He's known for his innovative approach to electric vehicles and space exploration.",
            
            "Microsoft was founded in 1975 by Bill Gates and Paul Allen. The company is headquartered in Redmond, Washington, and is a leader in software and cloud computing.",
            
            # Mixed content
            "Apple의 CEO인 Tim Cook은 2011년부터 회사를 이끌고 있습니다. Apple Inc.는 1976년에 설립되었으며 혁신적인 기술 제품으로 유명합니다.",
            
            # No entities
            "오늘은 날씨가 정말 좋습니다. 파란 하늘과 따뜻한 햇살이 기분을 좋게 만듭니다."
        ]
        
        print("🔍 개별 텍스트 추출 테스트:")
        print("=" * 50)
        
        for i, text in enumerate(test_texts, 1):
            print(f"\n📄 테스트 {i}: {text[:50]}...")
            
            # Extract with examples
            result = extractor.extract(text, use_examples=True)
            
            # Display results
            print("📊 추출 결과:")
            if result.people:
                print(f"  👤 인물: {len(result.people)}명")
                for person in result.people:
                    print(f"    - {person.name} ({person.age}세, {person.occupation}, {person.location})")
            
            if result.companies:
                print(f"  🏢 회사: {len(result.companies)}개")
                for company in result.companies:
                    print(f"    - {company.name} ({company.industry}, {company.location}, {company.founded_year})")
            
            if result.summary:
                print(f"  📝 요약: {result.summary}")
            
            if not result.people and not result.companies:
                print("  ❌ 추출된 엔티티가 없습니다.")
            
        print(f"\n{'='*50}")
        print("🚀 배치 추출 테스트:")
        
        # Test batch extraction
        batch_results = extractor.batch_extract(test_texts[:3], use_examples=True)
        
        print(f"📊 배치 처리 결과: {len(batch_results)}개 텍스트 처리 완료")
        
        total_people = sum(len(result.people) for result in batch_results)
        total_companies = sum(len(result.companies) for result in batch_results)
        
        print(f"  👥 총 인물: {total_people}명")
        print(f"  🏢 총 회사: {total_companies}개")
        
        print(f"\n{'='*50}")
        print("✅ 모든 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        print("\n💡 해결 방법:")
        print("1. OpenAI API 키가 설정되어 있는지 확인하세요")
        print("2. 필요한 패키지가 설치되어 있는지 확인하세요:")
        print("   pip install langchain-openai python-dotenv")
        print("3. .env 파일에 OPENAI_API_KEY를 설정하세요")

def test_extraction_without_examples():
    """Test extraction performance without reference examples.
    
    This function demonstrates the difference in extraction quality
    when using the model without few-shot examples vs. with examples.
    """
    
    print("🔬 참조 예시 없이 추출 테스트:")
    print("=" * 40)
    
    try:
        extractor = ExtractionChain()
        
        test_text = "김철수는 30세의 개발자로 부산에서 일하고 있습니다."
        
        # Without examples
        result_no_examples = extractor.extract(test_text, use_examples=False)
        print("❌ 예시 없음:")
        print(f"   인물: {len(result_no_examples.people)}명")
        
        # With examples
        result_with_examples = extractor.extract(test_text, use_examples=True)
        print("✅ 예시 포함:")
        print(f"   인물: {len(result_with_examples.people)}명")
        
    except Exception as e:
        print(f"테스트 실패: {str(e)}")

if __name__ == "__main__":
    """Entry point for the extraction demo.
    
    This script demonstrates the complete extraction functionality
    including both Korean and English text processing.
    """
    
    # Run main demonstration
    main()
    
    # Optional: Test without examples
    print("\n" + "="*60)
    test_extraction_without_examples()