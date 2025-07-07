from langchain_community.document_loaders import YoutubeLoader
import datetime
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv


class TutorialSearch(BaseModel):
    """Search over a database of tutorial videos about a software library."""

    content_search: str = Field(
        ...,
        description="Similarity search query applied to video transcripts.",
    )

    title_search: Optional[str] = Field(
        None,
        description=(
            "Alternate version of the content search query to apply to video titles. "
            "Should be succinct and only include key words that could be in a video "
            "title."
        ),
    )
    
    min_view_count: Optional[int] = Field(
        None,
        description="Minimum view count filter, inclusive. Only use if explicitly specified.",
    )
    
    max_view_count: Optional[int] = Field(
        None,
        description="Maximum view count filter, exclusive. Only use if explicitly specified.",
    )
    
    earliest_publish_date: Optional[datetime.date] = Field(
        None,
        description="Earliest publish date filter, inclusive. Only use if explicitly specified.",
    )
    
    latest_publish_date: Optional[datetime.date] = Field(
        None,
        description="Latest publish date filter, exclusive. Only use if explicitly specified.",
    )
    
    min_length_sec: Optional[int] = Field(
        None,
        description="Minimum video length in seconds, inclusive. Only use if explicitly specified.",
    )
    
    max_length_sec: Optional[int] = Field(
        None,
        description="Maximum video length in seconds, exclusive. Only use if explicitly specified.",
    )

    def pretty_print(self) -> None:
        """
        Pretty print the search parameters with visual formatting.
        Only displays non-null and non-default values.
        """
        print("🔍 Search Parameters:")
        print("-" * 40)
        
        for field_name, field_info in self.model_fields.items():
            value = getattr(self, field_name)
            default_value = field_info.default
            
            if value is not None and value != default_value:
                # Add emojis for different field types
                if "search" in field_name:
                    emoji = "🔎"
                elif "count" in field_name:
                    emoji = "👀"
                elif "date" in field_name:
                    emoji = "📅"
                elif "length" in field_name:
                    emoji = "⏱️"
                else:
                    emoji = "📝"
                
                print(f"{emoji} {field_name}: {value}")
        print("-" * 40)

def load_youtube_video_safely(url, max_retries=3):
    """
    Safely load YouTube video with multiple fallback strategies.
    
    Args:
        url: YouTube video URL
        max_retries: Number of retry attempts
        
    Returns:
        dict: Video metadata or None if failed
    """
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            print(f"📺 Attempting to load YouTube video (try {retry_count + 1}/{max_retries})...")
            
            # Try loading with different configurations
            loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
            docs = loader.load()
            
            if docs:
                print("✅ YouTube video loaded successfully!")
                return docs[0].metadata
            else:
                raise Exception("No documents loaded")
                
        except Exception as e:
            retry_count += 1
            print(f"❌ Attempt {retry_count} failed: {str(e)}")
            
            if retry_count < max_retries:
                print("🔄 Retrying with different approach...")
                # Try without video info on retry
                try:
                    loader = YoutubeLoader.from_youtube_url(url, add_video_info=False)
                    docs = loader.load()
                    if docs:
                        print("✅ YouTube video loaded successfully (without video info)!")
                        return {"title": "YouTube Video", "description": "Loaded successfully"}
                except Exception as e2:
                    print(f"❌ Retry also failed: {str(e2)}")
    
    print("🚫 All attempts failed, using mock data for demonstration")
    return None

def main():
    """
    Main execution function.
    Demonstrates query construction for tutorial video database search.
    """
    # Load environment variables
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    
    print("=" * 80)
    print("🎥 TUTORIAL VIDEO QUERY CONSTRUCTION SYSTEM")
    print("=" * 80)
    
    print("\n⚡ Loading Sample Video...")
    print("-" * 50)
    
    # Try multiple YouTube URLs for better success rate
    test_urls = [
        "https://www.youtube.com/watch?v=QsYGlZkevEg",  # Common LangChain tutorial
        "https://www.youtube.com/watch?v=kCc8FmEb1nY",  # Alternative tutorial
        "https://www.youtube.com/watch?v=LbT1yp6quS8",  # Another alternative
    ]
    
    video_metadata = None
    for url in test_urls:
        print(f"🔗 Trying URL: {url}")
        video_metadata = load_youtube_video_safely(url)
        if video_metadata:
            break
    
    if not video_metadata:
        # Use mock data as fallback
        video_metadata = {
            "title": "RAG from Scratch - LangChain Tutorial",
            "view_count": 15420,
            "publish_date": "2023-08-15",
            "length": 1245,  # seconds
            "description": "Learn how to build RAG systems from scratch using LangChain"
        }
        print("🎭 Using mock video data for demonstration")
    
    print("\n📊 Sample Video Metadata:")
    print("-" * 50)
    print("📹 Video Info:")
    for key, value in video_metadata.items():
        print(f"   • {key}: {value}")
    print("-" * 50)
    
    print("\n🤖 Initializing Query Analyzer...")
    print("-" * 50)
    
    # System prompt for query construction
    system = """You are an expert at converting user questions into database queries. \
    You have access to a database of tutorial videos about a software library for building LLM-powered applications. \
    Given a question, return a database query optimized to retrieve the most relevant results.

    If there are acronyms or words you are not familiar with, do not try to rephrase them.
    """

    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "{question}"),
    ])

    # Initialize LLM with structured output
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(TutorialSearch)
    query_analyzer = prompt | structured_llm
    
    print("✅ Query Analyzer Ready!")
    print("-" * 50)
    
    # Test queries with enhanced output
    test_queries = [
        "rag from scratch",
        "videos on chat langchain published in 2023",
        "how to use multi-modal models in an agent, only videos under 5 minutes"
    ]
    
    for i, question in enumerate(test_queries, 1):
        print(f"\n🎯 Test Query #{i}:")
        print("-" * 50)
        print(f"❓ Question: {question}")
        print()
        
        # Execute query analysis
        result = query_analyzer.invoke({"question": question})
        result.pretty_print()
        print()
    
    print("\n" + "=" * 80)
    print("✨ QUERY CONSTRUCTION COMPLETE!")
    print("=" * 80)
    
    print("\n💡 System Features:")
    print("   🔍 Converts natural language → structured database queries")
    print("   📊 Semantic search on content and titles")
    print("   🎯 Filters: view count, date, video length")
    print("   🤖 Optimized for LLM-powered applications")
    print("   🛠️ Robust error handling for YouTube API issues")

if __name__ == "__main__":
    main()