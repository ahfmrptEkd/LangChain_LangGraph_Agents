"""
기본 프롬프트 템플릿 사용 예제 (리팩토링 버전)

`templates.py`에 정의된 기본 프롬프트 생성 함수들을 사용하여
LLM 체인을 구성하고 실행하는 방법을 보여줍니다.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.panel import Panel

from ..templates import create_basic_prompt_template, create_blog_post_template


load_dotenv()
console = Console()


def run_simple_template_example(llm):
    """
    간단한 기본 프롬프트 템플릿 예제를 실행합니다.
    """
    console.print(Panel.fit("🔹 1. 간단한 기본 프롬프트 예제", style="blue"))

    # 1. 템플릿 생성
    simple_prompt = create_basic_prompt_template(
        system_message="당신은 친절한 AI 비서입니다.",
        human_message="{topic}에 대해 {length}로 설명해주세요."
    )

    # 2. 체인 생성
    chain = simple_prompt | llm

    # 3. 실행 및 결과 출력
    topic = "인공지능의 역사"
    length = "5문장"
    console.print(f"📝 질문: {topic}에 대해 {length}로 설명 요청")
    
    try:
        response = chain.invoke({"topic": topic, "length": length})
        console.print(f"🤖 AI 응답:\n{response.content}")
    except Exception as e:
        console.print(f"❌ 오류 발생: {e}")
    
    console.print("-" * 50)


def run_blog_post_template_example(llm):
    """
    블로그 포스트용 프롬프트 템플릿 예제를 실행합니다.
    """
    console.print(Panel.fit("🔹 2. 블로그 포스트 작성 프롬프트 예제", style="yellow"))

    # 1. 템플릿 생성
    blog_prompt = create_blog_post_template()

    # 2. 체인 생성
    chain = blog_prompt | llm

    # 3. 실행 및 결과 출력
    post_info = {
        "role": "기술 블로그 작가",
        "topic": "LangChain의 LCEL(LangChain Expression Language)의 장점",
        "audience": "초보 개발자",
        "tone": "친근하고 유익한",
        "length": "약 300자"
    }
    console.print(f"📝 요청: 블로그 포스트 작성\n{post_info}")

    try:
        response = chain.invoke(post_info)
        console.print(f"🤖 AI 응답:\n{response.content}")
    except Exception as e:
        console.print(f"❌ 오류 발생: {e}")

    console.print("-" * 50)


def main():
    """
    메인 실행 함수
    """
    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        console.print("❌ OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        return

    console.print(Panel.fit("🚀 기본 프롬프트 템플릿 예제 실행", style="bold blue"))
    
    # LLM 인스턴스 생성
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    # 예제 실행
    run_simple_template_example(llm)
    run_blog_post_template_example(llm)

    console.print(Panel.fit("✅ 모든 기본 예제 실행 완료", style="bold green"))


if __name__ == "__main__":
    main()
