"""
Self-Reflection 프롬프트 템플릿 사용 예제

`templates.py`에 정의된 Self-Reflection 관련 함수들을 사용하여
생성 -> 비평 -> 수정의 3단계 과정을 통해 결과물의 완성도를 높이는 방법을 보여줍니다.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.panel import Panel

from ..templates import create_self_critique_template, create_self_refine_template

load_dotenv()
console = Console()


def run_self_reflection_example():
    """
    Self-Reflection 예제를 실행합니다.
    """
    console.print(Panel.fit("🚀 Self-Reflection 예제 실행", style="bold yellow"))

    # LLM 인스턴스 생성
    llm = ChatOpenAI(model="gpt-4o", temperature=0.5)

    # --- 1단계: 초안 생성 ---
    console.print(Panel("📄 1. 초안 생성", style="blue"))
    draft_prompt = "LangChain의 LCEL(LangChain Expression Language)에 대해 초보자를 위해 설명하는 짧은 글을 작성해줘."
    console.print(f"[bold]요청:[/bold] {draft_prompt}")
    
    try:
        draft = llm.invoke(draft_prompt).content
        console.print(Panel(draft, title="생성된 초안", border_style="green"))
    except Exception as e:
        console.print(f"❌ 초안 생성 오류: {e}")
        return

    # --- 2단계: 스스로 비평 ---
    console.print(Panel("🤔 2. 스스로 비평하기", style="blue"))
    critique_prompt = create_self_critique_template()
    critique_chain = critique_prompt | llm

    try:
        critique = critique_chain.invoke({"draft": draft}).content
        console.print(Panel(critique, title="생성된 비평", border_style="yellow"))
    except Exception as e:
        console.print(f"❌ 비평 생성 오류: {e}")
        return

    # --- 3단계: 비평을 바탕으로 수정 ---
    console.print(Panel("✨ 3. 비평을 바탕으로 수정하기", style="blue"))
    refine_prompt = create_self_refine_template()
    refine_chain = refine_prompt | llm

    try:
        final_result = refine_chain.invoke({"draft": draft, "critique": critique}).content
        console.print(Panel(final_result, title="최종 결과물", border_style="magenta"))
    except Exception as e:
        console.print(f"❌ 최종 결과물 생성 오류: {e}")


def main():
    """
    메인 실행 함수
    """
    if not os.getenv("OPENAI_API_KEY"):
        console.print("❌ OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        return

    run_self_reflection_example()
    console.print(Panel.fit("✅ Self-Reflection 예제 실행 완료", style="bold green"))


if __name__ == "__main__":
    main()
