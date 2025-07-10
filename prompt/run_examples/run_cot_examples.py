"""
Chain of Thought (CoT) 프롬프트 템플릿 사용 예제 (리팩토링 버전)

`templates.py`에 정의된 CoT 프롬프트 생성 함수들을 사용하여
LLM 체인을 구성하고 실행하는 방법을 보여줍니다.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.panel import Panel

from ..templates import create_zero_shot_cot_template, create_few_shot_cot_template

load_dotenv()
console = Console()


def run_zero_shot_cot_example(llm):
    """
    Zero-shot CoT 프롬프트 템플릿 예제를 실행합니다.
    """
    console.print(Panel.fit("🔹 1. Zero-shot CoT 프롬프트 예제", style="blue"))

    # 1. 템플릿 생성
    zero_shot_cot_prompt = create_zero_shot_cot_template()

    # 2. 체인 생성
    chain = zero_shot_cot_prompt | llm

    # 3. 실행 및 결과 출력
    problem = "한 반에 학생이 30명 있습니다. 그 중 60%가 남학생이라면, 여학생은 몇 명일까요?"
    console.print(f"📝 문제: {problem}")

    try:
        response = chain.invoke({"problem": problem})
        console.print(f"🤖 AI 응답 (단계별 추론):\n{response.content}")
    except Exception as e:
        console.print(f"❌ 오류 발생: {e}")

    console.print("-" * 50)


def run_few_shot_cot_example(llm):
    """
    Few-shot CoT 프롬프트 템플릿 예제를 실행합니다.
    """
    console.print(Panel.fit("🔹 2. Few-shot CoT 프롬프트 예제", style="green"))

    # 1. CoT 예시 데이터
    examples = [
        {
            "problem": "사과 5개가 2000원입니다. 사과 8개를 사려면 얼마가 필요할까요?",
            "solution": "단계별 해결:\n1. 사과 1개의 가격 계산: 2000원 / 5개 = 400원\n2. 사과 8개의 가격 계산: 400원 * 8개 = 3200원\n최종 답: 3200원"
        },
        {
            "problem": "정사각형의 한 변이 4cm입니다. 이 정사각형의 넓이는 얼마일까요?",
            "solution": "단계별 해결:\n1. 정사각형 넓이 공식: 한 변 * 한 변\n2. 값 대입: 4cm * 4cm = 16cm²\n최종 답: 16cm²"
        }
    ]

    # 2. 템플릿 생성
    few_shot_cot_prompt = create_few_shot_cot_template(examples)

    # 3. 체인 생성 및 실행
    chain = few_shot_cot_prompt | llm
    problem = "원의 반지름이 5cm일 때, 원의 넓이를 구하세요. (π=3.14)"
    console.print(f"📝 문제: {problem}")

    try:
        response = chain.invoke({"problem": problem})
        console.print(f"🤖 AI 응답 (단계별 추론):\n{response.content}")
    except Exception as e:
        console.print(f"❌ 오류 발생: {e}")

    console.print("-" * 50)


def main():
    """
    메인 실행 함수
    """
    if not os.getenv("OPENAI_API_KEY"):
        console.print("❌ OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        return

    console.print(Panel.fit("🚀 CoT 프롬프트 예제 실행", style="bold blue"))
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)

    run_zero_shot_cot_example(llm)
    run_few_shot_cot_example(llm)

    console.print(Panel.fit("✅ 모든 CoT 예제 실행 완료", style="bold green"))


if __name__ == "__main__":
    main()

