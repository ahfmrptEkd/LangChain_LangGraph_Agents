"""
Shot 기반 프롬프트 템플릿 사용 예제

`templates.py`에 정의된 정적/동적 Few-shot 프롬프트 생성 함수들을
사용하여 LLM 체인을 구성하고 실행하는 방법을 보여줍니다.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.panel import Panel

from ..templates import create_static_few_shot_template, create_dynamic_few_shot_template

load_dotenv()
console = Console()


def run_static_few_shot_example(llm):
    """
    정적 Few-shot 프롬프트 템플릿 예제를 실행합니다.
    """
    console.print(Panel.fit("🔹 1. 정적 Few-shot 프롬프트 예제", style="green"))

    # 1. Few-shot 예제 데이터
    examples = [
        {"input": "이 영화는 정말 환상적이었어요!", "output": "긍정"},
        {"input": "음식이 너무 차갑고 맛이 없어요.", "output": "부정"},
        {"input": "그냥 평범한 하루였어요.", "output": "중립"},
    ]

    # 2. 템플릿 생성
    static_prompt = create_static_few_shot_template(
        examples=examples,
        system_prefix="다음은 텍스트의 감정을 분석하는 예시입니다. '긍정', '부정', '중립' 중 하나로 답해주세요.",
        human_suffix="텍스트: {text}"
    )

    # 3. 체인 생성 및 실행
    chain = static_prompt | llm
    text = "배송이 빠르고 제품 품질도 만족스러워요."
    console.print(f"📝 분석할 텍스트: {text}")

    try:
        response = chain.invoke({"text": text})
        console.print(f"🤖 감정 분석 결과: {response.content}")
    except Exception as e:
        console.print(f"❌ 오류 발생: {e}")

    console.print("-" * 50)


def run_dynamic_few_shot_example(llm):
    """
    동적(의미 기반) Few-shot 프롬프트 템플릿 예제를 실행합니다.
    """
    console.print(Panel.fit("🔹 2. 동적 Few-shot 프롬프트 예제", style="magenta"))

    # 1. 전체 예제 풀
    examples = [
        {"question": "파이썬에서 리스트를 정렬하는 법?", "answer": "`sorted()` 함수나 `list.sort()` 메서드를 사용합니다."},
        {"question": "SQL에서 중복을 제거하는 법?", "answer": "`SELECT DISTINCT` 구문을 사용합니다."},
        {"question": "React에서 상태를 어떻게 관리하나요?", "answer": "`useState` 또는 `useReducer` 훅을 사용합니다."},
        {"question": "파이썬 딕셔너리에서 키를 확인하는 법?", "answer": "`'key' in my_dict` 구문을 사용합니다."},
    ]

    # 2. 템플릿 생성
    try:
        dynamic_prompt = create_dynamic_few_shot_template(
            examples=examples,
            system_prefix="다음은 프로그래밍 질문에 대한 답변 예시입니다. 유사한 질문에 대해 답변해주세요.",
            human_suffix="{question}",
            k=2
        )
    except Exception as e:
        console.print(f"❌ 동적 프롬프트 생성 오류: {e}")
        console.print("💡 FAISS 등 필요한 패키지가 설치되었는지 확인하세요.")
        return

    # 3. 체인 생성 및 실행
    chain = dynamic_prompt | llm
    question = "파이썬에서 딕셔너리의 값을 가져오는 방법은?"
    console.print(f"📝 질문: {question}")

    try:
        response = chain.invoke({"question": question})
        console.print(f"🤖 AI 응답:\n{response.content}")
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

    console.print(Panel.fit("🚀 Shot 기반 프롬프트 예제 실행", style="bold green"))
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)

    run_static_few_shot_example(llm)
    run_dynamic_few_shot_example(llm)

    console.print(Panel.fit("✅ 모든 Shot 기반 예제 실행 완료", style="bold green"))


if __name__ == "__main__":
    main()
