"""
Self-Consistency 프롬프트 기법 사용 예제

하나의 문제에 대해 여러 개의 독립적인 추론 경로(Chain of Thought)를 생성하고,
다수결 투표를 통해 가장 일관성 있는 답변을 선택하여 신뢰도를 높이는 방법을 보여줍니다.
"""

import os
import re
from collections import Counter
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.panel import Panel

from ..templates import create_zero_shot_cot_template

load_dotenv()
console = Console()


def extract_final_answer(text: str) -> str:
    """
    텍스트에서 최종 답변을 추출하는 간단한 함수.
    예: "최종 답은 12입니다." -> "12"
    """
    # 숫자만 추출하는 간단한 정규식
    matches = re.findall(r'\d+', text.split('\n')[-1])
    if matches:
        return matches[-1]
    return text.split('\n')[-1].strip()


def run_self_consistency_example():
    """
    Self-Consistency 예제를 실행합니다.
    """
    console.print(Panel.fit("🚀 Self-Consistency 예제 실행", style="bold cyan"))

    # temperature를 높여 다양한 추론 경로를 생성하도록 LLM 설정
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    # CoT 프롬프트 템플릿 사용
    cot_prompt = create_zero_shot_cot_template()
    chain = cot_prompt | llm

    # 복잡한 추론 문제
    problem = "한 상점에 9개의 자전거가 있습니다. 3개는 이륜차이고 나머지는 삼륜차입니다. 이 상점의 모든 자전거 바퀴는 총 몇 개일까요?"
    console.print(Panel(problem, title="📝 문제 정의", border_style="yellow"))

    # --- 여러 번의 추론 경로 생성 ---
    num_trials = 5
    console.print(Panel(f"🔄 {num_trials}개의 서로 다른 추론 경로 생성", style="blue"))
    
    responses = []
    for i in range(num_trials):
        try:
            response = chain.invoke({"problem": problem}).content
            responses.append(response)
            console.print(Panel(response, title=f"추론 경로 {i+1}", border_style="green"))
        except Exception as e:
            console.print(f"❌ 추론 {i+1} 오류: {e}")

    # --- 다수결 투표로 최종 답변 결정 ---
    console.print(Panel("🗳️ 다수결 투표로 최종 답변 결정", style="blue"))
    
    final_answers = [extract_final_answer(res) for res in responses]
    console.print(f"[bold]추출된 최종 답변들:[/bold] {final_answers}")

    if not final_answers:
        console.print("❌ 답변을 추출할 수 없어 종료합니다.")
        return

    # 가장 많이 나온 답변을 찾음
    most_common_answer = Counter(final_answers).most_common(1)[0][0]

    console.print(Panel(f"🏆 최종 일관성 답변: {most_common_answer}", style="bold magenta"))


def main():
    """
    메인 실행 함수
    """
    if not os.getenv("OPENAI_API_KEY"):
        console.print("❌ OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        return

    run_self_consistency_example()
    console.print(Panel.fit("✅ Self-Consistency 예제 실행 완료", style="bold green"))


if __name__ == "__main__":
    main()
