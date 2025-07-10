"""
Tree of Thought (ToT) 프롬프트 템플릿 사용 예제 (리팩토링 버전)

`templates.py`에 정의된 ToT 프롬프트 생성 함수들을 사용하여
복잡한 문제 해결 과정을 시뮬레이션합니다.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree

from ..templates import (
    create_tot_branch_generation_template,
    create_tot_evaluation_template,
    create_tot_development_template,
)

load_dotenv()
console = Console()


def run_tot_example():
    """
    Tree of Thought 예제를 실행합니다.
    """
    console.print(Panel.fit("🚀 Tree of Thought (ToT) 예제 실행", style="bold magenta"))

    # LLM 인스턴스 생성
    generator_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    evaluator_llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
    developer_llm = ChatOpenAI(model="gpt-4o", temperature=0.5)

    # 문제 정의
    problem = "회사의 직원 만족도를 높이기 위한, 예산이 한정된 상태에서 모든 직원에게 공평하게 돌아갈 수 있는 방안을 제안하세요."
    console.print(Panel(problem, title="📝 문제 정의", border_style="yellow"))

    # --- 1단계: 사고 경로(가지) 생성 ---
    console.print(Panel("🌳 1. 사고 경로 생성", style="blue"))
    generation_prompt = create_tot_branch_generation_template()
    generation_chain = generation_prompt | generator_llm
    
    try:
        response = generation_chain.invoke({"problem": problem, "num_branches": 3})
        branches_text = response.content
        branches = [line.strip()[2:] for line in branches_text.split('\n') if line.strip().startswith(('1.', '2.', '3.'))]
        
        tree = Tree("🌿 생성된 사고 경로")
        for i, branch in enumerate(branches):
            tree.add(f"[green]경로 {i+1}:[/green] {branch}")
        console.print(tree)

    except Exception as e:
        console.print(f"❌ 사고 경로 생성 오류: {e}")
        return

    # --- 2단계: 각 경로 평가 ---
    console.print(Panel("📊 2. 각 경로 평가 (JSON 출력)", style="blue"))
    evaluation_prompt = create_tot_evaluation_template()
    evaluation_chain = evaluation_prompt | evaluator_llm | JsonOutputParser()

    evaluations = []
    for i, branch in enumerate(branches):
        console.print(f"평가 중: 경로 {i+1}")
        try:
            eval_result = evaluation_chain.invoke({"problem": problem, "branch": branch})
            eval_result['branch'] = branch # 결과에 브랜치 정보 추가
            evaluations.append(eval_result)
            console.print(eval_result)
        except Exception as e:
            console.print(f"❌ 경로 평가 오류: {e}")
            evaluations.append({'branch': branch, 'score': 0, 'pros': [], 'cons': []})

    # --- 3단계: 최고 점수 경로 선택 ---
    console.print(Panel("🏆 3. 최고 점수 경로 선택", style="blue"))
    if not evaluations:
        console.print("❌ 평가된 경로가 없어 실행을 중단합니다.")
        return
        
    best_evaluation = max(evaluations, key=lambda x: x.get('score', 0))
    console.print(f"✨ 선택된 최적 경로: {best_evaluation['branch']}")
    console.print(f"📈 점수: {best_evaluation.get('score', 'N/A')}")

    # --- 4단계: 최종 해결책 개발 ---
    console.print(Panel("🚀 4. 최종 해결책 개발", style="blue"))
    development_prompt = create_tot_development_template()
    development_chain = development_prompt | developer_llm

    try:
        final_solution = development_chain.invoke({
            "problem": problem,
            "best_branch": best_evaluation['branch']
        })
        console.print(Panel(final_solution.content, title="💡 최종 해결책", border_style="green"))
    except Exception as e:
        console.print(f"❌ 해결책 개발 오류: {e}")


def main():
    """
    메인 실행 함수
    """
    if not os.getenv("OPENAI_API_KEY"):
        console.print("❌ OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        return

    run_tot_example()
    console.print(Panel.fit("✅ ToT 예제 실행 완료", style="bold green"))


if __name__ == "__main__":
    main()
