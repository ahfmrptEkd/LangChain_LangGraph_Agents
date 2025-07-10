"""
프롬프트 기법별 템플릿 생성 모듈

이 파일은 다양한 프롬프트 엔지니어링 기법에 대한 
재사용 가능한 `ChatPromptTemplate` 생성 함수들을 정의합니다.

- 기본 템플릿
- Few-shot 템플릿 (정적/동적)
- Chain of Thought (CoT) 템플릿
- Tree of Thought (ToT) 템플릿
"""

from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    PromptTemplate,
)
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List, Dict, Any


# --- 1. 기본 프롬프트 템플릿 ---

def create_basic_prompt_template(system_message: str, human_message: str) -> ChatPromptTemplate:
    """
    가장 기본적인 시스템 + 사용자 메시지 템플릿을 생성합니다.
    
    Args:
        system_message: AI의 역할이나 지시사항 (변수 포함 가능)
        human_message: 사용자의 질문 (변수 포함 가능)
        
    Returns:
        ChatPromptTemplate 객체
    """
    return ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", human_message),
    ])


def create_blog_post_template() -> ChatPromptTemplate:
    """
    블로그 포스트 작성을 위한 복잡한 프롬프트 템플릿을 생성합니다.
    """
    system_template = """
    당신은 {audience}를 대상으로 하는 전문 {role}입니다.
    다음 정보를 바탕으로 블로그 포스트를 작성해주세요.
    
    - 주제: {topic}
    - 톤: {tone}
    - 길이: {length}
    - 추가 요구사항: 구체적인 예시와 실용적인 조언을 포함하여 명확하게 작성하세요.
    """
    return ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", "블로그 포스트 작성을 시작해주세요."),
    ])


# --- 2. Shot 기반 프롬프트 템플릿 ---

def create_static_few_shot_template(
    examples: List[Dict[str, str]],
    system_prefix: str,
    human_suffix: str,
) -> ChatPromptTemplate:
    """
    정적인 예제를 사용하는 Few-shot 프롬프트 템플릿을 생성합니다.
    
    Args:
        examples: Few-shot으로 제공할 예제 리스트 (e.g., [{"input": "...", "output": "..."}, ...])
        system_prefix: 예제가 시작되기 전 시스템 메시지
        human_suffix: 사용자의 최종 질문 (변수 포함)
        
    Returns:
        ChatPromptTemplate 객체
    """
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}"),
    ])
    
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )
    
    return ChatPromptTemplate.from_messages([
        ("system", system_prefix),
        few_shot_prompt,
        ("human", human_suffix),
    ])


def create_dynamic_few_shot_template(
    examples: List[Dict[str, str]],
    system_prefix: str,
    human_suffix: str,
    k: int = 2,
) -> ChatPromptTemplate:
    """
    의미적 유사도 기반의 동적 Few-shot 프롬프트 템플릿을 생성합니다.
    
    Args:
        examples: 전체 예제 풀
        system_prefix: 시스템 메시지
        human_suffix: 사용자 질문
        k: 선택할 예제의 수
        
    Returns:
        ChatPromptTemplate 객체
    """
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{question}"),
        ("ai", "{answer}"),
    ])

    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples,
        OpenAIEmbeddings(),
        FAISS,
        k=k,
        input_keys=["question"],
    )

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
    )

    return ChatPromptTemplate.from_messages([
        ("system", system_prefix),
        few_shot_prompt,
        ("human", "{question}"),
    ])


# --- 3. Chain of Thought (CoT) 프롬프트 템플릿 ---

def create_zero_shot_cot_template() -> ChatPromptTemplate:
    """
    Zero-shot CoT 프롬프트 템플릿을 생성합니다.
    """
    return ChatPromptTemplate.from_messages([
        ("system", "당신은 논리적인 추론가입니다."),
        ("human", "문제: {problem}\n\n이 문제를 해결하기 위해 단계별로 생각해보자."),
    ])


def create_few_shot_cot_template(examples: List[Dict[str, str]]) -> ChatPromptTemplate:
    """
    Few-shot CoT 프롬프트 템플릿을 생성합니다.
    """
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "문제: {problem}"),
        ("ai", "{solution}"),
    ])

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )

    return ChatPromptTemplate.from_messages([
        ("system", "당신은 수학 문제를 단계별로 해결하는 전문가입니다. 다음 예시와 같이 문제를 해결해주세요."),
        few_shot_prompt,
        ("human", "문제: {problem}"),
    ])


# --- 4. Tree of Thought (ToT) 프롬프트 템플릿 ---

def create_tot_branch_generation_template() -> ChatPromptTemplate:
    """
    ToT - 1단계: 여러 사고 경로(가지)를 생성하는 프롬프트 템플릿
    """
    return ChatPromptTemplate.from_template(
        "문제: {problem}\n\n"
        "이 문제를 해결하기 위한 {num_branches}가지의 서로 다른 접근 방법을 제안해주세요. "
        "각 접근 방법은 창의적이고 실용적이어야 합니다."
    )


def create_tot_evaluation_template() -> ChatPromptTemplate:
    """
    ToT - 2단계: 각 사고 경로를 평가하는 프롬프트 템플릿
    """
    return ChatPromptTemplate.from_template(
        "문제: {problem}\n"
        "제안된 접근 방법: {branch}\n"
        "이 접근 방법의 장점, 단점, 그리고 예상 성공률(1-10점)을 평가해주세요. "
        "결과는 JSON 형식으로 제공해주세요. 예: {{'pros': [...], 'cons': [...], 'score': 7}}"
    )


def create_tot_development_template() -> ChatPromptTemplate:
    """
    ToT - 3단계: 가장 좋은 경로를 발전시켜 최종 해결책을 만드는 프롬프트 템플릿
    """
    return ChatPromptTemplate.from_template(
        "문제: {problem}\n"
        "선택된 최적의 접근 방법: {best_branch}\n"
        "이 접근 방법을 바탕으로, 구체적이고 실행 가능한 최종 해결책을 단계별로 제시해주세요."
    )


# --- 5. Self-Reflection 프롬프트 템플릿 ---

def create_self_critique_template() -> ChatPromptTemplate:
    """
    Self-Reflection - 1단계: 생성된 초안을 비평하는 프롬프트 템플릿
    """
    system_message = """
    당신은 전문 비평가입니다. 주어진 초안을 분석하고, 다음 기준에 따라 구체적인 문제점과 개선 방안을 제시해주세요:
    - 논리적 오류나 비약은 없는가?
    - 내용이 명확하고 이해하기 쉬운가?
    - 더 추가되거나 보완되어야 할 정보는 없는가?
    """
    human_message = """
    다음 초안을 비평해주세요:
    -- 초안 시작 --
    {draft}
    -- 초안 끝 --\n\n
    """

    return ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", human_message),
    ])


def create_self_refine_template() -> ChatPromptTemplate:
    """
    Self-Reflection - 2단계: 비평을 바탕으로 초안을 수정하는 프롬프트 템플릿
    """
    system_message = """
    당신은 뛰어난 작가입니다. 주어진 초안과 비평 내용을 바탕으로, 모든 문제점을 해결한 최종 결과물을 작성해주세요.
    """
    human_message = """
    다음 초안과 비평 내용을 참고하여 최종본을 작성하세요.

    -- 초안 시작 --
    {draft}
    -- 초안 끝 --\n\n

    -- 비평 내용 시작 --\n
    {critique}
    -- 비평 내용 끝 --\n\n
    """
    return ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", human_message),
    ])
