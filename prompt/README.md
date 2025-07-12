# LangChain 프롬프트 엔지니어링 기법 라이브러리

이 디렉토리는 LangChain을 이용한 다양한 프롬프트 엔지니어링 기법을 재사용 가능한 템플릿 형태로 정리하고, 사용 예시를 가지고있다.

## 📚 목차

- [프로젝트 구조](#-프로젝트-구조)
- [핵심 컨셉](#-핵심-컨셉)
- [실행 환경 설정](#-실행-환경-설정)
- [사용 방법](#-사용-방법)
- [구현된 프롬프트 기법](#-구현된-프롬프트-기법)
  - [Self-Consistency vs. Self-Reflection](#self-consistency-vs-self-reflection)
  - [CoT vs. ToT: 더 깊은 이해](#cot-vs-tot-더-깊은-이해)
  - [SLM과 CoT: 주의사항](#slm과-cot-주의사항)
- [템플릿 라이브러리 (`templates.py`)](#템플릿-라이브러리-templatespy)
- [참고 자료](#-참고-자료)

## 📂 프로젝트 구조

```
prompt/
├── .env.example           # 환경 변수 예시 파일
├── prompt_templates_comparison.md # 프롬프트 템플릿 비교 문서
├── templates.py           # 템플릿 생성 함수 모음
├── run_examples/          # 템플릿 사용 예제 스크립트
│   ├── run_basic_examples.py
│   ├── run_cot_examples.py
│   ├── run_self_consistency_example.py
│   ├── run_self_reflection_example.py
│   ├── run_shot_examples.py
│   └── run_tot_examples.py
└── README.md              # 본 파일
```

## ✨ 핵심 컨셉

- **모듈화**: `templates.py` 파일에 모든 프롬프트 템플릿 생성 로직을 중앙화하여, 다른 애플리케이션에서 쉽게 `import`하여 사용할 수 있도록 설계되었다.
- **ChatPromptTemplate 우선**: 현대적인 챗봇 모델과의 호환성을 위해 모든 템플릿은 `ChatPromptTemplate`을 기반으로 작성되었다.
- **실행 가능한 예제**: `run_examples/` 디렉토리의 스크립트들은 `templates.py`에 정의된 함수들을 어떻게 활용하는지 보여주는 명확하고 실행 가능한 예제


## 💡 사용 방법

`run_examples/` 디렉토리의 각 파일은 독립적으로 실행 가능하며, 다음과 같이 실행할 수 있다:

```bash
# 예제 스크립트가 있는 디렉토리로 이동
cd prompt/run_examples

# 기본 프롬프트 예제 실행
python run_basic_examples.py

# Shot 기반 프롬프트 예제 실행
python run_shot_examples.py

# CoT 프롬프트 예제 실행
python run_cot_examples.py

# Self-Consistency 예제 실행
python run_self_consistency_example.py

# Self-Reflection 예제 실행
python run_self_reflection_example.py

# Tree of Thought 예제 실행
python run_tot_examples.py
```

## 📖 구현된 프롬프트 기법

아래와 같은 프롬프트 엔지니어링 기법을 구현하고 예제를 제공한다.

| 기법 | 특징 | 관련 함수 (`templates.py`) |
| --- | --- | --- |
| **Basic Prompting** | 가장 기본적인 형태로, 시스템 역할과 사용자 질문을 정의한다. | `create_basic_prompt_template`, `create_blog_post_template` |
| **Few-shot Prompting** | 모델에게 몇 가지 예시(shots)를 제공하여 원하는 결과물의 형식이나 패턴을 학습시킨다. | `create_static_few_shot_template`, `create_dynamic_few_shot_template` |
| **Chain of Thought (CoT)** | 모델이 최종 답변을 내기 전에, 문제 해결 과정을 단계별로 생각하도록 유도하여 논리적 추론 능력을 향상시킨다. | `create_zero_shot_cot_template`, `create_few_shot_cot_template` |
| **Self-Consistency** | 여러 독립적인 추론 경로를 생성하고, 다수결 투표로 가장 일관된 답변을 선택하여 신뢰도를 높인다. | `create_zero_shot_cot_template` (반복 사용) |
| **Self-Reflection** | 생성, 비평, 수정을 반복하여 결과물의 품질과 완성도를 점진적으로 향상시킨다. | `create_self_critique_template`, `create_self_refine_template` |
| **Tree of Thought (ToT)** | 단일 추론 경로가 아닌, 여러 사고의 흐름(가지)을 생성하고 평가하여 가장 유망한 경로를 선택하는 복잡한 문제 해결 기법이다. | `create_tot_..._template` |

### Self-Consistency vs. Self-Reflection

두 기법은 모델이 자신의 결과물을 활용한다는 점에서 유사해 보이지만, 목표와 과정이 다르다.

- **Self-Consistency**는 **정확성**을 높이기 위해, 여러 개의 **독립적인 답안지(추론 경로)를 비교**하여 가장 많이 나온 답을 선택하는 **병렬적** 과정이다.
- **Self-Reflection**은 **품질**을 높이기 위해, **하나의 답안지를 계속 고쳐 쓰는** **반복적** 과정이다.

| 구분 | Self-Consistency | Self-Reflection |
| :--- | :--- | :--- |
| **목표** | 정확성, 신뢰성 | 품질, 완성도 |
| **프로세스** | 병렬적 (다수결 투표) | 반복적 (생성-비평-수정) |
| **핵심 동작** | 여러 추론 경로 생성 | 단일 경로 개선 |

### CoT vs. ToT: 더 깊은 이해

ToT는 단순히 CoT를 병렬로 실행하는 것을 넘어, **'의식적인 탐색과 평가'** 과정이 추가된 진화된 형태다.

- **Chain of Thought (CoT)**는 **선형적인(Linear)** 사고의 사슬을 만듭니다. 한 명의 전문가가 한 길로만 쭉 걸어가는 것과 같아서, 초반에 실수가 발생하면 전체 결과가 틀어질 위험이 있다.

- **Tree of Thought (ToT)**는 문제 해결의 각 단계에서 **여러 갈래의 가능성(가지)을 생성**하고, 어떤 길이 가장 유망한지 **평가하고 선택**합니다. 이는 마치 여러 전문가로 구성된 팀이 다양한 해결책을 동시에 탐색하고 최선의 경로를 선택하는 것과 같습니다. 이 과정 덕분에 ToT는 더 복잡하고 창의적인 문제 해결에 강력한 성능을 보인다.

| 구분 | Chain of Thought (CoT) | Tree of Thought (ToT) |
| :--- | :--- | :--- |
| **탐색 방식** | 선형적 (단일 경로) | 비선형적 (다중 경로 탐색) |
| **의사결정** | 암묵적 (하나의 흐름) | **명시적 평가 및 선택** |
| **적합한 문제**| 논리적, 순차적 문제 | **창의적, 복잡한 계획 문제** |

### SLM과 CoT: 주의사항

한 가지 중요한 점은, CoT와 같은 고급 프롬프트 기법의 효과는 모델의 크기에 크게 의존한다는 것이다.

- **LLM (Large Language Models)**: CoT는 LLM의 추론 능력을 크게 향상시키는, 검증된 기법입니다. 이는 모델의 규모가 일정 수준 이상일 때 발현되는 **창발적 능력(Emergent Ability)**으로 여겨진다.

- **SLM (Small Language Models)**: 반면, 소형 언어 모델에 CoT를 직접 적용하면 오히려 논리적 오류가 포함된 추론을 생성하여 **성능이 저하될 수 있습니다.** SLM은 CoT를 수행할 만큼 충분한 내부 파라미터와 지식을 갖추지 못했을 가능성이 높다.

따라서 이 라이브러리의 CoT/ToT 템플릿들은 **강력한 LLM(예: GPT-4, Claude 3, Gemini Pro 등)과 함께 사용하는 것을 권장합니다.** 만약 SLM에 추론 능력을 부여하고 싶다면, LLM이 생성한 CoT 데이터를 SLM에 학습시키는 **지식 증류(Knowledge Distillation)** 와 같은 별도의 기법을 고려해야한다.

## 템플릿 라이브러리 (`templates.py`)

`prompt/templates.py` 파일은 위에 언급된 기법들을 구현한 템플릿 생성 함수들을 제공합니다. 자신의 코드에서 다음과 같이 쉽게 가져와 사용할 수 있다.

```python
from prompt.templates import create_zero_shot_cot_template
from langchain_openai import ChatOpenAI

# 1. LLM 준비
llm = ChatOpenAI()

# 2. 템플릿 가져오기
cot_prompt = create_zero_shot_cot_template()

# 3. 체인으로 묶어 실행
chain = cot_prompt | llm
response = chain.invoke({"problem": "100에서 27을 빼면 얼마인가요?"})
print(response.content)
```

## 📝 참고 자료

- [LangChain 공식 문서](https://docs.langchain.com/)
- [프롬프트 엔지니어링 가이드](https://www.promptingguide.ai/)
- [Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes](https://arxiv.org/abs/2305.02301) - CoT 지식 증류 관련 핵심 논문
- [Prompt 기법 종류](https://data-minggeul.tistory.com/19)
- [Prompt engineering 기법 종류2](https://modulabs.co.kr/blog/%ED%94%84%EB%A1%AC%ED%94%84%ED%8A%B8-%EC%97%94%EC%A7%80%EB%8B%88%EC%96%B4%EB%A7%81-%EA%B8%B0%EB%B2%95%EA%B3%BC-nlp-%EC%9E%91%EC%97%85-%EB%B6%84%EB%A5%98)
- [CoT prompt](https://plainenglish.io/blog/langchain-in-chains-22-chain-of-thought-prompting)