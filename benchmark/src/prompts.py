"""
Prompt templates for hallucination detection tasks
"""

BINARY_PROMPT_TEMPLATE = """

### 태스크:
다음에 제시되는 대화에서 현재 응답이 할루시네이션인지 판별하세요.
**중요**: 주어진 대화 히스토리와 관련 문서만을 기준으로 판단하세요. LLM의 내부 지식으로 판단하지 마세요.

### 분류 기준:
다음 중 하나라도 해당하면 할루시네이션입니다:
- 답변이 주어진 문서나 이전 대화 히스토리와 직접적으로 모순되는 경우
- 답변에 주어진 문서에 검증 불가능한 정보를 포함하는 경우
- 사용자의 의도나 질문과 논리적으로 연결되지 않는 응답을 하는 경우
- 주어진 문서에 충분한 증거가 있음에도 불구하고 답변을 거부하는 경우
- 주어진 문서로 답변할 수 없는 상황임에도 불구하고 답변을 제공하는 경우

다음의 경우만 할루시네이션이 아닙니다:
- 주어진 문서의 정보를 바탕으로 질문에 올바르게 답변한 경우
- 주어진 문서에 증거가 부족함을 인정하고 답변을 거부하는 경우

### 출력 형식:
다른 설명 없이 오직 0 또는 1만 출력하세요.

0: 할루시네이션 아님
1: 할루시네이션


### 대화:
{dialogue}

### 질문:
{question}

### 관련 문서:
{contexts}

### 답변:
{answer}

"""


FOURCLASS_PROMPT_TEMPLATE = """

### 태스크:
다음에 제시되는 대화에서 현재 응답을 4가지 유형으로 분류하세요.
**중요**: 주어진 대화 히스토리와 관련 문서만을 기준으로 판단하세요. LLM의 내부 지식으로 판단하지 마세요.

### 분류 기준:

**주어진 문서에 질문에 대한 충분한 증거가 있는 경우:**
- Faithful Answer: 주어진 문서의 정보를 바탕으로 질문에 올바르게 답변한 경우
- Hallucination: 다음 중 하나라도 해당하는 경우
  * 답변이 주어진 문서나 이전 대화 히스토리와 직접적으로 모순되는 경우
  * 답변에 주어진 문서에 검증 불가능한 정보를 포함하는 경우
  * 사용자의 의도나 질문과 논리적으로 연결되지 않는 응답을 하는 경우
- False Refusal: 주어진 문서에 충분한 증거가 있음에도 불구하고 답변을 거부하는 경우

**주어진 문서에 질문에 대한 충분한 증거가 없는 경우:**
- Hallucination: 주어진 문서로 답변할 수 없는 상황임에도 불구하고 답변을 제공하는 경우
- True Refusal: 주어진 문서에 증거가 부족함을 인정하고 적절하게 답변을 거부하는 경우

### 출력 형식:
다른 설명 없이 오직 0에서 3 사이의 숫자 하나만 출력하세요.

0: Faithful Answer
1: Hallucination
2: False Refusal
3: True Refusal


### 대화:
{dialogue}

### 질문:
{question}

### 관련 문서:
{contexts}

### 답변:
{answer}

"""


MULTICLASS_PROMPT_TEMPLATE = """

### 태스크:
다음에 제시되는 대화에서 현재 응답이 할루시네이션인지 판별하고, 할루시네이션인 경우 유형을 분류하세요.
**중요**: 주어진 대화 히스토리와 관련 문서만을 기준으로 판단하세요. LLM의 내부 지식으로 판단하지 마세요.

### 분류 기준:
주어진 문서에 질문에 대한 충분한 증거가 있는 경우:
- No Hallucination: 주어진 문서의 정보를 바탕으로 질문에 올바르게 답변한 경우
- Contradictory: 답변이 주어진 문서나 이전 대화 히스토리와 직접적으로 모순되는 경우.
- Unverifiable: 답변에 주어진 문서에 검증 불가능한 정보를 포함하는 경우.
- Irrelevance: 사용자의 의도나 질문과 논리적으로 연결되지 않는 응답을 하는 경우.
- False Refusal: 주어진 문서에 충분한 증거가 있음에도 불구하고 답변을 거부하는 경우.

주어진 문서에 질문에 대한 충거가 없는 경우:
- False Acceptance: 주어진 문서로 답변할 수 없는 상황임에도 불구하고 답변을 제공하는 경우
- No Hallucination: 주어진 문서에 증거가 부족함을 인정하고 답변을 거부하는 경우

### 출력 형식:
다른 설명 없이 오직 할루시네이션 유형을 나타내는 0에서 5 사이의 숫자 하나만 출력하면 됩니다.

0: No Hallucination
1: Contradictory
2: Unverifiable
3: Irrelevance
4: False Refusal
5: False Acceptance


### 대화:
{dialogue}

### 질문:
{question}

### 관련 문서:
{contexts}

### 답변:
{answer}

"""


def get_prompt_template(classification_type):
    """Get prompt template based on classification type"""
    if classification_type == "binary":
        return BINARY_PROMPT_TEMPLATE
    elif classification_type == "fourclass":
        return FOURCLASS_PROMPT_TEMPLATE
    elif classification_type == "fourclass_v2":
        from src.prompts_v2 import FOURCLASS_V2_PROMPT_TEMPLATE
        return FOURCLASS_V2_PROMPT_TEMPLATE
    elif classification_type == "multiclass":
        return MULTICLASS_PROMPT_TEMPLATE
    else:
        raise ValueError(f"Unknown classification type: {classification_type}")
