"""
Prompt templates v2 for fourclass hallucination detection

v1 대비 변경점:
- 출력 형식: "숫자 하나만 출력" → "판단근거: + 분류: {번호}"
- 학습 데이터 생성 시 clue_text 기반 템플릿 응답 생성
- 시스템 프롬프트, 분류 기준 등 나머지는 v1과 동일
"""

from src.prompts import FOURCLASS_PROMPT_TEMPLATE

# v1 프롬프트에서 출력 형식 부분만 변경
FOURCLASS_V2_PROMPT_TEMPLATE = FOURCLASS_PROMPT_TEMPLATE.replace(
    """### 출력 형식:
다른 설명 없이 오직 0에서 3 사이의 숫자 하나만 출력하세요.

0: Faithful Answer
1: Hallucination
2: False Refusal
3: True Refusal""",
    """### 출력 형식:
다음 형식에 맞추어 출력하시오.

판단근거: {{참조 문서의 근거 존재 여부 및 현재 응답의 적절성에 대한 판단}}
분류: {{0-3}}

0: Faithful Answer
1: Hallucination
2: False Refusal
3: True Refusal"""
)

# 학습 데이터 생성용 응답 템플릿 (4-class)
# {evidence}: 근거 존재 여부 문구, {judgement}: 응답 적절성 문구
RESPONSE_TEMPLATES = {
    0: '판단근거: 참조 문서에 {evidence} 답변이 가능하며, 현재 응답이 적절히 답변하였으므로 Faithful Answer.\n분류: 0',
    1: '판단근거: {question_prefix}참조 문서에 {evidence} 답변이 {possible}, 현재 응답이 {judgement} Hallucination.\n분류: 1',
    2: '판단근거: 참조 문서에 {evidence} 답변이 가능하나, 현재 응답이 답변을 거부하였으므로 False Refusal.\n분류: 2',
    3: '판단근거: {question_prefix}참조 문서에 관련 근거가 부재하여 답변이 불가하며, 현재 응답이 적절히 답변을 거부하였으므로 True Refusal.\n분류: 3',
}


def _format_evidence(clue_text):
    """clue_text 유무에 따라 근거 문구를 생성한다."""
    if clue_text:
        clue = clue_text if len(clue_text) <= 100 else clue_text[:100] + '...'
        return f'"{clue}"가 존재하여'
    return '관련 근거가 부재하여'


def generate_response(label, hallucination_type, mode, clue_text=None, question=None):
    """
    4-class 레이블과 메타데이터로부터 학습용 응답(판단근거 + 분류)을 생성한다.

    Args:
        label: 4-class 레이블 (0-3)
        hallucination_type: 원본 hallucination 유형
        mode: 'hallucination' 또는 'faithful'
        clue_text: 참조 문서 내 근거 텍스트 (positive context인 경우)
        question: 현재 질문 (negative context일 때 판단근거에 포함)

    Returns:
        "판단근거: ...\n분류: {번호}" 형식 문자열
    """
    # negative context 사용 여부 판별
    uses_negative_context = (
        (mode == 'hallucination' and hallucination_type == 'false_acceptance')
        or (mode == 'faithful' and hallucination_type == 'false_refusal')
    )

    # negative context면 clue 없음
    clue = None if uses_negative_context else clue_text
    evidence = _format_evidence(clue)

    # negative context일 때 질문 언급
    question_prefix = ''
    if uses_negative_context and question:
        q = question if len(question) <= 100 else question[:100] + '...'
        question_prefix = f'"{q}"에 대해 '

    if label == 0:
        return RESPONSE_TEMPLATES[0].format(evidence=evidence)
    elif label == 1:
        if uses_negative_context:
            return RESPONSE_TEMPLATES[1].format(question_prefix=question_prefix, evidence=evidence, possible='불가하나', judgement='답변을 제공하였으므로')
        else:
            return RESPONSE_TEMPLATES[1].format(question_prefix='', evidence=evidence, possible='가능하나', judgement='부적절한 답변을 제공하였으므로')
    elif label == 2:
        return RESPONSE_TEMPLATES[2].format(evidence=evidence)
    elif label == 3:
        return RESPONSE_TEMPLATES[3].format(question_prefix=question_prefix)
