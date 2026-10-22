"""
Utility functions for hallucination detection
"""

import json
import re
from src.prompts import get_prompt_template


def load_json(file_path):
    """JSON 파일 로드"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def format_dialogue_history(dialogue_history):
    """대화 히스토리를 문자열로 포맷"""
    if not dialogue_history:
        return "없음"

    formatted = []
    for turn in dialogue_history:
        formatted.append(f"사용자: {turn['user_question']}")
        formatted.append(f"어시스턴트: {turn['assistant_answer']}")

    return "\n".join(formatted)


def format_contexts(contexts):
    """컨텍스트 리스트를 문자열로 포맷"""
    if not contexts:
        return "없음"

    formatted = []
    for i, context in enumerate(contexts, 1):
        formatted.append(f"[문서 {i}]\n{context}")

    return "\n\n".join(formatted)


def create_prompt(sample, classification_type):
    """샘플 데이터로 프롬프트 생성"""
    dialogue = format_dialogue_history(sample['dialogue_history'])
    question = sample['current_question']
    contexts = format_contexts(sample['contexts'])
    answer = sample['current_answer']

    template = get_prompt_template(classification_type)
    prompt = template.format(
        dialogue=dialogue,
        question=question,
        contexts=contexts,
        answer=answer
    )

    return prompt


def sanitize_filename(name):
    """파일명에 사용할 수 없는 문자를 제거"""
    name = re.sub(r'[\/\\\s]+', '_', name)
    name = name.lower()
    return name


def parse_prediction(prediction_text, classification_type):
    """
    LLM 출력에서 reasoning과 final answer를 파싱

    v1 형식: <think>...</think>\n\n{숫자}
    v2 형식: <think>...</think>\n\n판단근거: ...\n분류: {숫자}

    Returns:
        tuple: (reasoning, final_answer_number)
    """
    if not prediction_text:
        return None, None

    reasoning = None
    final_answer = None

    # <think> 태그 추출
    think_match = re.search(r'<think>(.*?)</think>', prediction_text, re.DOTALL)
    think_content = think_match.group(1).strip() if think_match else None

    # <think> 태그 밖의 내용
    text_outside_think = re.sub(r'<think>.*?</think>', '', prediction_text, flags=re.DOTALL).strip()

    # 숫자 범위 결정
    if classification_type == "binary":
        num_range = r'[01]'
    elif classification_type in ("fourclass", "fourclass_v2"):
        num_range = r'[0-3]'
    else:  # multiclass
        num_range = r'[0-5]'

    # 1순위: "분류: {숫자}" 패턴 (v2 형식)
    classify_match = re.search(rf'분류:\s*({num_range})', text_outside_think)
    if classify_match:
        final_answer = int(classify_match.group(1))
        # 판단근거 추출
        reason_match = re.search(r'판단근거:\s*(.+)', text_outside_think)
        if reason_match:
            reasoning = reason_match.group(1).strip()
        return reasoning, final_answer

    # 2순위: 숫자만 있는 줄 (v1 형식)
    lines = text_outside_think.split('\n')
    bare_num_pattern = rf'^({num_range})$'
    for line in reversed(lines):
        line_stripped = line.strip()
        match = re.match(bare_num_pattern, line_stripped)
        if match:
            final_answer = int(match.group(1))
            if think_content:
                reasoning = think_content
            return reasoning, final_answer

    # 3순위: fallback - 텍스트에서 마지막 숫자
    fallback_pattern = rf'\b({num_range})\b'
    numbers = re.findall(fallback_pattern, text_outside_think)
    if numbers:
        final_answer = int(numbers[-1])
    else:
        numbers_all = re.findall(fallback_pattern, prediction_text)
        if numbers_all:
            final_answer = int(numbers_all[-1])

    return reasoning, final_answer


def map_hallucination_type_to_number(hallucination_type):
    """hallucination_type 문자열을 숫자로 매핑 (multiclass용)"""
    mapping = {
        'inconsistency': 1,
        'contradictory': 1,
        'unverifiable': 2,
        'irrelevance': 3,
        'false_refusal': 4,
        'false_acceptance': 5
    }
    return mapping.get(hallucination_type.lower(), 0)


def map_hallucination_type_to_fourclass(hallucination_type, mode):
    """
    hallucination_type을 4-class로 매핑

    0: Faithful Answer (정답 제공)
    1: Hallucination (잘못된 답변)
    2: False Refusal (부적절한 거부)
    3: True Refusal (적절한 거부)
    """
    hallucination_type = hallucination_type.lower()

    if mode == 'faithful':
        # Faithful mode
        if hallucination_type == 'false_refusal':
            # 문서에 증거가 없어서 적절히 거부한 경우
            return 3  # True Refusal
        else:
            # 문서 기반으로 올바르게 답변한 경우
            return 0  # Faithful Answer
    else:
        # Hallucination mode
        if hallucination_type == 'false_refusal':
            # 문서에 증거가 있는데 거부한 경우
            return 2  # False Refusal
        else:
            # contradictory, unverifiable, irrelevance, false_acceptance
            return 1  # Hallucination


def get_ground_truth(sample, classification_type):
    """Ground truth 계산"""
    if classification_type == "binary":
        # Binary: Faithful mode = 0, Hallucination mode = 1
        return 0 if sample['mode'] == 'faithful' else 1
    elif classification_type in ("fourclass", "fourclass_v2"):
        # Four-class classification
        return map_hallucination_type_to_fourclass(
            sample['hallucination_type'],
            sample['mode']
        )
    else:
        # Multiclass: Faithful mode = 0, Hallucination mode = 해당 타입 번호
        if sample['mode'] == 'faithful':
            return 0
        else:
            return map_hallucination_type_to_number(sample['hallucination_type'])
