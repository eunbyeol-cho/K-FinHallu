"""
Preprocess.py - 대화 데이터 전처리 스크립트

복원된 벤치마크 파일(data/full/kfinhallu_{split}.json, scripts/restore_data.py 참고)을
전처리하여 각 레코드의 마지막(평가 대상) 턴으로 두 가지 모드의 샘플을 만듭니다:

1. Hallucination Mode: hallucinated_answer를 사용 (test 404개)
2. Faithful Mode: assistant_answer를 사용 (test 404개)

Usage:
    python -m src.data.preprocess                      # test split (기본값)
    python -m src.data.preprocess --dialogues_path ../data/full/kfinhallu_train.json \
        --context_mapping_path ../data/full/context_mapping_train.json \
        --output_dir ../data/preprocessed_train
"""

import json
import os
import argparse
from typing import List, Dict, Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO_ROOT = os.path.dirname(PROJECT_ROOT)


def load_json(file_path: str) -> Any:
    """JSON 파일을 로드합니다."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, file_path: str) -> None:
    """데이터를 JSON 파일로 저장합니다."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_dialogue_history(dialogue: List[Dict], up_to_turn: int) -> List[Dict]:
    """
    특정 턴까지의 대화 히스토리를 가져옵니다.
    hallucination이 발생한 턴은 제외합니다 (그 이전 턴까지만).
    """
    history = []
    for turn in dialogue:
        if turn['turn'] < up_to_turn:
            history.append({
                'turn': turn['turn'],
                'user_question': turn['user_question'],
                'assistant_answer': turn['assistant_answer']
            })
    return history


def get_contexts_from_passages(passages: List[Dict], context_mapping: Dict) -> List[str]:
    """
    passages에서 context_id를 추출하여 context_mapping에서 실제 context 텍스트를 가져옵니다.
    """
    contexts = []
    for passage in passages:
        context_id = passage['context_id']
        if context_id in context_mapping:
            contexts.append(context_mapping[context_id]['context'])
    return contexts


def get_context_ids_for_mode(turn: Dict, mode: str) -> List[str]:
    """
    모드에 따라 적절한 context_ids를 선택합니다.

    Hallucination mode:
        - false_acceptance: negative_context_ids 사용
        - 나머지: positive_context_ids 사용

    Faithful mode:
        - false_refusal: negative_context_ids 사용
        - 나머지: positive_context_ids 사용
    """
    hallucination_type = turn.get('hallucination_type')

    if mode == 'hallucination':
        if hallucination_type == 'false_acceptance':
            return turn.get('negative_context_ids', [])
        else:
            return turn.get('positive_context_ids', [])
    else:  # faithful mode
        if hallucination_type == 'false_refusal':
            return turn.get('negative_context_ids', [])
        else:
            return turn.get('positive_context_ids', [])


def get_contexts_by_ids(context_ids: List[str], context_mapping: Dict) -> List[str]:
    """
    context_ids 리스트를 받아서 context_mapping에서 실제 context 텍스트를 가져옵니다.
    """
    contexts = []
    for context_id in context_ids:
        if context_id in context_mapping:
            contexts.append(context_mapping[context_id]['context'])
    return contexts


def process_dialogues(
    dialogues_path: str = None,
    context_mapping_path: str = None,
    output_path_hallucination: str = None,
    output_path_faithful: str = None
) -> None:
    """
    대화 데이터를 전처리합니다.
    """
    # 기본 경로 설정
    if dialogues_path is None:
        dialogues_path = os.path.join(REPO_ROOT, 'data', 'full', 'kfinhallu_test.json')
    if context_mapping_path is None:
        context_mapping_path = os.path.join(REPO_ROOT, 'data', 'full', 'context_mapping_test.json')
    if output_path_hallucination is None:
        output_path_hallucination = os.path.join(REPO_ROOT, 'data', 'preprocessed', 'preprocessed_hallucination_mode.json')
    if output_path_faithful is None:
        output_path_faithful = os.path.join(REPO_ROOT, 'data', 'preprocessed', 'preprocessed_faithful_mode.json')

    # 데이터 로드
    print("Loading data...")
    dialogues = load_json(dialogues_path)
    context_mapping = load_json(context_mapping_path)

    hallucination_mode_data = []
    faithful_mode_data = []

    print(f"Processing {len(dialogues)} dialogues...")

    for dialogue in dialogues:
        dialogue_id = dialogue['dialogue_id']

        # hallucination이 발생한 턴 찾기
        hallucination_turn = None
        for turn in dialogue['dialogue']:
            if turn.get('hallucination_type') is not None:
                hallucination_turn = turn
                break

        if hallucination_turn is None:
            continue

        # 이전 대화 히스토리
        dialogue_history = get_dialogue_history(
            dialogue['dialogue'],
            hallucination_turn['turn']
        )

        # 현재 질문
        current_question = hallucination_turn['user_question']

        # Hallucination mode 처리
        hallucination_answer = hallucination_turn.get('hallucinated_answer')
        hallucination_context_ids = get_context_ids_for_mode(hallucination_turn, 'hallucination')
        hallucination_contexts = get_contexts_by_ids(hallucination_context_ids, context_mapping)

        injection_model = dialogue.get('injection_model')

        hallu_sample = {
            'dialogue_id': dialogue_id,
            'turn': hallucination_turn['turn'],
            'mode': 'hallucination',
            'hallucination_type': hallucination_turn['hallucination_type'],
            'dialogue_history': dialogue_history,
            'current_question': current_question,
            'current_answer': hallucination_answer,
            'contexts': hallucination_contexts,
            'context_ids': hallucination_context_ids,
            'clue_text': hallucination_turn.get('clue_text'),
        }
        if injection_model:
            hallu_sample['injection_model'] = injection_model
        hallucination_mode_data.append(hallu_sample)

        # Faithful mode 처리
        faithful_answer = hallucination_turn.get('assistant_answer')
        faithful_context_ids = get_context_ids_for_mode(hallucination_turn, 'faithful')
        faithful_contexts = get_contexts_by_ids(faithful_context_ids, context_mapping)

        faithful_sample = {
            'dialogue_id': dialogue_id,
            'turn': hallucination_turn['turn'],
            'mode': 'faithful',
            'hallucination_type': hallucination_turn['hallucination_type'],
            'dialogue_history': dialogue_history,
            'current_question': current_question,
            'current_answer': faithful_answer,
            'contexts': faithful_contexts,
            'context_ids': faithful_context_ids,
            'clue_text': hallucination_turn.get('clue_text'),
        }
        if injection_model:
            faithful_sample['injection_model'] = injection_model
        faithful_mode_data.append(faithful_sample)

    # 결과 저장
    os.makedirs(os.path.dirname(output_path_hallucination), exist_ok=True)
    os.makedirs(os.path.dirname(output_path_faithful), exist_ok=True)

    print(f"\nSaving results...")
    print(f"Hallucination mode samples: {len(hallucination_mode_data)}")
    print(f"Faithful mode samples: {len(faithful_mode_data)}")

    save_json(hallucination_mode_data, output_path_hallucination)
    save_json(faithful_mode_data, output_path_faithful)

    print(f"\nSaved to:")
    print(f"  - {output_path_hallucination}")
    print(f"  - {output_path_faithful}")

    # 샘플 출력
    if hallucination_mode_data:
        print("\n" + "="*80)
        print("Sample from Hallucination Mode:")
        print("="*80)
        sample = hallucination_mode_data[0]
        print(f"Dialogue ID: {sample['dialogue_id']}")
        print(f"Turn: {sample['turn']}")
        print(f"Hallucination Type: {sample['hallucination_type']}")
        print(f"Number of context IDs: {len(sample['context_ids'])}")
        print(f"\nCurrent Question: {sample['current_question']}")
        print(f"\nCurrent Answer: {sample['current_answer']}")
        print(f"\nDialogue History: {len(sample['dialogue_history'])} previous turns")
        print(f"Contexts: {len(sample['contexts'])} contexts retrieved")

    if faithful_mode_data:
        print("\n" + "="*80)
        print("Sample from Faithful Mode:")
        print("="*80)
        sample = faithful_mode_data[0]
        print(f"Dialogue ID: {sample['dialogue_id']}")
        print(f"Turn: {sample['turn']}")
        print(f"Hallucination Type: {sample['hallucination_type']}")
        print(f"Number of context IDs: {len(sample['context_ids'])}")
        print(f"\nCurrent Question: {sample['current_question']}")
        print(f"\nCurrent Answer: {sample['current_answer']}")
        print(f"\nDialogue History: {len(sample['dialogue_history'])} previous turns")
        print(f"Contexts: {len(sample['contexts'])} contexts retrieved")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess dialogue data')
    parser.add_argument('--dialogues_path', type=str, default=None,
                        help='Path to dialogues JSON file')
    parser.add_argument('--context_mapping_path', type=str, default=None,
                        help='Path to context mapping JSON file')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for preprocessed data')
    args = parser.parse_args()

    # output_dir이 지정된 경우 개별 output path로 변환
    output_path_hallucination = None
    output_path_faithful = None
    if args.output_dir is not None:
        output_path_hallucination = os.path.join(args.output_dir, 'preprocessed_hallucination_mode.json')
        output_path_faithful = os.path.join(args.output_dir, 'preprocessed_faithful_mode.json')

    process_dialogues(
        dialogues_path=args.dialogues_path,
        context_mapping_path=args.context_mapping_path,
        output_path_hallucination=output_path_hallucination,
        output_path_faithful=output_path_faithful
    )
