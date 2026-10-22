"""
SFT 학습 데이터 변환 스크립트 (v2 - 판단 근거 포함)

v1 대비 변경점:
- 프롬프트: FOURCLASS_V2_PROMPT_TEMPLATE 사용
- 응답: 판단 근거(한 줄) + 레이블 번호 (clue_text 기반 템플릿)

출력 형식:
    {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}

Usage:
    python -m src.data.prepare_sft_data_v2 --input_dir data/preprocessed_train
"""

import json
import os
import random
import argparse
from typing import List, Dict

from src.prompts_v2 import (
    FOURCLASS_V2_PROMPT_TEMPLATE,
    generate_response,
)
from src.data.prepare_sft_data import SYSTEM_PROMPT
from src.utils import (
    load_json,
    format_dialogue_history,
    format_contexts,
    get_ground_truth,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO_ROOT = os.path.dirname(PROJECT_ROOT)


def create_sft_sample_v2(sample: Dict) -> Dict:
    """단일 샘플을 ms-swift messages 형식으로 변환 (v2)"""
    # 사용자 프롬프트
    dialogue = format_dialogue_history(sample['dialogue_history'])
    question = sample['current_question']
    contexts = format_contexts(sample['contexts'])
    answer = sample['current_answer']

    user_prompt = FOURCLASS_V2_PROMPT_TEMPLATE.format(
        dialogue=dialogue,
        question=question,
        contexts=contexts,
        answer=answer,
    )

    # 레이블 및 판단 근거 생성
    label = get_ground_truth(sample, 'fourclass')
    reasoning_and_label = generate_response(
        label=label,
        hallucination_type=sample['hallucination_type'],
        mode=sample['mode'],
        clue_text=sample.get('clue_text'),
        question=question,
    )

    # Qwen3 thinking mode: 판단 근거를 <think> 안에, 레이블 번호만 밖에
    assistant_response = f"<think>\n{reasoning_and_label}\n</think>\n\n{label}"

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response},
        ]
    }


def prepare_sft_data_v2(
    input_dir: str = None,
    output_dir: str = None,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    """SFT 학습 데이터를 생성한다 (v2)."""
    if input_dir is None:
        input_dir = os.path.join(REPO_ROOT, 'data', 'preprocessed')
    if output_dir is None:
        output_dir = os.path.join(REPO_ROOT, 'data', 'sft')

    os.makedirs(output_dir, exist_ok=True)

    # 데이터 로드
    print("Loading preprocessed data...")
    hallucination_data = load_json(os.path.join(input_dir, 'preprocessed_hallucination_mode.json'))
    faithful_data = load_json(os.path.join(input_dir, 'preprocessed_faithful_mode.json'))

    print(f"Loaded {len(hallucination_data)} hallucination samples")
    print(f"Loaded {len(faithful_data)} faithful samples")

    # clue_text 존재 여부 확인
    clue_count = sum(1 for s in hallucination_data + faithful_data if s.get('clue_text'))
    total = len(hallucination_data) + len(faithful_data)
    print(f"Samples with clue_text: {clue_count}/{total}")

    # 모든 샘플 변환
    all_samples = []
    label_counts = {}

    for sample in hallucination_data + faithful_data:
        sft_sample = create_sft_sample_v2(sample)
        all_samples.append(sft_sample)

        gt = get_ground_truth(sample, 'fourclass')
        label_counts[gt] = label_counts.get(gt, 0) + 1

    print(f"\nTotal SFT samples: {len(all_samples)}")
    print(f"\nLabel distribution:")
    label_names = {0: 'Faithful Answer', 1: 'Hallucination', 2: 'False Refusal', 3: 'True Refusal'}
    for label in sorted(label_counts.keys()):
        print(f"  {label} ({label_names[label]}): {label_counts[label]}")

    # Train/Val 분할
    random.seed(seed)
    random.shuffle(all_samples)

    if val_ratio > 0:
        val_size = max(1, int(len(all_samples) * val_ratio))
        val_samples = all_samples[:val_size]
        train_samples = all_samples[val_size:]
    else:
        train_samples = all_samples
        val_samples = []

    print(f"\nTrain samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")

    # JSONL 저장
    train_path = os.path.join(output_dir, 'train_fourclass_v2.jsonl')
    with open(train_path, 'w', encoding='utf-8') as f:
        for sample in train_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"\nSaved train data to: {train_path}")

    if val_samples:
        val_path = os.path.join(output_dir, 'val_fourclass_v2.jsonl')
        with open(val_path, 'w', encoding='utf-8') as f:
            for sample in val_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        print(f"Saved val data to: {val_path}")

    # 샘플 출력
    print("\n" + "=" * 80)
    print("Sample assistant response (first train sample):")
    print("=" * 80)
    print(train_samples[0]['messages'][2]['content'])


def main():
    parser = argparse.ArgumentParser(description='Prepare SFT training data (v2)')
    parser.add_argument('--input_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    prepare_sft_data_v2(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
