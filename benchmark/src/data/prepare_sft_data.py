"""
SFT 학습 데이터 변환 스크립트

기존 preprocessed 데이터를 ms-swift messages format (JSONL)으로 변환합니다.

출력 형식:
    {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "<prompt>"}, {"role": "assistant", "content": "<label_with_reasoning>"}]}

Usage:
    python -m src.data.prepare_sft_data --classification_type fourclass
    python -m src.data.prepare_sft_data --classification_type fourclass --val_ratio 0.1
"""

import json
import os
import random
import argparse
from typing import List, Dict

from src.utils import (
    load_json,
    create_prompt,
    get_ground_truth
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO_ROOT = os.path.dirname(PROJECT_ROOT)

# 분류 유형별 레이블 매핑
LABEL_DESCRIPTIONS = {
    'binary': {
        0: "0 (할루시네이션 아님)",
        1: "1 (할루시네이션)"
    },
    'fourclass': {
        0: "0 (Faithful Answer)",
        1: "1 (Hallucination)",
        2: "2 (False Refusal)",
        3: "3 (True Refusal)"
    },
    'multiclass': {
        0: "0 (No Hallucination)",
        1: "1 (Contradictory)",
        2: "2 (Unverifiable)",
        3: "3 (Irrelevance)",
        4: "4 (False Refusal)",
        5: "5 (False Acceptance)"
    }
}

SYSTEM_PROMPT = "당신은 한국 금융 도메인의 할루시네이션 탐지 전문가입니다. 주어진 대화와 문서를 바탕으로 응답의 정확성을 판단하세요."


def create_sft_sample(sample: Dict, classification_type: str) -> Dict:
    """단일 샘플을 ms-swift messages 형식으로 변환"""
    # 사용자 프롬프트 생성 (기존 인퍼런스와 동일한 프롬프트)
    user_prompt = create_prompt(sample, classification_type)

    # Ground truth 레이블
    gt_label = get_ground_truth(sample, classification_type)
    label_desc = LABEL_DESCRIPTIONS[classification_type][gt_label]

    # 어시스턴트 응답: 빈 <think> 태그 + 레이블
    # Qwen3의 thinking mode를 비활성화하기 위해 빈 think 블록 포함
    # (ms-swift loss_scale=ignore_empty_think와 함께 사용)
    assistant_response = f"<think>\n\n</think>\n\n{gt_label}"

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response}
        ]
    }


def prepare_sft_data(
    input_dir: str = None,
    output_dir: str = None,
    classification_type: str = "fourclass",
    val_ratio: float = 0.1,
    seed: int = 42
):
    """SFT 학습 데이터를 생성합니다."""
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

    # 모든 샘플 변환
    all_samples = []
    for sample in hallucination_data:
        sft_sample = create_sft_sample(sample, classification_type)
        all_samples.append(sft_sample)

    for sample in faithful_data:
        sft_sample = create_sft_sample(sample, classification_type)
        all_samples.append(sft_sample)

    print(f"Total SFT samples: {len(all_samples)}")

    # 레이블 분포 출력
    label_counts = {}
    for sample in hallucination_data + faithful_data:
        gt = get_ground_truth(sample, classification_type)
        label_counts[gt] = label_counts.get(gt, 0) + 1

    print(f"\nLabel distribution ({classification_type}):")
    for label in sorted(label_counts.keys()):
        desc = LABEL_DESCRIPTIONS[classification_type][label]
        print(f"  {desc}: {label_counts[label]}")

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

    # JSONL 형식으로 저장
    train_path = os.path.join(output_dir, f'train_{classification_type}.jsonl')
    with open(train_path, 'w', encoding='utf-8') as f:
        for sample in train_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"\nSaved train data to: {train_path}")

    if val_samples:
        val_path = os.path.join(output_dir, f'val_{classification_type}.jsonl')
        with open(val_path, 'w', encoding='utf-8') as f:
            for sample in val_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        print(f"Saved val data to: {val_path}")


def main():
    parser = argparse.ArgumentParser(description='Prepare SFT training data')
    parser.add_argument('--input_dir', type=str, default=None,
                        help='Input directory with preprocessed data')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for SFT data')
    parser.add_argument('--classification_type', type=str, default='fourclass',
                        choices=['binary', 'fourclass', 'multiclass'],
                        help='Classification type (default: fourclass)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation split ratio (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()

    prepare_sft_data(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        classification_type=args.classification_type,
        val_ratio=args.val_ratio,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
