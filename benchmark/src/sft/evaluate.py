"""
SFT 모델 평가 스크립트

학습된 LoRA 어댑터를 로드하여 벤치마크 인퍼런스를 실행합니다.
ms-swift infer 또는 vLLM 서빙을 통해 기존 인퍼런스 파이프라인을 재사용합니다.

Usage:
    # ms-swift infer로 직접 평가
    python -m src.sft.evaluate --adapter_path data/results/sft/checkpoint-xxx --classification_type fourclass

    # vLLM 서빙 후 기존 인퍼런스 파이프라인 사용
    python -m src.sft.evaluate --model_name Qwen/Qwen3-8B --port 8000 --classification_type fourclass
"""

import json
import os
import sys
import subprocess
import argparse
from typing import Optional

from src.utils import (
    load_json,
    create_prompt,
    parse_prediction,
    get_ground_truth,
    sanitize_filename
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO_ROOT = os.path.dirname(PROJECT_ROOT)


def evaluate_with_swift_infer(
    adapter_path: str,
    classification_type: str = "fourclass",
    data_dir: str = None,
    output_dir: str = None,
    max_length: int = 512
):
    """ms-swift infer를 사용하여 평가"""
    if data_dir is None:
        data_dir = os.path.join(REPO_ROOT, 'data', 'preprocessed')
    if output_dir is None:
        output_dir = os.path.join(REPO_ROOT, 'results', 'sft')

    os.makedirs(output_dir, exist_ok=True)

    # 데이터 로드
    print("Loading data...")
    hallucination_data = load_json(os.path.join(data_dir, 'preprocessed_hallucination_mode.json'))
    faithful_data = load_json(os.path.join(data_dir, 'preprocessed_faithful_mode.json'))

    all_samples = hallucination_data + faithful_data
    print(f"Total samples: {len(all_samples)}")

    # ms-swift infer를 위한 입력 파일 생성
    infer_input_path = os.path.join(output_dir, 'infer_input.jsonl')
    with open(infer_input_path, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            prompt = create_prompt(sample, classification_type)
            entry = {
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"Created inference input: {infer_input_path}")

    # ms-swift infer 실행
    infer_output_path = os.path.join(output_dir, 'infer_output.jsonl')

    # swift 바이너리 경로 탐색
    import shutil
    swift_bin = shutil.which('swift') or os.path.join(os.path.dirname(sys.executable), 'swift')

    cmd = [
        swift_bin, 'infer',
        '--adapters', adapter_path,
        '--infer_backend', 'vllm',
        '--max_new_tokens', str(max_length),
        '--val_dataset', infer_input_path,
        '--result_path', infer_output_path,
        '--vllm_enforce_eager',
        '--vllm_max_model_len', '9216',
        '--vllm_gpu_memory_utilization', '0.9',
    ]

    print(f"\nRunning ms-swift infer...")
    print(f"Command: {' '.join(cmd)}")

    env = os.environ.copy()
    env.setdefault('HF_HOME', os.environ.get('HF_CACHE', os.path.expanduser('~/.cache/huggingface/hub')))
    env.setdefault('USE_HF', '1')

    result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)

    if result.returncode != 0:
        print(f"Inference failed with return code: {result.returncode}")
        return

    # 결과 분석
    print("\nAnalyzing results...")
    _analyze_results(all_samples, infer_output_path, classification_type, output_dir, adapter_path)


def evaluate_with_vllm(
    model_name: str,
    port: int,
    classification_type: str = "fourclass",
    data_dir: str = None,
    output_dir: str = None,
    sample_size: Optional[int] = None
):
    """vLLM 서빙된 모델로 평가 (기존 인퍼런스 파이프라인 재사용)"""
    from src.inference.run_inference import evaluate_samples

    if data_dir is None:
        data_dir = os.path.join(REPO_ROOT, 'data', 'preprocessed')
    if output_dir is None:
        output_dir = os.path.join(REPO_ROOT, 'results', 'sft')

    os.makedirs(output_dir, exist_ok=True)

    # 데이터 로드
    print("Loading data...")
    hallucination_data = load_json(os.path.join(data_dir, 'preprocessed_hallucination_mode.json'))
    faithful_data = load_json(os.path.join(data_dir, 'preprocessed_faithful_mode.json'))

    if sample_size:
        hallucination_data = hallucination_data[:sample_size]
        faithful_data = faithful_data[:sample_size]

    model_name_safe = sanitize_filename(model_name) + '_sft'

    # 평가 실행
    hal_results, hal_metrics = evaluate_samples(
        hallucination_data, "Hallucination Mode", model_name,
        classification_type, 'vllm', port
    )

    faith_results, faith_metrics = evaluate_samples(
        faithful_data, "Faithful Mode", model_name,
        classification_type, 'vllm', port
    )

    # 결과 저장
    if classification_type == "binary":
        overall_accuracy = (hal_metrics['accuracy'] + faith_metrics['accuracy']) / 2
    else:
        overall_accuracy = (hal_metrics + faith_metrics) / 2

    all_results = {
        'metadata': {
            'model_name': model_name,
            'classification_type': classification_type,
            'eval_type': 'sft_vllm',
        },
        'hallucination_mode': {
            'results': hal_results,
            'metrics': hal_metrics
        },
        'faithful_mode': {
            'results': faith_results,
            'metrics': faith_metrics
        },
        'overall_accuracy': overall_accuracy
    }

    output_file = os.path.join(output_dir, f'sft_eval_{model_name_safe}_{classification_type}.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nOverall Accuracy: {overall_accuracy:.2%}")
    print(f"Results saved to: {output_file}")


def _analyze_results(all_samples, infer_output_path, classification_type, output_dir, adapter_path):
    """ms-swift infer 결과 분석"""
    predictions = []
    with open(infer_output_path, 'r', encoding='utf-8') as f:
        for line in f:
            predictions.append(json.loads(line.strip()))

    correct = 0
    total = 0
    results = []

    for sample, pred_entry in zip(all_samples, predictions):
        prediction_raw = pred_entry.get('response', '')
        reasoning, pred_num = parse_prediction(prediction_raw, classification_type)
        ground_truth = get_ground_truth(sample, classification_type)

        is_correct = pred_num is not None and pred_num == ground_truth
        if is_correct:
            correct += 1
        total += 1

        results.append({
            'dialogue_id': sample['dialogue_id'],
            'turn': sample['turn'],
            'mode': sample['mode'],
            'ground_truth': ground_truth,
            'prediction': pred_num,
            'correct': is_correct,
        })

    accuracy = correct / total if total > 0 else 0
    print(f"\nAccuracy: {correct}/{total} = {accuracy:.2%}")

    # 결과 저장
    adapter_name = os.path.basename(adapter_path)
    output_file = os.path.join(output_dir, f'sft_eval_{adapter_name}_{classification_type}.json')
    eval_results = {
        'metadata': {
            'adapter_path': adapter_path,
            'classification_type': classification_type,
            'eval_type': 'swift_infer',
        },
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'results': results
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=2)

    print(f"Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='SFT Model Evaluation')
    parser.add_argument('--adapter_path', type=str, default=None,
                        help='Path to LoRA adapter (for ms-swift infer)')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Model name (for vLLM evaluation)')
    parser.add_argument('--port', type=int, default=None,
                        help='vLLM port (for vLLM evaluation)')
    parser.add_argument('--classification_type', type=str, default='fourclass',
                        choices=['binary', 'fourclass', 'fourclass_v2', 'multiclass'],
                        help='Classification type (default: fourclass)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Preprocessed data directory')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--sample', type=int, default=None,
                        help='Number of samples (for vLLM eval)')

    args = parser.parse_args()

    if args.adapter_path:
        evaluate_with_swift_infer(
            adapter_path=args.adapter_path,
            classification_type=args.classification_type,
            data_dir=args.data_dir,
            output_dir=args.output_dir
        )
    elif args.model_name and args.port:
        evaluate_with_vllm(
            model_name=args.model_name,
            port=args.port,
            classification_type=args.classification_type,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            sample_size=args.sample
        )
    else:
        print("Error: Provide either --adapter_path (for swift infer) or --model_name + --port (for vLLM)")
        parser.print_help()


if __name__ == '__main__':
    main()
