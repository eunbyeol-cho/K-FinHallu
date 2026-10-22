"""
Test a single dialogue sample by ID and turn number (for debugging)

Usage:
    python -m src.inference.run_single --model_name Qwen/Qwen3-14B --port 8000 --dialogue_id 66 --turn 2
"""

import os
import argparse
from src.utils import (
    load_json,
    create_prompt,
    parse_prediction,
    get_ground_truth
)
from src.inference.run_inference import call_llm, detect_backend

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def find_sample(data, dialogue_id, turn):
    """특정 dialogue_id와 turn을 가진 샘플 찾기"""
    for sample in data:
        if sample.get('dialogue_id') == dialogue_id and sample.get('turn') == turn:
            return sample
    return None


def main():
    parser = argparse.ArgumentParser(description='Test single sample by dialogue_id and turn')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Model name (e.g., Qwen/Qwen3-14B)')
    parser.add_argument('--port', type=int, default=8000,
                        help='API port (default: 8000)')
    parser.add_argument('--dialogue_id', type=int, required=True,
                        help='Dialogue ID to test')
    parser.add_argument('--turn', type=int, required=True,
                        help='Turn number to test')
    parser.add_argument('--classification_type', type=str, default='binary',
                        choices=['binary', 'fourclass', 'multiclass'],
                        help='Classification type (default: binary)')
    parser.add_argument('--backend', type=str, default='auto',
                        choices=['auto', 'vllm', 'openai', 'gemini'],
                        help='API backend (default: auto-detect)')

    args = parser.parse_args()

    model_name = args.model_name
    port = args.port
    dialogue_id = args.dialogue_id
    turn = args.turn
    classification_type = args.classification_type
    backend = args.backend if args.backend != 'auto' else detect_backend(model_name)

    # JSON 파일 로드
    data_dir = os.path.join(PROJECT_ROOT, 'data', 'preprocessed')
    print("Loading data...")
    hallucination_data = load_json(os.path.join(data_dir, 'preprocessed_hallucination_mode.json'))
    faithful_data = load_json(os.path.join(data_dir, 'preprocessed_faithful_mode.json'))

    print(f"Loaded {len(hallucination_data)} hallucination mode samples")
    print(f"Loaded {len(faithful_data)} faithful mode samples")

    # 특정 샘플 찾기 (faithful_data에서 먼저 찾기)
    sample = find_sample(faithful_data, dialogue_id, turn)
    if sample is None:
        sample = find_sample(hallucination_data, dialogue_id, turn)

    if sample is None:
        print(f"\nError: Sample with dialogue_id={dialogue_id}, turn={turn} not found!")
        return

    print(f"\n{'='*80}")
    print(f"Testing single sample: dialogue_id={dialogue_id}, turn={turn}")
    print(f"Mode: {sample['mode']}")
    print(f"Hallucination Type: {sample.get('hallucination_type', 'N/A')}")
    print('='*80)

    # 프롬프트 생성
    prompt = create_prompt(sample, classification_type)

    print("\n[Prompt]")
    print(prompt)
    print("\n" + "="*80)

    # LLM 호출
    print("\n[Calling LLM...]")
    prediction_raw = call_llm(prompt, model_name, backend, port)

    if prediction_raw is None:
        print("Error: Failed to get prediction from LLM")
        return

    print(f"\n[Raw Prediction]")
    print(prediction_raw)
    print("\n" + "="*80)

    # 예측 결과 파싱
    reasoning, pred_num = parse_prediction(prediction_raw, classification_type)

    # Ground truth
    ground_truth = get_ground_truth(sample, classification_type)

    # 결과 출력
    print(f"\n[Results]")
    print(f"Dialogue ID: {dialogue_id}")
    print(f"Turn: {turn}")
    print(f"Mode: {sample['mode']}")
    if sample['mode'] == 'faithful':
        print(f"Ground Truth: {ground_truth} (No Hallucination - faithful mode)")
    else:
        if classification_type == "binary":
            print(f"Ground Truth: {ground_truth} (Hallucination - {sample['hallucination_type']})")
        else:
            print(f"Ground Truth: {ground_truth} ({sample['hallucination_type']})")
    print(f"Prediction: {pred_num}")
    if reasoning:
        print(f"\n[Reasoning]")
        print(reasoning)

    correct = pred_num is not None and pred_num == ground_truth
    print(f"\nCorrect: {correct}")


if __name__ == '__main__':
    main()
