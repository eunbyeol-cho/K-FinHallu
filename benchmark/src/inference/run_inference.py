"""
Unified Hallucination Detection Inference Script
Supports: Local vLLM, OpenAI GPT, Google Gemini
Classification: Binary, Fourclass, or Multiclass

Usage:
    python -m src.inference.run_inference --model_name gemini-2.5-pro --classification_type fourclass
    python -m src.inference.run_inference --model_name Qwen/Qwen3-8B --port 8000 --classification_type fourclass
"""

import json
import os
import argparse
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai

from src.utils import (
    load_json,
    create_prompt,
    parse_prediction,
    get_ground_truth,
    sanitize_filename
)

load_dotenv()

# Project root directory (benchmark/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO_ROOT = os.path.dirname(PROJECT_ROOT)


def detect_backend(model_name):
    """모델 이름 기반으로 API 백엔드 자동 감지"""
    if model_name.lower().startswith('gemini'):
        return 'gemini'
    elif model_name.lower().startswith('gpt'):
        return 'openai'
    else:
        return 'vllm'


def call_llm(prompt, model_name, backend, port=None, api_key=None, temperature=0.0):
    """
    통합 LLM API 호출
    - gemini: Google Gemini API
    - openai: OpenAI GPT API
    - vllm: Local vLLM (OpenAI compatible)
    """
    try:
        if backend == 'gemini':
            if not api_key:
                api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not set")

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                )
            )
            return response.text.strip()

        elif backend == 'openai':
            if not api_key:
                api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")

            client = OpenAI(api_key=api_key)
            # GPT-5는 temperature 파라미터를 받지 않음
            if model_name.lower().startswith('gpt-5'):
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}]
                )
            else:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature
                )
            return response.choices[0].message.content.strip()

        else:  # vllm
            if not port:
                raise ValueError("Port is required for local vLLM models")

            client = OpenAI(
                api_key="dummy-key",
                base_url=f"http://localhost:{port}/v1"
            )
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            msg = response.choices[0].message
            content = msg.content or ""
            # vLLM이 Qwen3 think 토큰을 reasoning_content로 분리하는 경우 처리
            reasoning = getattr(msg, 'reasoning_content', None) or ""
            if reasoning and content:
                return f"<think>\n{reasoning}\n</think>\n\n{content}".strip()
            elif reasoning:
                return f"<think>\n{reasoning}\n</think>".strip()
            return content.strip()

    except Exception as e:
        print(f"Error calling LLM: {e}")
        return None


def evaluate_samples(samples, mode_name, model_name, classification_type, backend,
                     port=None, api_key=None, checkpoint_file=None, save_interval=100):
    """샘플들을 평가하고 결과 반환"""
    results = []
    correct = 0
    total = 0
    start_idx = 0

    # Binary classification용 confusion matrix
    if classification_type == "binary":
        tp = tn = fp = fn = 0

    # 기존 체크포인트가 있으면 로드
    if checkpoint_file:
        try:
            checkpoint_data = load_json(checkpoint_file)
            if 'results' in checkpoint_data and checkpoint_data['results']:
                results = checkpoint_data['results']
                start_idx = len(results)
                # 정확도 재계산
                for r in results:
                    if r.get('correct', False):
                        correct += 1
                        if classification_type == "binary":
                            gt = r.get('ground_truth', 0)
                            if gt == 1:
                                tp += 1
                            else:
                                tn += 1
                    else:
                        if classification_type == "binary":
                            gt = r.get('ground_truth', 0)
                            pred = r.get('prediction', 0)
                            if pred == 1 and gt == 0:
                                fp += 1
                            elif pred == 0 and gt == 1:
                                fn += 1
                    total += 1
                print(f"Loaded checkpoint: {start_idx} samples already processed")
        except FileNotFoundError:
            print(f"No checkpoint found, starting from scratch")
        except Exception as e:
            print(f"Error loading checkpoint: {e}, starting from scratch")

    print(f"\n{'='*80}")
    print(f"Evaluating {mode_name} ({len(samples)} samples, starting from {start_idx})")
    print('='*80)

    for i, sample in enumerate(tqdm(samples[start_idx:], desc=f"Processing {mode_name}", initial=start_idx, total=len(samples))):
        actual_idx = start_idx + i

        # 프롬프트 생성
        prompt = create_prompt(sample, classification_type)

        # LLM 호출
        prediction_raw = call_llm(prompt, model_name, backend, port, api_key)

        # 예측 결과 파싱
        reasoning, pred_num = parse_prediction(prediction_raw, classification_type)

        # Ground truth
        ground_truth = get_ground_truth(sample, classification_type)

        # 결과 저장
        result = {
            'dialogue_id': sample['dialogue_id'],
            'turn': sample['turn'],
            'mode': sample['mode'],
            'hallucination_type': sample['hallucination_type'],
            'ground_truth': ground_truth,
            'prediction_raw': prediction_raw,
            'reasoning': reasoning,
            'prediction': pred_num,
            'correct': False
        }
        if sample.get('injection_model'):
            result['injection_model'] = sample['injection_model']

        # 정확도 계산
        if pred_num is not None:
            if pred_num == ground_truth:
                correct += 1
                result['correct'] = True
                if classification_type == "binary":
                    if ground_truth == 1:
                        tp += 1
                    else:
                        tn += 1
            else:
                if classification_type == "binary":
                    if pred_num == 1 and ground_truth == 0:
                        fp += 1
                    elif pred_num == 0 and ground_truth == 1:
                        fn += 1

        total += 1
        results.append(result)

        # 처음 몇 개 샘플 출력
        if actual_idx < 3:
            print(f"\n[Sample {actual_idx+1}]")
            print(f"Dialogue ID: {sample['dialogue_id']}")
            print(f"Mode: {sample['mode']}")
            if sample['mode'] == 'faithful':
                if classification_type == "binary":
                    print(f"Ground Truth: {ground_truth} (No Hallucination - faithful mode)")
                elif classification_type == "fourclass":
                    gt_label = "Faithful Answer" if ground_truth == 0 else "True Refusal"
                    print(f"Ground Truth: {ground_truth} ({gt_label} - {sample['hallucination_type']})")
                else:
                    print(f"Ground Truth: {ground_truth} (No Hallucination - faithful mode)")
            else:
                if classification_type == "binary":
                    print(f"Ground Truth: {ground_truth} (Hallucination - {sample['hallucination_type']})")
                elif classification_type == "fourclass":
                    gt_labels = {0: "Faithful Answer", 1: "Hallucination", 2: "False Refusal", 3: "True Refusal"}
                    print(f"Ground Truth: {ground_truth} ({gt_labels.get(ground_truth, 'Unknown')} - {sample['hallucination_type']})")
                else:
                    print(f"Ground Truth: {ground_truth} ({sample['hallucination_type']})")
            print(f"Prediction: {pred_num}")
            if reasoning:
                print(f"Reasoning (first 100 chars): {reasoning[:100]}...")
            print(f"Correct: {result['correct']}")

        # 100개마다 중간 저장
        if checkpoint_file and (actual_idx + 1) % save_interval == 0:
            accuracy = correct / total if total > 0 else 0
            checkpoint_data = {
                'results': results,
                'accuracy': accuracy,
                'processed': actual_idx + 1,
                'total': len(samples)
            }
            if classification_type == "binary":
                checkpoint_data['tp'] = tp
                checkpoint_data['tn'] = tn
                checkpoint_data['fp'] = fp
                checkpoint_data['fn'] = fn

            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
            print(f"\n[Checkpoint saved: {actual_idx + 1}/{len(samples)} samples, accuracy: {accuracy:.2%}]")

    accuracy = correct / total if total > 0 else 0

    if classification_type == "binary":
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"\n{mode_name} Results:")
        print(f"  Accuracy: {correct}/{total} = {accuracy:.2%}")
        print(f"  Precision: {precision:.2%}")
        print(f"  Recall: {recall:.2%}")
        print(f"  F1 Score: {f1:.2%}")
        print(f"  TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

        return results, {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn
        }
    else:
        print(f"\n{mode_name} Accuracy: {correct}/{total} = {accuracy:.2%}")
        return results, accuracy


def main():
    parser = argparse.ArgumentParser(description='Unified Hallucination Detection Inference')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Model name (gemini-*, gpt-*, or local model name)')
    parser.add_argument('--classification_type', type=str, default='fourclass',
                        choices=['binary', 'fourclass', 'fourclass_v2', 'multiclass'],
                        help='Classification type (default: fourclass)')
    parser.add_argument('--backend', type=str, default='auto',
                        choices=['auto', 'vllm', 'openai', 'gemini'],
                        help='API backend (default: auto-detect from model name)')
    parser.add_argument('--port', type=int, default=None,
                        help='API port (required for local vLLM models)')
    parser.add_argument('--api_key', type=str, default=None,
                        help='API key (for GPT/Gemini, or use env var)')
    parser.add_argument('--sample', type=int, default=None,
                        help='Number of samples to test (default: all)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Data directory (default: <repo_root>/data/preprocessed)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: <repo_root>/results/inference)')

    args = parser.parse_args()

    model_name = args.model_name
    classification_type = args.classification_type
    port = args.port
    api_key = args.api_key
    sample_size = args.sample

    # 백엔드 결정
    backend = args.backend if args.backend != 'auto' else detect_backend(model_name)

    if backend == 'vllm' and not port:
        print("Error: --port is required for local vLLM models")
        return

    # 경로 설정
    data_dir = args.data_dir or os.path.join(REPO_ROOT, 'data', 'preprocessed')
    output_dir = args.output_dir or os.path.join(REPO_ROOT, 'results', 'inference')
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print("Configuration")
    print('='*80)
    print(f"Model: {model_name}")
    print(f"Backend: {backend}")
    if backend == 'vllm':
        print(f"Port: {port}")
    print(f"Classification: {classification_type}")

    # JSON 파일 로드
    print("\nLoading data...")
    hallucination_data = load_json(os.path.join(data_dir, 'preprocessed_hallucination_mode.json'))
    faithful_data = load_json(os.path.join(data_dir, 'preprocessed_faithful_mode.json'))

    print(f"Loaded {len(hallucination_data)} hallucination mode samples")
    print(f"Loaded {len(faithful_data)} faithful mode samples")

    # 샘플 크기 제한
    if sample_size:
        hallucination_data = hallucination_data[:sample_size]
        faithful_data = faithful_data[:sample_size]
        print(f"\nTesting with {sample_size} samples per mode")

    # 모델 이름을 파일명에 사용할 수 있도록 변환
    model_name_safe = sanitize_filename(model_name)

    # 체크포인트 파일명 생성
    hal_checkpoint_file = os.path.join(output_dir, f'checkpoint_hallucination_{model_name_safe}_{classification_type}.json')
    faith_checkpoint_file = os.path.join(output_dir, f'checkpoint_faithful_{model_name_safe}_{classification_type}.json')
    output_file = os.path.join(output_dir, f'test_results_{model_name_safe}_{classification_type}.json')

    # Hallucination mode 평가
    hal_results, hal_metrics = evaluate_samples(
        hallucination_data, "Hallucination Mode", model_name,
        classification_type, backend, port, api_key,
        checkpoint_file=hal_checkpoint_file
    )

    # Faithful mode 평가
    faith_results, faith_metrics = evaluate_samples(
        faithful_data, "Faithful Mode", model_name,
        classification_type, backend, port, api_key,
        checkpoint_file=faith_checkpoint_file
    )

    # 결과 취합
    if classification_type == "binary":
        total_tp = hal_metrics['tp'] + faith_metrics['tp']
        total_tn = hal_metrics['tn'] + faith_metrics['tn']
        total_fp = hal_metrics['fp'] + faith_metrics['fp']
        total_fn = hal_metrics['fn'] + faith_metrics['fn']

        total_correct = total_tp + total_tn
        total_samples = len(hallucination_data) + len(faithful_data)

        overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    else:
        overall_accuracy = (hal_metrics + faith_metrics) / 2

    # 결과 저장
    num_classes_map = {
        'binary': 2,
        'fourclass': 4,
        'fourclass_v2': 4,
        'multiclass': 6
    }

    all_results = {
        'metadata': {
            'model_name': model_name,
            'classification_type': classification_type,
            'num_classes': num_classes_map.get(classification_type, 2),
            'sample_size': sample_size if sample_size else 'all',
            'backend': backend
        },
        'hallucination_mode': {
            'results': hal_results,
            'metrics': hal_metrics
        },
        'faithful_mode': {
            'results': faith_results,
            'metrics': faith_metrics
        }
    }

    if classification_type == "binary":
        all_results['overall'] = {
            'accuracy': overall_accuracy,
            'precision': overall_precision,
            'recall': overall_recall,
            'f1': overall_f1,
            'tp': total_tp,
            'tn': total_tn,
            'fp': total_fp,
            'fn': total_fn
        }
    else:
        all_results['overall_accuracy'] = overall_accuracy

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # 체크포인트 파일 삭제
    for ckpt_file in [hal_checkpoint_file, faith_checkpoint_file]:
        if os.path.exists(ckpt_file):
            os.remove(ckpt_file)
            print(f"Removed checkpoint file: {ckpt_file}")

    # 최종 결과 출력
    print(f"\n{'='*80}")
    print("Final Results")
    print('='*80)
    print(f"Model: {model_name}")
    print(f"Backend: {backend}")
    print(f"Classification Type: {classification_type}")
    if sample_size:
        print(f"Sample Size: {sample_size}")

    if classification_type == "binary":
        print(f"\nHallucination Mode:")
        print(f"  Accuracy: {hal_metrics['accuracy']:.2%}")
        print(f"  Precision: {hal_metrics['precision']:.2%}")
        print(f"  Recall: {hal_metrics['recall']:.2%}")
        print(f"  F1 Score: {hal_metrics['f1']:.2%}")
        print(f"\nFaithful Mode:")
        print(f"  Accuracy: {faith_metrics['accuracy']:.2%}")
        print(f"  Precision: {faith_metrics['precision']:.2%}")
        print(f"  Recall: {faith_metrics['recall']:.2%}")
        print(f"  F1 Score: {faith_metrics['f1']:.2%}")
        print(f"\nOverall:")
        print(f"  Accuracy: {overall_accuracy:.2%}")
        print(f"  Precision: {overall_precision:.2%}")
        print(f"  Recall: {overall_recall:.2%}")
        print(f"  F1 Score: {overall_f1:.2%}")
        print(f"  TP: {total_tp}, TN: {total_tn}, FP: {total_fp}, FN: {total_fn}")
    else:
        print(f"Hallucination Mode Accuracy: {hal_metrics:.2%}")
        print(f"Faithful Mode Accuracy: {faith_metrics:.2%}")
        print(f"Overall Accuracy: {overall_accuracy:.2%}")

    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
