"""
할루시네이션 평가 결과를 기반으로 저품질 대화를 필터링

- evaluate_hallucination.py 출력(평가 점수)으로 통과/실패 판정
- 원본 주입 데이터(--original)에서 전체 대화(모든 턴)를 가져와서 출력
  (evaluate 출력은 할루시네이션 턴만 포함하므로 원본이 필요)
- 할루시네이션 턴 중 하나라도 기준 미달이면 대화 전체 제거
- 필터링 후 conversations 형태로 저장
"""

import json
import argparse
from collections import Counter


# evaluate_hallucination.py에서 평가를 스킵하는 타입들
SKIP_HALLUCINATION_TYPES = {"true_refusal", "false_acceptance"}


def _load_conversations(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "conversations" in data:
        return data["conversations"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unknown format: {type(data)}")


def passes_threshold(turn: dict, min_appropriateness: int, min_naturalness: int) -> bool:
    """턴이 품질 기준을 통과하는지"""
    evals = turn.get("hallucination_evaluations", {})
    app = evals.get("appropriateness", 0)
    nat = evals.get("naturalness", 0)
    return app >= min_appropriateness and nat >= min_naturalness


def filter_conversations(
    eval_path: str,
    original_path: str,
    output_path: str,
    min_appropriateness: int = 2,
    min_naturalness: int = 2,
):
    # 1) 평가 결과에서 통과/실패 판정
    eval_convs = _load_conversations(eval_path)
    print(f"평가 결과: {len(eval_convs)}개 대화")
    print(f"기준: appropriateness >= {min_appropriateness}, naturalness >= {min_naturalness}")

    failed_ids = set()
    reason_counter = Counter()

    for conv in eval_convs:
        dialogue_id = conv.get("dialogue_id", "unknown")
        for turn in conv.get("dialogue", []):
            if "hallucination_evaluations" not in turn:
                continue
            if not passes_threshold(turn, min_appropriateness, min_naturalness):
                evals = turn.get("hallucination_evaluations", {})
                app = evals.get("appropriateness", 0)
                nat = evals.get("naturalness", 0)
                if app < min_appropriateness and nat < min_naturalness:
                    reason_counter["both_low"] += 1
                elif app < min_appropriateness:
                    reason_counter["appropriateness_low"] += 1
                else:
                    reason_counter["naturalness_low"] += 1
                failed_ids.add(dialogue_id)
                break

    print(f"실패 대화: {len(failed_ids)}개")

    # 2) 원본 주입 데이터에서 통과한 대화만 추출 (전체 턴 유지)
    original_convs = _load_conversations(original_path)
    print(f"원본 데이터: {len(original_convs)}개 대화")

    kept = [conv for conv in original_convs
            if conv.get("dialogue_id", "unknown") not in failed_ids]

    # 3) 저장
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"conversations": kept, "total_conversations": len(kept)},
                  f, ensure_ascii=False, indent=2)

    print(f"\n결과: {len(kept)}개 유지, {len(failed_ids)}개 제거")
    if reason_counter:
        print("제거 사유:")
        for reason, cnt in reason_counter.most_common():
            print(f"  {reason}: {cnt}")
    print(f"출력: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="평가 기반 저품질 대화 필터링")
    parser.add_argument("--eval", type=str, required=True,
                        help="evaluate_hallucination.py 출력 파일 (평가 점수 포함)")
    parser.add_argument("--original", type=str, required=True,
                        help="원본 주입 데이터 (전체 턴 포함)")
    parser.add_argument("--output", type=str, required=True,
                        help="필터링된 결과 출력 파일")
    parser.add_argument("--min-appropriateness", type=int, default=2,
                        help="최소 적절성 점수 (default: 2)")
    parser.add_argument("--min-naturalness", type=int, default=2,
                        help="최소 자연스러움 점수 (default: 2)")
    args = parser.parse_args()

    filter_conversations(
        eval_path=args.eval,
        original_path=args.original,
        output_path=args.output,
        min_appropriateness=args.min_appropriateness,
        min_naturalness=args.min_naturalness,
    )
