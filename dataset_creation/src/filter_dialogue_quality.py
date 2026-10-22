"""
Train 데이터 필터링 스크립트

eval_step1_by_llm의 평가 결과를 기반으로 low-score 대화를 제외합니다.
기준: any_min_relaxed_correctness (test와 동일)
- 각 메트릭에서 최저점이 하나라도 있는 대화 제외
- 단, correctness는 최저점 턴이 2개 이상일 때만 제외 (1번 실수 허용)
"""

import json
import argparse
from pathlib import Path


MIN_VALUE = {
    "answerability": 0,
    "plausibility": 1,
    "correctness": 0,
    "diversity": 1,
    "coherency": 1,
}
TURN_METRICS = ["answerability", "plausibility", "correctness"]
DIALOGUE_METRICS = ["diversity", "coherency"]


def find_bad_dialogue_ids(evaluated_data):
    """평가 결과에서 low-score 대화 ID를 추출한다."""
    convs = evaluated_data["conversations"]
    dialogue_has_min = []
    correctness_min_turns = {}

    for conv in convs:
        did = conv.get("dialogue_id")
        avg_eval = conv.get("average_evaluations", {}) or {}
        turns = conv.get("dialogue", []) or []
        flags = {}

        for m in DIALOGUE_METRICS:
            flags[m] = avg_eval.get(m) == MIN_VALUE[m]

        for m in TURN_METRICS:
            mv = MIN_VALUE[m]
            has_min = False
            min_count = 0
            for t in turns:
                ev = t.get("evaluations", {}) or {}
                v = ev.get(m)
                if v is not None and v == mv:
                    has_min = True
                    if m == "correctness":
                        min_count += 1
            flags[m] = has_min
            if m == "correctness":
                correctness_min_turns[did] = min_count

        dialogue_has_min.append(flags)

    bad_ids = set()
    for i, f in enumerate(dialogue_has_min):
        did = convs[i].get("dialogue_id")
        flags_relaxed = dict(f)
        flags_relaxed["correctness"] = correctness_min_turns.get(did, 0) >= 2
        if any(flags_relaxed.values()):
            bad_ids.add(did)

    return bad_ids


def filter_conversations(data, good_ids):
    """conversations에서 good_ids에 해당하는 대화만 유지한다."""
    original = data["conversations"]
    filtered = [c for c in original if c.get("dialogue_id") in good_ids]

    # correctness 마킹 (턴별 is_correct 플래그)
    for conv in filtered:
        for turn in conv.get("dialogue", []):
            evaluations = turn.get("evaluations", {})
            assistant_answer = turn.get("assistant_answer", "")
            is_unanswerable = "주어진 정보만으로는 대답 불가능합니다." in assistant_answer

            if evaluations:
                correctness_score = evaluations.get("correctness")
                if correctness_score is not None:
                    turn["is_correct"] = (correctness_score != 0) and (not is_unanswerable)
                else:
                    turn["is_correct"] = False if is_unanswerable else None
            else:
                turn["is_correct"] = False if is_unanswerable else None

    data["conversations"] = filtered
    data["total_conversations"] = len(filtered)
    return data


def main():
    parser = argparse.ArgumentParser(description="Train 데이터 필터링")
    parser.add_argument("--eval-input", type=str, required=True,
                        help="평가 결과 JSON 파일 경로")
    parser.add_argument("--original-input", type=str, required=True,
                        help="원본 대화 JSON 파일 경로")
    parser.add_argument("--created-data-input", type=str, required=True,
                        help="원본 created_data JSON 파일 경로")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="필터링 결과 저장 디렉토리")
    parser.add_argument("--split", type=str, required=True,
                        choices=["train", "test"],
                        help="데이터 split (train 또는 test)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 평가 결과에서 bad dialogue IDs 추출
    print(f"평가 결과 로드: {args.eval_input}")
    with open(args.eval_input, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    bad_ids = find_bad_dialogue_ids(eval_data)
    all_ids = {c.get("dialogue_id") for c in eval_data["conversations"]}
    good_ids = all_ids - bad_ids

    total = len(all_ids)
    print(f"전체: {total}개, 제외: {len(bad_ids)}개, 유지: {len(good_ids)}개")

    # 2. 원본 대화 필터링 (평가 결과 파일에는 evaluations가 포함되어 있으므로 이를 사용)
    print(f"\n대화 데이터 필터링: {args.eval_input}")
    filtered_data = filter_conversations(eval_data, good_ids)

    output_dialogue = output_dir / f"dialogue_results_filtered_{args.split}.json"
    with open(output_dialogue, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)
    print(f"  -> {output_dialogue} ({len(filtered_data['conversations'])}개 대화)")

    # 3. created_data 복사 (필터링 없이 전체 유지)
    print(f"\ncreated_data 복사: {args.created_data_input}")
    with open(args.created_data_input, "r", encoding="utf-8") as f:
        created_data = json.load(f)

    output_created = output_dir / f"created_data_{args.split}.json"
    with open(output_created, "w", encoding="utf-8") as f:
        json.dump(created_data, f, ensure_ascii=False, indent=2)

    qid_count = len(created_data.get("qid_data", {}))
    print(f"  -> {output_created} ({qid_count}개 QID)")

    print(f"\n필터링 완료! 결과: {output_dir}")


if __name__ == "__main__":
    main()
