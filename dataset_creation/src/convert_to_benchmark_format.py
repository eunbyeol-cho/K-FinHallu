"""
inject_hallucinations.py 출력을 benchmark 최종 포맷으로 변환

변환:
- {"conversations": [...]} → [...] (list 형태)
- total_conversations 등 메타 키 제거
- 원본 대화도 같은 포맷으로 변환 가능
"""

import json
import argparse
from collections import Counter


def convert_to_benchmark_format(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "conversations" in data:
        conversations = data["conversations"]
    elif isinstance(data, list):
        conversations = data
    else:
        raise ValueError(f"Unknown format: {type(data)}, keys={list(data.keys())[:5]}")

    # 대화 레벨 불필요 키 제거
    remove_conv_keys = {"average_evaluations", "total_conversations"}
    # 턴 레벨 불필요 키 제거 (평가 단계 부산물)
    remove_turn_keys = {"evaluations", "is_correct", "hallucination_evaluations", "evidence_text"}
    cleaned = []
    for conv in conversations:
        cleaned_conv = {k: v for k, v in conv.items() if k not in remove_conv_keys}
        if "dialogue" in cleaned_conv:
            cleaned_conv["dialogue"] = [
                {k: v for k, v in turn.items() if k not in remove_turn_keys}
                for turn in cleaned_conv["dialogue"]
            ]
        cleaned.append(cleaned_conv)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    print(f"입력: {input_path} ({len(conversations)}개 대화)")
    print(f"출력: {output_path} ({len(cleaned)}개 대화, list 형태)")

    # hallucination 통계 (있는 경우)
    types = Counter()
    for conv in cleaned:
        for t in conv.get("dialogue", []):
            ht = t.get("hallucination_type")
            if ht is not None:
                types[ht] += 1
    if types:
        print(f"\nhallucination_type 분포:")
        for ht, cnt in types.most_common():
            print(f"  {ht}: {cnt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark 포맷 변환")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    convert_to_benchmark_format(args.input, args.output)
