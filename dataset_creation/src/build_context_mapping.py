"""
processed_data의 context_mapping 전체를 추출하여 context_mapping JSON 생성

- data/processed_data_{split}.json의 context_mapping에서 모든 항목 추출
- q_id 필드 제거
- 출력: {context_id: {context, doc_id, doc_source, doc_class_class, doc_class_code}}
"""

import json
import argparse


def build_context_mapping(
    source_path: str,
    output_path: str,
):
    # 1) 원본 context_mapping 로드
    with open(source_path, "r", encoding="utf-8") as f:
        source_data = json.load(f)

    source_mapping = source_data["context_mapping"]
    print(f"원본 context_mapping 항목 수: {len(source_mapping)}")

    # 2) 전체 context_mapping 추출, q_id 제거
    keep_fields = {"context", "doc_id", "doc_source", "doc_class_class", "doc_class_code"}
    result = {}

    # processed_data 순서를 그대로 유지한다 (논문 데이터의 context_mapping 파일과 동일한 순서).
    for cid in source_mapping:
        entry = source_mapping[cid]
        result[cid] = {k: v for k, v in entry.items() if k in keep_fields}

    print(f"매핑 완료: {len(result)}개 context_id")

    # 3) 저장
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"출력: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="context_mapping JSON 생성")
    parser.add_argument(
        "--source", type=str, required=True,
        help="원본 데이터 (context_mapping이 포함된 processed_data JSON)",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="출력 파일 경로",
    )
    args = parser.parse_args()
    build_context_mapping(args.source, args.output)
