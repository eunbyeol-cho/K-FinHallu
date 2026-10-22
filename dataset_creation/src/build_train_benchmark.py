#!/usr/bin/env python3
"""
Train 벤치마크 truncated_with_contexts 생성 스크립트

hallucination_filtered_dialogue_results_train.json을 SFT 학습용으로 변환:
1. 환각 0개 대화 제외
2. 첫 번째 환각 턴까지 truncate
3. 불필요 키 제거 (evaluations, is_correct, average_evaluations)
4. positive_context_ids 추가 (EmbeddingSearcher, top-10)
5. false_refusal/false_acceptance/true_refusal → negative_context_ids 추가 (NoAnswerPassageRetriever)
6. 나머지 유형 → negative_context_ids: []

출력: list 형태 JSON
"""

import sys
import os
import json
import random
import argparse
import threading
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# src 디렉토리를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from add_context_ids import EmbeddingSearcher
from retrieve_hardnegative_passages import NoAnswerPassageRetriever
from qa_model import QAModel


def load_json(filepath: str) -> Any:
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# 환각 턴에서 negative_context_ids를 필요로 하는 유형
NEGATIVE_CONTEXT_TYPES = {'false_refusal', 'false_acceptance', 'true_refusal'}

# 턴 레벨에서 제거할 키
TURN_KEYS_TO_REMOVE = {'evaluations', 'is_correct', 'hallucination_evaluations', 'evidence_text'}

# 대화 레벨에서 제거할 키
DIALOGUE_KEYS_TO_REMOVE = {'average_evaluations'}


def truncate_dialogue(dialogue: Dict) -> Optional[Dict]:
    """
    첫 번째 환각 턴까지 truncate하고, 불필요 키를 제거.
    환각이 없으면 None 반환.
    """
    turns = dialogue.get('dialogue', [])

    # 첫 번째 환각 턴 인덱스 찾기
    hallu_idx = None
    for i, turn in enumerate(turns):
        if turn.get('hallucination_type') is not None:
            hallu_idx = i
            break

    if hallu_idx is None:
        return None  # 환각 없음 → 제외

    # 첫 번째 환각 턴까지 잘라내기 (inclusive)
    truncated_turns = turns[:hallu_idx + 1]

    # 턴 레벨 키 제거
    cleaned_turns = []
    for turn in truncated_turns:
        cleaned = {k: v for k, v in turn.items() if k not in TURN_KEYS_TO_REMOVE}
        cleaned_turns.append(cleaned)

    # 대화 레벨 키 제거
    result = {k: v for k, v in dialogue.items() if k not in DIALOGUE_KEYS_TO_REMOVE}
    result['dialogue'] = cleaned_turns
    result['num_turns'] = len(cleaned_turns)

    return result


def build_context_mapping_and_qid_data(
    processed_data_path: str,
    created_data_path: Optional[str] = None
) -> tuple:
    """processed_data와 created_data를 로드·병합하여 (qid_data, context_mapping) 반환"""
    data = load_json(processed_data_path)
    qid_data = data.get('qid_data', {})
    context_mapping = data.get('context_mapping', {})

    if created_data_path:
        try:
            created = load_json(created_data_path)

            if isinstance(created, list):
                created_qid_data = {}
                created_context_mapping = {}
                for item in created:
                    qid = item.get('qid')
                    if qid:
                        created_qid_data[qid] = {
                            'question': item.get('question', ''),
                            'answer': item.get('answer', ''),
                            'clue_text': item.get('clue_text', ''),
                            'context_id': item.get('context_id', ''),
                            'doc_id': item.get('doc_id', '')
                        }
                        ctx_id = item.get('context_id')
                        if ctx_id and ctx_id not in created_context_mapping:
                            created_context_mapping[ctx_id] = {
                                'context_id': ctx_id,
                                'doc_id': item.get('doc_id', ''),
                                'q_id': []
                            }
                        if ctx_id and qid:
                            created_context_mapping[ctx_id]['q_id'].append(qid)
            else:
                created_qid_data = created.get('qid_data', {})
                created_context_mapping = created.get('context_mapping', {})

            qid_data.update(created_qid_data)
            for ctx_id, ctx_info in created_context_mapping.items():
                if ctx_id in context_mapping:
                    context_mapping[ctx_id].update(ctx_info)
                    context_mapping[ctx_id]['q_id'] = list(set(
                        context_mapping[ctx_id].get('q_id', []) +
                        ctx_info.get('q_id', [])
                    ))
                else:
                    context_mapping[ctx_id] = ctx_info

            print(f"  Merged created_data: {len(created_qid_data)} questions, {len(created_context_mapping)} contexts")
        except Exception as e:
            print(f"  Warning: Could not load created_data: {e}")

    return qid_data, context_mapping


def build_original_dialogue_mapping(original_dialogue_path: str) -> tuple:
    """original dialogue에서 (dialogue_id, turn) → (qid, original_question) 매핑 생성"""
    data = load_json(original_dialogue_path)
    convs = data if isinstance(data, list) else data.get('conversations', [])

    dialogue_turn_to_qid = {}
    dialogue_turn_to_original_question = {}

    for conv in convs:
        dialogue_id = conv.get('dialogue_id')
        for turn in conv.get('dialogue', []):
            turn_idx = turn.get('turn')
            qid = turn.get('qid')
            original_question = turn.get('original_question')
            if dialogue_id is not None and turn_idx is not None and qid:
                key = (dialogue_id, turn_idx)
                dialogue_turn_to_qid[key] = qid
                if original_question:
                    dialogue_turn_to_original_question[key] = original_question

    return dialogue_turn_to_qid, dialogue_turn_to_original_question


def add_positive_context_ids(
    dialogues: List[Dict],
    embedder: EmbeddingSearcher
) -> None:
    """모든 환각 턴(마지막 턴)에 positive_context_ids 추가 (in-place)"""
    stats = {'total': 0, 'with_positive': 0, 'already_included': 0, 'added': 0}
    position_stats = {i: 0 for i in range(1, 11)}

    for dialogue in tqdm(dialogues, desc="Adding positive_context_ids"):
        # 마지막 턴이 환각 턴
        hallu_turn = dialogue['dialogue'][-1]
        stats['total'] += 1

        original_question = hallu_turn.get('original_question', '')
        current_context_id = None
        if hallu_turn.get('passages') and len(hallu_turn['passages']) > 0:
            current_context_id = hallu_turn['passages'][0].get('context_id')

        positive_context_ids = []

        if original_question:
            similar_context_ids = embedder.find_similar_contexts(original_question, top_k=10)

            if similar_context_ids and current_context_id:
                positive_context_ids = similar_context_ids.copy()

                if current_context_id in positive_context_ids:
                    stats['already_included'] += 1
                    positive_context_ids = positive_context_ids[:10]
                else:
                    stats['added'] += 1
                    if len(positive_context_ids) >= 10:
                        positive_context_ids = positive_context_ids[:9]
                    positive_context_ids.append(current_context_id)

                random.shuffle(positive_context_ids)

                pos = positive_context_ids.index(current_context_id) + 1
                if pos <= 10:
                    position_stats[pos] += 1
            elif current_context_id:
                positive_context_ids = [current_context_id]
                position_stats[1] += 1
        else:
            if current_context_id:
                positive_context_ids = [current_context_id]
                position_stats[1] += 1

        if positive_context_ids:
            stats['with_positive'] += 1

        hallu_turn['positive_context_ids'] = positive_context_ids

    print(f"\n=== Positive Context IDs ===")
    print(f"  Total hallucination turns: {stats['total']}")
    print(f"  With positive_context_ids: {stats['with_positive']}")
    print(f"  Already included in embedding results: {stats['already_included']}")
    print(f"  Newly appended: {stats['added']}")
    print(f"  Position distribution (after shuffle):")
    for pos in range(1, 11):
        print(f"    {pos}: {position_stats[pos]}")


def _process_single_turn(
    retriever: NoAnswerPassageRetriever,
    info: Dict,
    context_mapping: Dict,
    num_workers: int,
    target_count: int,
    backup_count: int,
    retrieve_top_k: int,
    verbose: bool
) -> int:
    """단일 턴의 hard negative 검색. 찾은 negative 개수를 반환."""
    try:
        result = retriever.find_no_answer_passages(
            question=info['original_question'],
            answer=info['answer'],
            clue=info['clue'],
            context_mapping=context_mapping,
            target_count=target_count,
            backup_count=backup_count,
            retrieve_top_k=retrieve_top_k,
            num_workers=num_workers,
            verbose=verbose,
            exclude_context_ids=info.get('exclude_context_ids')
        )
        no_answer_passages = result.get('no_answer_passages', [])
        neg_ids = [p['context_id'] for p in no_answer_passages[:target_count]]
        info['hallu_turn']['negative_context_ids'] = neg_ids
        return len(neg_ids)
    except Exception as e:
        key_str = f"d{info['dialogue_id']}_t{info['turn_idx']}_{info['qid']}"
        print(f"    Error for {key_str}: {e}")
        info['hallu_turn']['negative_context_ids'] = []
        return 0


def add_negative_context_ids(
    dialogues: List[Dict],
    retriever: NoAnswerPassageRetriever,
    qid_data: Dict,
    context_mapping: Dict,
    dialogue_turn_to_qid: Dict,
    dialogue_turn_to_original_question: Dict,
    num_workers: int = 1,
    num_turn_workers: int = 1,
    target_count: int = 10,
    backup_count: int = 2,
    retrieve_top_k: int = 100,
    max_neg_turns: Optional[int] = None
) -> None:
    """false_refusal/false_acceptance/true_refusal 환각 턴에 negative_context_ids 추가 (in-place)"""
    # 대상 턴 수집
    target_turns = []
    for dialogue in dialogues:
        hallu_turn = dialogue['dialogue'][-1]
        hallu_type = hallu_turn.get('hallucination_type')

        if hallu_type not in NEGATIVE_CONTEXT_TYPES:
            hallu_turn['negative_context_ids'] = []
            continue

        dialogue_id = dialogue['dialogue_id']
        turn_idx = hallu_turn['turn']
        key = (dialogue_id, turn_idx)

        qid = hallu_turn.get('qid') or dialogue_turn_to_qid.get(key)
        if not qid or qid not in qid_data:
            print(f"  Warning: qid not found for d{dialogue_id}_t{turn_idx}, setting empty negative_context_ids")
            hallu_turn['negative_context_ids'] = []
            continue

        # positive와 동일한 소스 사용: hallu_turn에서 직접 가져오고, 없으면 fallback
        original_question = hallu_turn.get('original_question') \
            or dialogue_turn_to_original_question.get(key) \
            or qid_data[qid].get('question', '')
        answer = hallu_turn.get('original_answer') or qid_data[qid].get('answer', '')
        clue = hallu_turn.get('clue_text') or qid_data[qid].get('clue_text', '')

        if not (original_question and answer and clue):
            print(f"  Warning: missing data for d{dialogue_id}_t{turn_idx} (qid={qid}), setting empty negative_context_ids")
            hallu_turn['negative_context_ids'] = []
            continue

        # 자기 자신 context_id만 제외 (정답이 확실히 있는 passage)
        exclude_ids = set()
        if hallu_turn.get('passages'):
            for p in hallu_turn['passages']:
                ctx_id = p.get('context_id')
                if ctx_id:
                    exclude_ids.add(ctx_id)

        target_turns.append({
            'dialogue': dialogue,
            'hallu_turn': hallu_turn,
            'dialogue_id': dialogue_id,
            'turn_idx': turn_idx,
            'qid': qid,
            'original_question': original_question,
            'answer': answer,
            'clue': clue,
            'hallu_type': hallu_type,
            'exclude_context_ids': exclude_ids,
        })

    if max_neg_turns is not None and max_neg_turns < len(target_turns):
        print(f"\n  [테스트 모드] {len(target_turns)}개 중 {max_neg_turns}개 턴만 처리")
        for info in target_turns[max_neg_turns:]:
            info['hallu_turn']['negative_context_ids'] = []
        target_turns = target_turns[:max_neg_turns]

    print(f"\n=== Negative Context IDs ===")
    print(f"  Turns needing hard negatives: {len(target_turns)}")

    if not target_turns:
        return

    effective_turn_workers = min(num_turn_workers, len(target_turns))

    if effective_turn_workers <= 1:
        # 순차 처리: verbose 출력
        total_found = 0
        for idx, info in enumerate(target_turns, 1):
            key_str = f"d{info['dialogue_id']}_t{info['turn_idx']}_{info['qid']}"
            print(f"\n  [{idx}/{len(target_turns)}] {key_str} ({info['hallu_type']})")
            total_found += _process_single_turn(
                retriever, info, context_mapping,
                num_workers, target_count, backup_count, retrieve_top_k,
                verbose=True
            )
    else:
        # 턴 레벨 병렬 처리
        total_concurrent = effective_turn_workers * num_workers
        print(f"  Turn-level parallelism: {effective_turn_workers} turns concurrent")
        print(f"  Passage-level parallelism: {num_workers} workers per turn")
        print(f"  Max concurrent LLM calls: ~{total_concurrent}")

        total_found = 0
        lock = threading.Lock()

        with ThreadPoolExecutor(max_workers=effective_turn_workers) as executor:
            futures = {
                executor.submit(
                    _process_single_turn,
                    retriever, info, context_mapping,
                    num_workers, target_count, backup_count, retrieve_top_k,
                    verbose=False  # 병렬 시 verbose 끔
                ): info
                for info in target_turns
            }

            with tqdm(total=len(futures), desc="Hard negatives (turns)") as pbar:
                for future in as_completed(futures):
                    info = futures[future]
                    found = future.result()
                    with lock:
                        total_found += found
                    pbar.update(1)
                    pbar.set_postfix(found=total_found)

    avg = total_found / len(target_turns) if target_turns else 0
    print(f"\n  Total negative passages found: {total_found}")
    print(f"  Average per turn: {avg:.1f}")


def verify_output(dialogues: List[Dict]) -> None:
    """출력 검증"""
    print(f"\n=== Verification ===")
    print(f"  Total dialogues: {len(dialogues)}")

    issues = []
    hallu_count = 0
    pos_count = 0
    neg_count = 0
    neg_needed = 0

    for d in dialogues:
        turns = d['dialogue']
        last_turn = turns[-1]

        # 마지막 턴이 환각 턴인지
        if last_turn.get('hallucination_type') is None:
            issues.append(f"  d{d['dialogue_id']}: last turn has no hallucination_type")
        else:
            hallu_count += 1

        # positive_context_ids 존재 확인
        pos = last_turn.get('positive_context_ids')
        if pos and len(pos) > 0:
            pos_count += 1
        else:
            issues.append(f"  d{d['dialogue_id']}: missing positive_context_ids")

        # negative_context_ids 확인
        hallu_type = last_turn.get('hallucination_type')
        if hallu_type in NEGATIVE_CONTEXT_TYPES:
            neg_needed += 1
            neg = last_turn.get('negative_context_ids', [])
            if neg and len(neg) > 0:
                neg_count += 1

        # 불필요 키 잔존 확인
        for key in TURN_KEYS_TO_REMOVE:
            if key in last_turn:
                issues.append(f"  d{d['dialogue_id']}: turn still has key '{key}'")

        for key in DIALOGUE_KEYS_TO_REMOVE:
            if key in d:
                issues.append(f"  d{d['dialogue_id']}: dialogue still has key '{key}'")

    print(f"  Dialogues ending with hallucination: {hallu_count}/{len(dialogues)}")
    print(f"  With positive_context_ids: {pos_count}/{len(dialogues)}")
    print(f"  Needing negative_context_ids: {neg_needed}")
    print(f"  With negative_context_ids: {neg_count}/{neg_needed}")

    if issues:
        print(f"\n  Issues ({len(issues)}):")
        for issue in issues[:10]:
            print(f"    {issue}")
        if len(issues) > 10:
            print(f"    ... and {len(issues) - 10} more")
    else:
        print(f"  All checks passed!")


def main():
    parser = argparse.ArgumentParser(
        description="Train 벤치마크 truncated_with_contexts 생성"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="hallucination_filtered_dialogue_results_train.json")
    parser.add_argument("--original-dialogue", type=str, required=True,
                        help="dialogue_results_train.json (original_question, qid 참조)")
    parser.add_argument("--processed-data", type=str, required=True,
                        help="processed_data_train.json (context_mapping)")
    parser.add_argument("--created-data", type=str, default=None,
                        help="created_data_train.json (추가 QID 데이터)")
    parser.add_argument("--output", type=str, required=True,
                        help="출력 파일 경로")
    parser.add_argument("--embedding-cache-dir", type=str, default="vector_db_cache/train",
                        help="임베딩 캐시 디렉토리")
    parser.add_argument("--embedding-model", type=str, default="nlpai-lab/KURE-v1",
                        help="임베딩 모델")
    parser.add_argument("--provider", type=str, default="openai",
                        help="LLM provider (hard negative용)")
    parser.add_argument("--model", type=str, default="gpt-5-mini",
                        help="LLM model (hard negative용)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="passage 레벨 병렬 워커 수 (턴 내 LLM 호출)")
    parser.add_argument("--num-turn-workers", type=int, default=1,
                        help="턴 레벨 병렬 워커 수 (동시 처리 턴 수)")
    parser.add_argument("--target-count", type=int, default=10,
                        help="목표 negative passage 개수")
    parser.add_argument("--backup-count", type=int, default=2,
                        help="백업 negative passage 개수")
    parser.add_argument("--retrieve-top-k", type=int, default=150,
                        help="FAISS retrieve 개수")
    parser.add_argument("--cache-path", type=str, default=None,
                        help="LLM 판정 캐시 파일 경로 (재실행 시 비용 절감)")
    parser.add_argument("--max-neg-turns", type=int, default=None,
                        help="테스트용: negative 처리할 최대 턴 수")
    parser.add_argument("--seed", type=int, default=42,
                        help="positive_context_ids 셔플 시드 (재실행 시 같은 순서 보장)")

    args = parser.parse_args()
    random.seed(args.seed)

    # ── Step 1: 데이터 로드 ──
    print("=" * 60)
    print("Step 1: Loading data")
    print("=" * 60)

    hallu_data = load_json(args.input)
    conversations = hallu_data if isinstance(hallu_data, list) else hallu_data.get('conversations', [])
    print(f"  Input conversations: {len(conversations)}")

    # ── Step 2: Truncate ──
    print("\n" + "=" * 60)
    print("Step 2: Truncating dialogues to first hallucination turn")
    print("=" * 60)

    truncated = []
    skipped = 0
    for conv in conversations:
        result = truncate_dialogue(conv)
        if result is not None:
            truncated.append(result)
        else:
            skipped += 1

    print(f"  Truncated: {len(truncated)} dialogues")
    print(f"  Skipped (no hallucination): {skipped}")

    # ── Step 3: positive_context_ids 추가 ──
    print("\n" + "=" * 60)
    print("Step 3: Adding positive_context_ids")
    print("=" * 60)

    embedder = EmbeddingSearcher(
        embedding_model=args.embedding_model,
        cache_dir=args.embedding_cache_dir
    )
    add_positive_context_ids(truncated, embedder)

    # ── Step 4: negative_context_ids 추가 ──
    print("\n" + "=" * 60)
    print("Step 4: Adding negative_context_ids (hard negatives)")
    print("=" * 60)

    print("  Loading auxiliary data...")
    qid_data, context_mapping = build_context_mapping_and_qid_data(
        args.processed_data, args.created_data
    )
    print(f"  qid_data: {len(qid_data)} entries")
    print(f"  context_mapping: {len(context_mapping)} entries")

    dialogue_turn_to_qid, dialogue_turn_to_original_question = \
        build_original_dialogue_mapping(args.original_dialogue)
    print(f"  Original dialogue mapping: {len(dialogue_turn_to_qid)} (dialogue_id, turn) pairs")

    qa_model = QAModel(provider=args.provider, model=args.model, api_key=args.api_key)
    retriever = NoAnswerPassageRetriever(
        qa_model,
        embedding_model=args.embedding_model,
        embedding_cache_dir=args.embedding_cache_dir,
        classification_cache_path=args.cache_path
    )

    add_negative_context_ids(
        truncated,
        retriever,
        qid_data,
        context_mapping,
        dialogue_turn_to_qid,
        dialogue_turn_to_original_question,
        num_workers=args.num_workers,
        num_turn_workers=args.num_turn_workers,
        target_count=args.target_count,
        backup_count=args.backup_count,
        retrieve_top_k=args.retrieve_top_k,
        max_neg_turns=args.max_neg_turns
    )

    # ── Step 5: 검증 및 저장 ──
    print("\n" + "=" * 60)
    print("Step 5: Verification & Save")
    print("=" * 60)

    verify_output(truncated)

    print(f"\n  Saving to {args.output}...")
    save_json(truncated, args.output)
    print(f"  Done! {len(truncated)} dialogues saved.")


if __name__ == "__main__":
    main()
