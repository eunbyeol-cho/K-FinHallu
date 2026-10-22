#!/usr/bin/env python3
"""
Main script for RAG-based dialogue simulation
"""

import json
import argparse
import copy
import os
import threading
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from retriever import Retriever, DummyRetriever
from llm_components import (
    QuestionSelector,
    ReconfirmationGenerator,
    QuestionAnswerExtractor,
    QuestionRewriter,
    AnswerGenerator
)
from dialogue_simulator import DialogueSimulator
from preprocess import DataPreprocessor


def load_data(processed_data_path: str, seed_questions_path: str) -> tuple:
    """
    데이터 로드
    
    Returns:
        (qid_data, context_mapping, seed_questions)
    """
    print("📁 Loading processed data...")
    with open(processed_data_path, 'r', encoding='utf-8') as f:
        processed_data = json.load(f)
    
    # qid_data와 context_mapping이 최상위 키인 경우
    if 'qid_data' in processed_data and 'context_mapping' in processed_data:
        qid_data = processed_data.get('qid_data', {})
        context_mapping = processed_data.get('context_mapping', {})
    else:
        # 다른 형식인 경우 변환 (context_id 기반 구조를 qid_data로 변환)
        print("⚠️ Converting data format...")
        qid_data = {}
        context_mapping = {}
        
        for context_id, context_data in processed_data.items():
            context_mapping[context_id] = {
                'context': context_data.get('context', ''),
                'doc_id': context_data.get('doc_id', ''),
                'q_id': []
            }
            
            questions_answers = context_data.get('questions_answers', [])
            for qa in questions_answers:
                question_id = qa.get('question_id')
                if question_id:
                    qid_data[question_id] = {
                        'question': qa.get('question', ''),
                        'answer': qa.get('answer_text', ''),
                        'clue_text': qa.get('clue_text', ''),
                        'context_id': context_id,
                        'similar_context_id': context_data.get('similar_context_ids', []),
                        'doc_id': context_data.get('doc_id', '')
                    }
                    context_mapping[context_id]['q_id'].append(question_id)
    
    print(f"✅ Loaded {len(qid_data)} questions")
    print(f"✅ Loaded {len(context_mapping)} contexts")
    
    print("📁 Loading seed questions...")
    with open(seed_questions_path, 'r', encoding='utf-8') as f:
        seed_questions = json.load(f)
    
    print(f"✅ Loaded {len(seed_questions)} seed questions")
    
    return qid_data, context_mapping, seed_questions


def create_llm_components(
    provider: str = "openai",
    model: str = "gpt-4o-mini"
) -> Dict[str, Any]:
    """LLM 컴포넌트들 생성"""
    print(f"🔧 Initializing LLM components (provider: {provider}, model: {model})...")
    
    components = {
        'question_selector': QuestionSelector(provider=provider, model=model),
        'reconfirmation_generator': ReconfirmationGenerator(provider=provider, model=model),
        'question_answer_extractor': QuestionAnswerExtractor(provider=provider, model=model),
        'question_rewriter': QuestionRewriter(provider=provider, model=model),
        'answer_generator': AnswerGenerator(provider=provider, model=model)
    }
    
    print("✅ LLM components initialized")
    return components


def _process_single_conversation(
    simulator: DialogueSimulator,
    seed_q: Dict[str, Any],
    index: int,
    total: int,
    min_turns_per_conversation: int,
    max_turns_per_conversation: int,
    retrieval_mode: str
) -> Optional[Dict[str, Any]]:
    """
    단일 대화 시뮬레이션 처리 (병렬 처리용)
    
    Args:
        simulator: DialogueSimulator 인스턴스
        seed_q: seed question 딕셔너리
        index: 현재 인덱스
        total: 전체 개수
        min_turns_per_conversation: 대화당 최소 턴 수
        max_turns_per_conversation: 대화당 최대 턴 수
        retrieval_mode: "gold" 또는 "topk"
        
    Returns:
        conversation_record 또는 None (에러 시)
    """
    seed_qid = seed_q.get('question_id')
    seed_question_text = seed_q.get('question', '')
    doc_id = seed_q.get('doc_id', '')
    
    if not seed_qid:
        print(f"⚠️ Skipping seed question {index}: missing question_id")
        return None
    
    print(f"\n[{index}/{total}] Processing seed question: {seed_question_text[:50]}...")
    
    try:
        # 해당 qid의 evidence_text 가져오기
        evidence_text = None
        if seed_qid in simulator.qid_data:
            evidence_text = simulator.qid_data[seed_qid].get('clue_text')
        
        # 대화 시뮬레이션
        dialogue_log, created_questions = simulator.simulate_conversation(
            seed_qid=seed_qid,
            min_turns=min_turns_per_conversation,
            max_turns=max_turns_per_conversation,
            evidence_text=evidence_text,
            retrieval_mode=retrieval_mode
        )
        
        conversation_record = {
            'seed_qid': seed_qid,
            'seed_question': seed_question_text,
            'doc_id': doc_id,
            'dialogue': dialogue_log,
            'num_turns': len(dialogue_log),
            'created_questions': created_questions
        }
        
        print(f"✅ [{index}/{total}] Completed: {len(dialogue_log)} turns")
        return conversation_record
        
    except Exception as e:
        print(f"❌ [{index}/{total}] Error processing seed question {seed_qid}: {e}")
        import traceback
        traceback.print_exc()
        return None


def simulate_conversations(
    simulator: DialogueSimulator,
    seed_questions: List[Dict[str, Any]],
    max_questions: int = None,
    min_turns_per_conversation: int = 4,
    max_turns_per_conversation: int = 8,
    retrieval_mode: str = "gold",
    num_workers: int = 4
) -> List[Dict[str, Any]]:
    """
    여러 대화 시뮬레이션 실행 (병렬 처리)
    
    Args:
        simulator: DialogueSimulator 인스턴스
        seed_questions: seed question 리스트
        max_questions: 최대 처리할 질문 수 (None이면 모두)
        min_turns_per_conversation: 대화당 최소 턴 수
        max_turns_per_conversation: 대화당 최대 턴 수
        retrieval_mode: "gold" 또는 "topk" (기본값: "gold")
        num_workers: 병렬 처리할 워커 수 (기본값: 4)
        
    Returns:
        모든 대화 로그 리스트
    """
    all_conversations = []
    
    num_questions = len(seed_questions) if max_questions is None else min(max_questions, len(seed_questions))
    questions_to_process = seed_questions[:num_questions]
    
    print(f"\n🚀 Starting dialogue simulation for {num_questions} seed questions...")
    print(f"📊 Using {num_workers} workers for parallel processing")
    
    # seed_qid -> index 매핑 생성 (정렬용)
    seed_qid_to_index = {
        q.get('question_id'): i 
        for i, q in enumerate(questions_to_process) 
        if q.get('question_id')
    }
    
    # 병렬 처리
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # 모든 작업 제출
        future_to_index = {
            executor.submit(
                _process_single_conversation,
                simulator,
                seed_q,
                i + 1,
                num_questions,
                min_turns_per_conversation,
                max_turns_per_conversation,
                retrieval_mode
            ): i
            for i, seed_q in enumerate(questions_to_process)
        }
        
        # 완료된 작업 수집
        for future in as_completed(future_to_index):
            result = future.result()
            if result is not None:
                all_conversations.append(result)
    
    # seed_qid 순서대로 정렬 (원래 순서 유지)
    all_conversations.sort(key=lambda x: seed_qid_to_index.get(x.get('seed_qid'), float('inf')))
    
    return all_conversations


def save_results(conversations: List[Dict[str, Any]], output_path: str):
    """
    결과 저장
    
    Args:
        conversations: 모든 대화 기록
        output_path: 저장할 파일 경로
    """
    print(f"\n💾 Saving results to {output_path}...")
    
    # conversations 복사하여 수정 (created_questions 필드 제거 + dialogue_id 추가)
    updated_conversations = []
    
    for dialogue_idx, conv in enumerate(conversations, 1):
        # conversation 깊은 복사
        base_conv = copy.deepcopy(conv)
        
        # created_questions 필드 제거
        if 'created_questions' in base_conv:
            del base_conv['created_questions']
        
        # 대화 단위 고유 ID 추가
        # 기존에 dialogue_id가 있다면 그대로 사용하고, 없으면 1부터 시작하는 연속 ID를 부여
        dialogue_id = base_conv.get('dialogue_id', dialogue_idx)
        
        # JSON에서의 key 순서를 명시적으로 정리
        # (읽기 편하게 하기 위해 dialogue_id를 앞쪽에 배치)
        ordered_keys = [
            ('dialogue_id', dialogue_id),
            ('seed_qid', base_conv.get('seed_qid')),
            ('seed_question', base_conv.get('seed_question')),
            ('doc_id', base_conv.get('doc_id')),
            ('num_turns', base_conv.get('num_turns')),
            ('dialogue', base_conv.get('dialogue')),
        ]
        
        updated_conv: Dict[str, Any] = {}
        for k, v in ordered_keys:
            if v is not None:
                updated_conv[k] = v
        
        # 그 외 남은 필드는 기존 순서대로 뒤에 붙이되, 이미 추가된 키는 제외
        for k, v in base_conv.items():
            if k not in updated_conv:
                updated_conv[k] = v
        
        updated_conversations.append(updated_conv)
    
    output_data = {
        'total_conversations': len(updated_conversations),
        'conversations': updated_conversations
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Saved {len(updated_conversations)} conversations to {output_path}")


def get_max_qid(qid_data: Dict[str, Dict[str, Any]]) -> int:
    """processed_data.json의 최대 qid 찾기"""
    max_qid = 0
    for qid_str in qid_data.keys():
        try:
            qid_num = int(qid_str)
            if qid_num > max_qid:
                max_qid = qid_num
        except ValueError:
            continue
    return max_qid


def save_created_data(
    conversations: List[Dict[str, Any]], 
    created_data_path: str,
    context_mapping: Dict[str, Dict[str, Any]],
    qid_data: Dict[str, Dict[str, Any]]
):
    """
    새로 생성된 질문들을 processed_data.json과 동일한 형식으로 저장
    
    Args:
        conversations: 모든 대화 기록
        created_data_path: 저장할 파일 경로
        context_mapping: context_mapping (context 정보 참조용)
        qid_data: 기존 qid_data (참조용)
    """
    print(f"\n💾 Saving created questions to {created_data_path}...")
    
    # processed_data.json 형식: qid_data와 context_mapping
    created_qid_data = {}
    created_context_mapping = {}
    
    for conv in conversations:
        created_questions = conv.get('created_questions', [])
        
        for cq in created_questions:
            context_id = cq.get('context_id', '')
            doc_id = cq.get('doc_id', '')
            question_id = cq.get('qid')  # dialogue_simulator에서 이미 할당된 qid 사용
            
            if not context_id or not question_id:
                # context_id나 qid가 없으면 건너뛰기
                continue
            
            # context_mapping에서 정보 가져오기 (있으면)
            context_info = context_mapping.get(context_id, {})
            
            # context_mapping에 추가/업데이트
            if context_id not in created_context_mapping:
                created_context_mapping[context_id] = {
                    'context': cq.get('context', '') or context_info.get('context', ''),
                    'doc_id': doc_id or context_info.get('doc_id', ''),
                    'doc_source': context_info.get('doc_source', ''),
                    'doc_class_class': context_info.get('doc_class_class', ''),
                    'doc_class_code': context_info.get('doc_class_code', ''),
                    'q_id': []
                }
            
            # q_id에 추가
            if question_id not in created_context_mapping[context_id]['q_id']:
                created_context_mapping[context_id]['q_id'].append(question_id)
            
            # qid_data에 추가
            created_qid_data[question_id] = {
                'question': cq.get('question', ''),
                'answer': cq.get('answer', ''),
                'clue_text': cq.get('reference_sentence', ''),
                'qa_type': None,  # 새로 생성된 질문은 qa_type이 없음
                'context_id': context_id,
                'similar_context_id': cq.get('similar_context_id', []),  # retrieval로 찾은 유사 컨텍스트
                'doc_id': doc_id or context_info.get('doc_id', ''),
                'doc_source': context_info.get('doc_source', ''),
                'doc_class_class': context_info.get('doc_class_class', ''),
                'doc_class_code': context_info.get('doc_class_code', '')
            }
    
    # processed_data.json 형식으로 저장
    if created_qid_data:
        output_data = {
            'qid_data': created_qid_data,
            'context_mapping': created_context_mapping
        }
        
        with open(created_data_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Saved {len(created_qid_data)} created questions in {len(created_context_mapping)} contexts to {created_data_path}")
    else:
        print("ℹ️ No created questions to save")


def main():
    parser = argparse.ArgumentParser(description="RAG-based dialogue simulation")
    
    # 데이터 경로
    parser.add_argument(
        "--processed-data",
        type=str,
        default="data/processed_data.json",
        help="전처리된 데이터 파일 경로"
    )
    parser.add_argument(
        "--seed-questions",
        type=str,
        default="data/seed_questions.json",
        help="Seed questions 파일 경로"
    )
    
    # LLM 설정
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai", "google"],
        help="LLM provider"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model name"
    )
    
    # 시뮬레이션 설정
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="최대 처리할 seed question 수 (테스트용)"
    )
    parser.add_argument(
        "--min-turns",
        type=int,
        default=4,
        help="대화당 최소 턴 수"
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=8,
        help="대화당 최대 턴 수"
    )
    parser.add_argument(
        "--retrieval-mode",
        type=str,
        default="gold",
        choices=["gold", "topk"],
        help="검색 모드: 'gold'만 또는 'topk'만 (기본값: gold)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="병렬 처리할 워커 수 (기본값: 4)"
    )
    
    # split 설정
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["train", "test"],
        help="데이터 split (train 또는 test)"
    )

    # 출력 설정
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="결과 출력 파일 경로 (기본값: dialogue_data_final_ver2/dialogue_results_{split}.json)"
    )
    parser.add_argument(
        "--created-data",
        type=str,
        default=None,
        help="생성된 질문 저장 파일 경로 (기본값: dialogue_data_final_ver2/created_data_{split}.json)"
    )
    
    args = parser.parse_args()

    split_mode = args.split

    # output/created-data 기본값 설정
    default_output_dir = f"data/dialogue_results"
    if args.output is None:
        args.output = f"{default_output_dir}/dialogue_results_{split_mode}.json"
    if args.created_data is None:
        args.created_data = f"{default_output_dir}/created_data_{split_mode}.json"

    # 출력 디렉토리 자동 생성
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(args.created_data), exist_ok=True)

    processed_data_path = args.processed_data

    print(f"📊 Using split mode: {split_mode}")
    
    # 데이터 로드
    qid_data, context_mapping, seed_questions = load_data(
        processed_data_path=processed_data_path,
        seed_questions_path=args.seed_questions
    )
    
    # 이미 생성된 dialogue 결과가 있으면 해당 seed_qid는 건너뛰기
    if os.path.exists(args.output):
        try:
            with open(args.output, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
            existing_seed_qids = {
                conv.get('seed_qid')
                for conv in existing_results.get('conversations', [])
                if isinstance(conv, dict) and conv.get('seed_qid')
            }
            if existing_seed_qids:
                original_count = len(seed_questions)
                seed_questions = [
                    q for q in seed_questions
                    if q.get('question_id') not in existing_seed_qids
                ]
                skipped = original_count - len(seed_questions)
                print(f"ℹ️ Skipping {skipped} seed questions already in {args.output}")
        except Exception as e:
            print(f"⚠️ Could not read existing results from {args.output}: {e}")
    
    # DataPreprocessor 초기화 (retrieval을 위해)
    print(f"🔧 Initializing DataPreprocessor for retrieval ({split_mode} mode)...")
    preprocessor = DataPreprocessor(
        embedding_model="nlpai-lab/KURE-v1",
        max_token_length=1000,
        top_k_similar=10,
        split_mode=split_mode
    )
    
    # 임베딩 인덱스 로드 (캐시에서)
    # context_mapping에서 raw_paragraphs 재구성
    raw_paragraphs = {}
    context_id_mapping = {}
    for context_id, context_data in context_mapping.items():
        raw_paragraphs[context_id] = {
            'context': context_data.get('context', ''),
            'doc_id': context_data.get('doc_id', ''),
            'doc_source': context_data.get('doc_source', ''),
            'doc_class_class': context_data.get('doc_class_class', ''),
            'doc_class_code': context_data.get('doc_class_code', '')
        }
        context_id_mapping[context_id] = context_id
    
    preprocessor.build_embedding_index(raw_paragraphs, context_id_mapping)
    print(f"✅ Loaded embedding index with {len(preprocessor.context_ids)} contexts")
    
    # Retriever 초기화 (현재는 DummyRetriever, 나중에 실제 구현 추가)
    retriever = DummyRetriever()
    
    # LLM 컴포넌트 생성
    llm_components = create_llm_components(
        provider=args.provider,
        model=args.model
    )
    
    # 최대 qid 계산 및 next_qid 초기화
    max_qid = get_max_qid(qid_data)
    next_qid = max_qid + 1
    qid_lock = threading.Lock()  # thread-safe를 위한 lock
    
    # 다음 qid를 반환하는 함수 생성 (thread-safe)
    def get_next_qid() -> str:
        nonlocal next_qid
        with qid_lock:
            qid = str(next_qid)
            next_qid += 1
            return qid
    
    # Dialogue Simulator 생성
    simulator = DialogueSimulator(
        retriever=retriever,
        question_selector=llm_components['question_selector'],
        reconfirmation_generator=llm_components['reconfirmation_generator'],
        question_answer_extractor=llm_components['question_answer_extractor'],
        question_rewriter=llm_components['question_rewriter'],
        answer_generator=llm_components['answer_generator'],
        qid_data=qid_data,
        context_mapping=context_mapping,
        get_next_qid=get_next_qid,
        preprocessor=preprocessor
    )
    
    # 대화 시뮬레이션 실행
    conversations = simulate_conversations(
        simulator=simulator,
        seed_questions=seed_questions,
        max_questions=args.max_questions,
        min_turns_per_conversation=args.min_turns,
        max_turns_per_conversation=args.max_turns,
        retrieval_mode=args.retrieval_mode,
        num_workers=args.num_workers
    )
    
    # 생성된 질문 저장 (processed_data.json 형식)
    save_created_data(conversations, args.created_data, context_mapping, qid_data)
    
    # 결과 저장
    save_results(conversations, args.output)

    # 통계 출력
    print("\n" + "="*50)
    print("📊 시뮬레이션 결과 통계")
    print("="*50)
    print(f"총 대화 수: {len(conversations)}개")
    if conversations:
        total_turns = sum(c['num_turns'] for c in conversations)
        avg_turns = total_turns / len(conversations)
        print(f"총 턴 수: {total_turns}개")
        print(f"대화당 평균 턴 수: {avg_turns:.2f}개")
        
        # 생성된 질문 통계
        total_created = sum(len(c.get('created_questions', [])) for c in conversations)
        print(f"생성된 새 질문 수: {total_created}개")
    
    print("\n🎉 Dialogue simulation completed!")


if __name__ == "__main__":
    main()

