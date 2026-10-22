#!/usr/bin/env python3
"""
False Acceptance/True Refusal용 정답 없는 Passage 검색
정답이 없는 가까운 passage 10개(+2개 백업)를 찾아서 반환
저장된 임베딩을 로드해서 직접 retrieve 수행
"""

import json
import argparse
import os
import pickle
import hashlib
import threading
import numpy as np
import faiss
from typing import Dict, Any, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer
from passage_classification import PassageClassifier
from qa_model import QAModel


class NoAnswerPassageRetriever:
    """정답이 없는 passage를 찾는 클래스"""
    
    def __init__(
        self,
        qa_model: QAModel,
        embedding_model: str = "nlpai-lab/KURE-v1",
        embedding_cache_dir: str = "vector_db_cache/test",
        classification_cache_path: Optional[str] = None
    ):
        """
        Args:
            qa_model: QAModel 인스턴스
            embedding_model: 임베딩 모델 이름
            embedding_cache_dir: 임베딩 캐시 디렉토리
            classification_cache_path: LLM 판정 캐시 파일 경로 (None이면 캐시 비활성화)
        """
        self.classifier = PassageClassifier(qa_model)
        self.embedding_cache_dir = embedding_cache_dir

        # LLM 판정 캐시
        self._cache: Dict[str, Dict] = {}
        self._cache_path = classification_cache_path
        self._cache_lock = threading.Lock()
        self._cache_hits = 0
        self._cache_misses = 0
        self._load_classification_cache()

        # 임베딩 모델 로드
        print(f"🔧 Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)

        # FAISS 인덱스와 context_ids 로드
        self.faiss_index = None
        self.context_ids = []
        self._load_embedding_cache()

    def _make_cache_key(self, question: str, clue: str, context_id: str) -> str:
        """캐시 키 생성: hash(question+clue) + context_id"""
        h = hashlib.md5((question + "\n" + clue).encode()).hexdigest()[:12]
        return f"{h}_{context_id}"

    def _load_classification_cache(self):
        """LLM 판정 캐시 로드"""
        if self._cache_path and os.path.exists(self._cache_path):
            try:
                with open(self._cache_path, 'r', encoding='utf-8') as f:
                    self._cache = json.load(f)
                print(f"📂 Loaded {len(self._cache)} cached LLM classifications from {self._cache_path}")
            except Exception as e:
                print(f"⚠️  Failed to load classification cache: {e}")
                self._cache = {}
        elif self._cache_path:
            print(f"📂 Classification cache not found, starting fresh: {self._cache_path}")

    def save_classification_cache(self):
        """LLM 판정 캐시를 파일에 저장"""
        if not self._cache_path:
            return
        with self._cache_lock:
            cache_copy = dict(self._cache)
        os.makedirs(os.path.dirname(self._cache_path) or '.', exist_ok=True)
        with open(self._cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_copy, f, ensure_ascii=False)

    def get_cache_stats(self) -> Dict[str, int]:
        """캐시 통계 반환"""
        return {
            'total': len(self._cache),
            'hits': self._cache_hits,
            'misses': self._cache_misses,
        }
    
    def _load_embedding_cache(self):
        """저장된 임베딩 캐시 로드"""
        embeddings_path = os.path.join(self.embedding_cache_dir, "context_embeddings.npy")
        context_ids_path = os.path.join(self.embedding_cache_dir, "context_ids.pkl")
        faiss_index_path = os.path.join(self.embedding_cache_dir, "faiss.index")
        
        if not all(os.path.exists(path) for path in [embeddings_path, context_ids_path, faiss_index_path]):
            raise FileNotFoundError(
                f"Embedding cache not found in {self.embedding_cache_dir}. "
                f"Please run preprocess.py first to generate embeddings."
            )
        
        print(f"📂 Loading embedding cache from {self.embedding_cache_dir}...")
        
        # context_ids 로드
        with open(context_ids_path, 'rb') as f:
            self.context_ids = pickle.load(f)
        
        # FAISS 인덱스 로드
        self.faiss_index = faiss.read_index(faiss_index_path)
        
        print(f"✅ Loaded FAISS index with {len(self.context_ids)} contexts")
    
    def retrieve_similar_contexts(self, question: str, top_k: int = 50) -> List[str]:
        """
        질문으로 유사한 context들을 retrieve
        
        Args:
            question: 질문 텍스트 (original query)
            top_k: 검색할 context 개수
            
        Returns:
            가까운 순서대로 정렬된 context_id 리스트
        """
        if self.faiss_index is None:
            raise RuntimeError("FAISS index not loaded")
        
        # 질문 임베딩 생성
        question_embedding = self.embedder.encode([question])
        faiss.normalize_L2(question_embedding)
        
        # 유사도 검색
        scores, indices = self.faiss_index.search(question_embedding.astype('float32'), top_k)
        
        # context_id 리스트 생성 (중복 제거)
        similar_context_ids = []
        seen_context_ids = set()
        
        for i in indices[0]:
            if i < len(self.context_ids):
                context_id = self.context_ids[i]
                if context_id not in seen_context_ids:
                    similar_context_ids.append(context_id)
                    seen_context_ids.add(context_id)
        
        return similar_context_ids[:top_k]
    
    def find_no_answer_passages(
        self,
        question: str,
        answer: str,
        clue: str,
        context_mapping: Dict[str, Dict[str, Any]],
        target_count: int = 10,
        backup_count: int = 2,
        retrieve_top_k: int = 100,
        num_workers: int = 1,
        max_in_flight: Optional[int] = None,
        verbose: bool = True,
        exclude_context_ids: Optional[set] = None
    ) -> Dict[str, Any]:
        """
        정답이 없는(step1=NO and step2=NO) passage를 찾음
        
        Args:
            question: 원본 질문 (original query, 대명사나 생략 없는 버전)
            answer: 정답 텍스트
            clue: 정답의 핵심 근거가 되는 단서 (R)
            context_mapping: 컨텍스트 매핑 딕셔너리
            target_count: 목표 개수 (기본 10개)
            backup_count: 백업 개수 (기본 2개)
            retrieve_top_k: retrieve할 context 개수 (기본 100개)
            verbose: 상세 출력 여부
            
        Returns:
            {
                "question": "...",
                "answer": "...",
                "no_answer_passages": [
                    {
                        "context_id": "...",
                        "context": "...",
                        "step1": "NO",
                        "step1_confidence": "HIGH/LOW",
                        "step2": "NO",
                        "step2_confidence": "HIGH/LOW",
                        "step1_reason": "...",
                        "step2_reason": "..."
                    }
                ],
                "stats": {
                    "target_count": 10,
                    "backup_count": 2,
                    "total_found": 12,
                    "total_checked": 15
                }
            }
        """
        if verbose:
            print(f"🔍 Finding no-answer passages")
            print(f"   Question: {question[:80]}...")
            print(f"   Target: {target_count} passages (+ {backup_count} backup)")
            if num_workers and num_workers > 1:
                effective_max_in_flight = max_in_flight if max_in_flight is not None else num_workers
                print(f"   Parallel: {num_workers} workers (max_in_flight={effective_max_in_flight})")
        
        # 질문으로 유사한 context들을 retrieve
        similar_context_ids = self.retrieve_similar_contexts(question, top_k=retrieve_top_k)

        # exclude_context_ids 제외 (positive_context_ids, 자기 자신 등)
        if exclude_context_ids:
            similar_context_ids = [cid for cid in similar_context_ids if cid not in exclude_context_ids]

        if verbose:
            print(f"   Retrieved: {len(similar_context_ids)} similar contexts")
        
        no_answer_passages = []
        total_checked = 0
        total_needed = target_count + backup_count

        def _check_one_context(context_id: str, context_text: str) -> Tuple[str, str, str, str, str, str]:
            """
            Worker function: 캐시 확인 후 LLM 호출.
            Returns: (step1, step1_confidence, step2, step2_confidence, step1_reason, step2_reason)
            """
            cache_key = self._make_cache_key(question, clue, context_id)

            # 캐시 히트 확인
            with self._cache_lock:
                cached = self._cache.get(cache_key)
            if cached:
                with self._cache_lock:
                    self._cache_hits += 1
                return (cached['step1'], cached['step1_confidence'],
                        cached['step2'], cached['step2_confidence'],
                        cached['step1_reason'], cached['step2_reason'])

            # 캐시 미스 → LLM 호출
            result = self.classifier.check_answer_in_context(
                context_text, question, answer, clue, early_skip_on_step1_yes=True
            )

            # 결과 캐시에 저장
            with self._cache_lock:
                self._cache_misses += 1
                self._cache[cache_key] = {
                    'step1': result[0], 'step1_confidence': result[1],
                    'step2': result[2], 'step2_confidence': result[3],
                    'step1_reason': result[4], 'step2_reason': result[5],
                }

            return result

        # 병렬 처리 없이: 기존 로직 유지
        if not num_workers or num_workers <= 1:
            # 가까운 순서대로 similar_context_ids를 순회
            for i, context_id in enumerate(similar_context_ids, 1):
                # 이미 충분한 개수를 찾았으면 중단
                if len(no_answer_passages) >= total_needed:
                    if verbose:
                        print(f"\n✅ Found enough passages ({len(no_answer_passages)}/{total_needed})")
                    break
                
                # context 정보 가져오기
                if context_id not in context_mapping:
                    if verbose:
                        print(f"⚠️  [{i}] Context {context_id} not found in mapping, skipping...")
                    continue
                
                context_info = context_mapping[context_id]
                context = context_info.get('context', '')
                
                if not context:
                    if verbose:
                        print(f"⚠️  [{i}] Context {context_id} is empty, skipping...")
                    continue
                
                total_checked += 1
                
                try:
                    # 2단계 검증 수행
                    step1, step1_confidence, step2, step2_confidence, step1_reason, step2_reason = \
                        _check_one_context(context_id, context)
                    
                    # step1과 step2 모두 NO인지 확인
                    step1_upper = step1.upper() if step1 else ""
                    step2_upper = step2.upper() if step2 else ""
                    both_no = (step1_upper == "NO" and step2_upper == "NO")
                    
                    if both_no:
                        # 정답이 없는 passage 발견
                        no_answer_passages.append({
                            "context_id": context_id,
                            "context": context,
                            "doc_id": context_info.get('doc_id', ''),
                            "step1": step1,
                            "step1_confidence": step1_confidence,
                            "step2": step2,
                            "step2_confidence": step2_confidence,
                            "step1_reason": step1_reason,
                            "step2_reason": step2_reason
                        })
                        
                        if verbose:
                            progress = len(no_answer_passages)
                            status = "✓" if progress <= target_count else "✓(backup)"
                            print(f"{status} [{i}] {context_id}: NO/NO [{step1_confidence}/{step2_confidence}] ({progress}/{total_needed})")
                    else:
                        # 정답이 있는 passage
                        if verbose:
                            print(f"✗ [{i}] {context_id}: {step1}/{step2} [{step1_confidence}/{step2_confidence}] - skipped")
                            
                except Exception as e:
                    if verbose:
                        print(f"❌ [{i}] Error checking {context_id}: {e}")
                    continue
        else:
            # 병렬 처리: context 체크를 bounded in-flight로 스케줄링
            effective_max_in_flight = max_in_flight if max_in_flight is not None else num_workers
            if effective_max_in_flight < 1:
                effective_max_in_flight = 1

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_to_meta: Dict[Any, Tuple[int, str, str, Dict[str, Any]]] = {}

                # Submit index over retrieved list
                submit_pos = 0  # 0-based position in similar_context_ids

                while (len(no_answer_passages) < total_needed) and (submit_pos < len(similar_context_ids) or future_to_meta):
                    # Fill in-flight queue
                    while (
                        submit_pos < len(similar_context_ids)
                        and len(future_to_meta) < effective_max_in_flight
                        and len(no_answer_passages) < total_needed
                    ):
                        i = submit_pos + 1
                        context_id = similar_context_ids[submit_pos]
                        submit_pos += 1

                        if context_id not in context_mapping:
                            if verbose:
                                print(f"⚠️  [{i}] Context {context_id} not found in mapping, skipping...")
                            continue

                        context_info = context_mapping[context_id]
                        context = context_info.get('context', '')
                        if not context:
                            if verbose:
                                print(f"⚠️  [{i}] Context {context_id} is empty, skipping...")
                            continue

                        # LLM 체크 (캐시 히트면 즉시 반환)
                        total_checked += 1
                        fut = executor.submit(_check_one_context, context_id, context)
                        future_to_meta[fut] = (i, context_id, context, context_info)

                    if not future_to_meta:
                        continue

                    # Process one completed future at a time to support early stop
                    done_future = next(as_completed(future_to_meta))
                    i, context_id, context, context_info = future_to_meta.pop(done_future)

                    try:
                        step1, step1_confidence, step2, step2_confidence, step1_reason, step2_reason = done_future.result()
                    except Exception as e:
                        if verbose:
                            print(f"❌ [{i}] Error checking {context_id}: {e}")
                        continue

                    step1_upper = step1.upper() if step1 else ""
                    step2_upper = step2.upper() if step2 else ""
                    both_no = (step1_upper == "NO" and step2_upper == "NO")

                    if both_no:
                        no_answer_passages.append({
                            "context_id": context_id,
                            "context": context,
                            "doc_id": context_info.get('doc_id', ''),
                            "step1": step1,
                            "step1_confidence": step1_confidence,
                            "step2": step2,
                            "step2_confidence": step2_confidence,
                            "step1_reason": step1_reason,
                            "step2_reason": step2_reason
                        })

                        if verbose:
                            progress = len(no_answer_passages)
                            status = "✓" if progress <= target_count else "✓(backup)"
                            print(f"{status} [{i}] {context_id}: NO/NO [{step1_confidence}/{step2_confidence}] ({progress}/{total_needed})")
                    else:
                        if verbose:
                            print(f"✗ [{i}] {context_id}: {step1}/{step2} [{step1_confidence}/{step2_confidence}] - skipped")

                if verbose and len(no_answer_passages) >= total_needed:
                    print(f"\n✅ Found enough passages ({len(no_answer_passages)}/{total_needed})")
        
        # 캐시 저장 (매 턴 완료 후)
        self.save_classification_cache()

        # 결과 통계
        found_count = len(no_answer_passages)
        stats = {
            "target_count": target_count,
            "backup_count": backup_count,
            "total_needed": total_needed,
            "total_found": found_count,
            "total_checked": total_checked,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
        }

        if verbose:
            print(f"\n📊 Statistics:")
            print(f"   Checked: {total_checked} passages")
            print(f"   Found: {found_count}/{total_needed} no-answer passages")
            cache_stats = self.get_cache_stats()
            print(f"   Cache: {cache_stats['hits']} hits / {cache_stats['misses']} misses / {cache_stats['total']} total")
            if found_count < total_needed:
                print(f"   ⚠️  Warning: Only found {found_count} passages (needed {total_needed})")
        
        return {
            "question": question,
            "answer": answer,
            "clue": clue,
            "no_answer_passages": no_answer_passages,
            "stats": stats
        }


def process_dialogue_file(
    hallucination_dialogue_path: str,
    original_dialogue_path: str,
    processed_data_path: str,
    output_path: str,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    embedding_model: str = "nlpai-lab/KURE-v1",
    embedding_cache_dir: str = "vector_db_cache/test",
    created_data_path: Optional[str] = None,
    target_count: int = 10,
    backup_count: int = 2,
    retrieve_top_k: int = 100,
    num_workers: int = 1,
    max_in_flight: Optional[int] = None,
    max_dialogues: Optional[int] = None
) -> Dict[str, Any]:
    """
    hallucination_injected_dialogue_results_test.json에서
    false_acceptance와 true_refusal이 필요한 턴들을 찾아
    정답이 없는 passage들을 검색
    
    Args:
        hallucination_dialogue_path: hallucination_injected_dialogue_results_test.json 경로
        original_dialogue_path: dialogue_results_test.json 경로 (original question 추출용)
        processed_data_path: processed_data.json 경로
        output_path: 결과 저장 경로
        provider: LLM provider
        model: LLM model name
        api_key: API key
        embedding_model: 임베딩 모델 이름
        embedding_cache_dir: 임베딩 캐시 디렉토리
        target_count: 목표 passage 개수 (기본 10개)
        backup_count: 백업 passage 개수 (기본 2개)
        retrieve_top_k: retrieve할 context 개수 (기본 100개)
        max_dialogues: 최대 처리할 대화 수 (테스트용, None이면 모두)
        
    Returns:
        결과 딕셔너리
    """
    # 1. 데이터 로드
    print("📁 Loading data...")
    with open(hallucination_dialogue_path, 'r', encoding='utf-8') as f:
        hallucination_data = json.load(f)
    
    with open(original_dialogue_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    with open(processed_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 포맷 정규화: 리스트 형태와 {"conversations": [...]} 형태 모두 지원
    hallucination_conversations = hallucination_data if isinstance(hallucination_data, list) else hallucination_data.get('conversations', [])
    original_conversations = original_data if isinstance(original_data, list) else original_data.get('conversations', [])
    
    qid_data = data.get('qid_data', {})
    context_mapping = data.get('context_mapping', {})
    
    # created_data 병합 (필터링 전에 병합해야 함!)
    if created_data_path:
        try:
            print(f"\n📁 Loading created_data from {created_data_path}...")
            with open(created_data_path, 'r', encoding='utf-8') as f:
                created = json.load(f)
            
            # 포맷 정규화: 리스트 형태와 {"qid_data": {...}, "context_mapping": {...}} 형태 모두 지원
            if isinstance(created, list):
                # 리스트 형태면 qid_data, context_mapping으로 변환
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
                        # context_mapping도 생성
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
                # 딕셔너리 형태면 기존 방식대로
                created_qid_data = created.get('qid_data', {})
                created_context_mapping = created.get('context_mapping', {})
            
            # qid_data 병합
            qid_data.update(created_qid_data)
            
            # context_mapping 병합
            for ctx_id, ctx_info in created_context_mapping.items():
                if ctx_id in context_mapping:
                    context_mapping[ctx_id].update(ctx_info)
                    context_mapping[ctx_id]['q_id'] = list(set(
                        context_mapping[ctx_id].get('q_id', []) + 
                        ctx_info.get('q_id', [])
                    ))
                else:
                    context_mapping[ctx_id] = ctx_info
            
            print(f"✅ Merged created_data: {len(created_qid_data)} questions, {len(created_context_mapping)} contexts")
        except Exception as e:
            print(f"⚠️ Warning: Could not load created_data: {e}")
    
    print(f"✅ Loaded {len(hallucination_conversations)} conversations (hallucination)")
    print(f"✅ Loaded {len(original_conversations)} conversations (original)")
    print(f"✅ Loaded {len(qid_data)} questions (after merge)")
    print(f"✅ Loaded {len(context_mapping)} contexts (after merge)")
    
    # original dialogue에서 (dialogue_id, turn) -> qid 매핑 생성
    print("\n🔧 Building (dialogue_id, turn) to qid mapping...")
    dialogue_turn_to_qid = {}
    dialogue_turn_to_original_question = {}
    for conv in original_conversations:
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
    print(f"✅ Built mapping with {len(dialogue_turn_to_qid)} (dialogue_id, turn) -> qid pairs")
    
    # 2. QA Model 및 Retriever 초기화
    print(f"\n🔧 Initializing QA Model (provider: {provider}, model: {model})...")
    qa_model = QAModel(provider=provider, model=model, api_key=api_key)
    retriever = NoAnswerPassageRetriever(
        qa_model,
        embedding_model=embedding_model,
        embedding_cache_dir=embedding_cache_dir
    )
    print("✅ QA Model and Retriever initialized")
    
    # 3. false_acceptance/true_refusal 턴 찾기
    print("\n🔍 Finding turns that need no-answer passages...")
    target_turns = []
    
    # 먼저 모든 false_acceptance/true_refusal 턴 찾기
    for conv in hallucination_conversations:
        dialogue_id = conv.get('dialogue_id')
        for turn in conv.get('dialogue', []):
            hallucination_type = turn.get('hallucination_type')
            
            # false_acceptance 또는 true_refusal인 경우
            if hallucination_type in ['false_acceptance', 'true_refusal', 'false_refusal']:
                # dialogue_id와 turn으로 original 파일에서 qid 찾기
                turn_idx = turn.get('turn')
                key = (dialogue_id, turn_idx)
                
                qid = dialogue_turn_to_qid.get(key)
                
                if qid and qid in qid_data:
                    # original_question 가져오기
                    original_question = dialogue_turn_to_original_question.get(key)
                    if not original_question:
                        # 매핑에 없으면 qid_data에서 가져오기
                        original_question = qid_data[qid].get('question', '')
                    
                    answer = qid_data[qid].get('answer', '')
                    clue = qid_data[qid].get('clue_text', '')
                    
                    if original_question and answer and clue:
                        target_turns.append({
                            'dialogue_id': dialogue_id,
                            'turn_idx': turn_idx,
                            'hallucination_type': hallucination_type,
                            'qid': qid,
                            'original_question': original_question,
                            'answer': answer,
                            'clue': clue
                        })
                    else:
                        print(f"⚠️  Skipping d{dialogue_id}_t{turn_idx}: missing data (qid={qid}, has_question={bool(original_question)}, has_answer={bool(answer)}, has_clue={bool(clue)})")
                else:
                    if qid:
                        print(f"⚠️  Skipping d{dialogue_id}_t{turn_idx}: qid {qid} not found in qid_data")
                    else:
                        print(f"⚠️  Skipping d{dialogue_id}_t{turn_idx}: qid not found in original dialogue")
    
    # max_dialogues가 지정된 경우, target_turns를 제한
    if max_dialogues and len(target_turns) > max_dialogues:
        print(f"\n📊 Limiting to {max_dialogues} turns (found {len(target_turns)} total)")
        target_turns = target_turns[:max_dialogues]
    
    print(f"✅ Found {len(target_turns)} turns needing no-answer passages")
    if not target_turns:
        print("⚠️  No turns found. Nothing to process.")
        return {}
    
    # 4. 각 턴에 대해 no-answer passages 검색
    print(f"\n🚀 Processing {len(target_turns)} turns...\n")
    results = {}
    
    for idx, turn_info in enumerate(target_turns, 1):
        dialogue_id = turn_info['dialogue_id']
        turn_idx = turn_info['turn_idx']
        hallucination_type = turn_info['hallucination_type']
        qid = turn_info['qid']
        
        key = f"d{dialogue_id}_t{turn_idx}_{qid}"
        
        print(f"\n[{idx}/{len(target_turns)}] Processing {key}")
        print(f"   Type: {hallucination_type}")
        
        try:
            result = retriever.find_no_answer_passages(
                question=turn_info['original_question'],
                answer=turn_info['answer'],
                clue=turn_info['clue'],
                context_mapping=context_mapping,
                target_count=target_count,
                backup_count=backup_count,
                retrieve_top_k=retrieve_top_k,
                num_workers=num_workers,
                max_in_flight=max_in_flight,
                verbose=True
            )
            
            # 메타 정보 추가
            result['dialogue_id'] = dialogue_id
            result['turn_idx'] = turn_idx
            result['hallucination_type'] = hallucination_type
            result['qid'] = qid
            
            results[key] = result
            
        except Exception as e:
            print(f"❌ Error processing {key}: {e}")
            continue
    
    # 5. 결과 저장
    print(f"\n💾 Saving results to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 통계 출력
    total_turns = len(results)
    total_passages_found = sum(len(r['no_answer_passages']) for r in results.values())
    avg_passages = total_passages_found / total_turns if total_turns > 0 else 0
    
    print(f"\n📊 Final Statistics:")
    print(f"   Processed turns: {total_turns}")
    print(f"   Total no-answer passages found: {total_passages_found}")
    print(f"   Average passages per turn: {avg_passages:.1f}")
    print(f"\n✅ Done!")
    
    return results


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="Retrieve no-answer passages for False Acceptance/True Refusal")
    parser.add_argument("--provider", type=str, default="openai", choices=["openai", "google"])
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--hallucination-dialogue", type=str, required=True,
                        help="Path to hallucination_injected_dialogue_results_test.json")
    parser.add_argument("--original-dialogue", type=str, required=True,
                        help="Path to dialogue_results_test.json (for original questions)")
    parser.add_argument("--processed-data", type=str, required=True,
                        help="Path to processed_data.json")
    parser.add_argument("--output", type=str, default="no_answer_passages.json",
                        help="Output JSON file path")
    parser.add_argument("--embedding-model", type=str, default="nlpai-lab/KURE-v1",
                        help="Embedding model name")
    parser.add_argument("--embedding-cache-dir", type=str, default="vector_db_cache/test",
                        help="Embedding cache directory")
    parser.add_argument("--created-data", type=str, default=None,
                        help="Path to created_data.json (optional, for newly created questions)")
    parser.add_argument("--target-count", type=int, default=10,
                        help="Target number of no-answer passages per turn")
    parser.add_argument("--backup-count", type=int, default=2,
                        help="Additional backup passages per turn")
    parser.add_argument("--retrieve-top-k", type=int, default=100,
                        help="Number of contexts to retrieve from FAISS")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Number of parallel workers for LLM-based context checking (per turn)")
    parser.add_argument("--max-in-flight", type=int, default=None,
                        help="Bound on in-flight LLM checks when num-workers > 1 (default: num_workers)")
    parser.add_argument("--max-dialogues", type=int, default=None,
                        help="Maximum number of dialogues to process (for testing)")
    
    args = parser.parse_args()
    
    # 처리 실행
    process_dialogue_file(
        hallucination_dialogue_path=args.hallucination_dialogue,
        original_dialogue_path=args.original_dialogue,
        processed_data_path=args.processed_data,
        output_path=args.output,
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        embedding_model=args.embedding_model,
        embedding_cache_dir=args.embedding_cache_dir,
        created_data_path=args.created_data,
        target_count=args.target_count,
        backup_count=args.backup_count,
        retrieve_top_k=args.retrieve_top_k,
        num_workers=args.num_workers,
        max_in_flight=args.max_in_flight,
        max_dialogues=args.max_dialogues
    )


if __name__ == "__main__":
    main()

