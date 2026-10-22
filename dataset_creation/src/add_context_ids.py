#!/usr/bin/env python3
"""
할루시네이션이 주입된 대화에 positive_context_ids와 negative_context_ids를 추가하는 스크립트

- positive_context_ids: original_question을 임베딩하여 유사한 context_id를 찾아 추가
- negative_context_ids: hardnegative_passages_test.json에서 false_refusal 또는 false_acceptance 10개를 가져옴
"""

import json
import random
import os
import pickle
import argparse
import numpy as np
import faiss
from pathlib import Path
from typing import Dict, List, Any
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def load_json(filepath: str) -> Any:
    """JSON 파일을 로드합니다."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, filepath: str) -> None:
    """JSON 데이터를 파일로 저장합니다."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


class EmbeddingSearcher:
    """임베딩 기반 유사 컨텍스트 검색기"""
    
    def __init__(self, embedding_model: str = "nlpai-lab/KURE-v1", 
                 cache_dir: str = "vector_db_cache/test"):
        """
        Args:
            embedding_model: 사용할 임베딩 모델 이름
            cache_dir: 임베딩 캐시 디렉토리 경로
        """
        self.embedding_model = embedding_model
        self.cache_dir = cache_dir
        
        # 모델 로드
        print(f"🔧 Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)
        
        # 캐시 로드
        self.context_embeddings = None
        self.context_ids = []
        self.faiss_index = None
        self._load_cache()
    
    def _get_cache_paths(self):
        """캐시 파일 경로 반환"""
        embeddings_path = os.path.join(self.cache_dir, "context_embeddings.npy")
        context_ids_path = os.path.join(self.cache_dir, "context_ids.pkl")
        faiss_index_path = os.path.join(self.cache_dir, "faiss.index")
        return embeddings_path, context_ids_path, faiss_index_path
    
    def _load_cache(self):
        """임베딩 캐시 로드"""
        embeddings_path, context_ids_path, faiss_index_path = self._get_cache_paths()
        
        if not all(os.path.exists(path) for path in [embeddings_path, context_ids_path, faiss_index_path]):
            raise FileNotFoundError(f"Embedding cache not found in {self.cache_dir}. Please run preprocess.py first.")
        
        print(f"📂 Loading embedding cache from {self.cache_dir}...")
        self.context_embeddings = np.load(embeddings_path)
        with open(context_ids_path, 'rb') as f:
            self.context_ids = pickle.load(f)
        self.faiss_index = faiss.read_index(faiss_index_path)
        print(f"✅ Loaded {len(self.context_ids)} context embeddings")
    
    def find_similar_contexts(self, question: str, top_k: int = 10) -> List[str]:
        """
        질문과 유사한 컨텍스트 ID들을 찾습니다.
        
        Args:
            question: 검색할 질문
            top_k: 반환할 유사 컨텍스트 개수
        
        Returns:
            유사한 context_id 리스트
        """
        if self.faiss_index is None:
            return []
        
        # 질문 임베딩 생성
        question_embedding = self.embedder.encode([question])
        faiss.normalize_L2(question_embedding)
        
        # 유사도 검색
        scores, indices = self.faiss_index.search(question_embedding.astype('float32'), top_k)
        
        # 유사한 context_id 반환
        similar_context_ids = []
        seen_context_ids = set()
        
        for i in indices[0]:
            if i < len(self.context_ids):
                context_id = self.context_ids[i]
                if context_id not in seen_context_ids:
                    similar_context_ids.append(context_id)
                    seen_context_ids.add(context_id)
        
        return similar_context_ids[:top_k]


def get_negative_context_ids(hardnegative_data: Dict, dialogue_id: int, turn: int, qid: str, num_samples: int = 10) -> List[str]:
    """
    hardnegative_passages_test.json에서 false_refusal 또는 false_acceptance 항목을 가져옵니다.
    
    Args:
        hardnegative_data: hardnegative_passages_test.json 데이터
        dialogue_id: 대화 ID
        turn: 턴 번호
        qid: 질문 ID
        num_samples: 가져올 샘플 개수 (기본값: 10)
    
    Returns:
        negative context_id 리스트
    """
    # 현재 dialogue의 키 생성
    key = f"d{dialogue_id}_t{turn}_{qid}"
    
    # 해당 키가 없으면 빈 리스트 반환
    if key not in hardnegative_data:
        return []
    
    # no_answer_passages에서 false_refusal 또는 false_acceptance 필터링
    negative_passages = []
    for passage in hardnegative_data[key].get('no_answer_passages', []):
        step1 = passage.get('step1', '')
        step2 = passage.get('step2', '')
        
        # false_refusal: step1="NO" and step2="NO"
        # false_acceptance: step1="YES" and step2="NO"
        if (step1 == "NO" and step2 == "NO") or (step1 == "YES" and step2 == "NO"):
            negative_passages.append(passage['context_id'])
    
    # 최대 num_samples개만 선택
    if len(negative_passages) > num_samples:
        negative_passages = random.sample(negative_passages, num_samples)
    
    return negative_passages


def add_context_ids_to_hallucinated_turns(
    hallucination_data: List[Dict],
    embedder: EmbeddingSearcher,
    hardnegative_data: Dict
) -> List[Dict]:
    """
    할루시네이션이 주입된 턴에 positive_context_ids와 negative_context_ids를 추가합니다.
    
    Args:
        hallucination_data: hallucination_injected_dialogue_results_test.json 데이터
        embedder: 임베딩 검색기
        hardnegative_data: hardnegative_passages_test.json 데이터
    
    Returns:
        수정된 hallucination_data
    """
    total_hallucinated_turns = 0
    total_with_positive = 0
    total_with_negative = 0
    context_id_already_included = 0
    context_id_added = 0
    position_stats = {i: 0 for i in range(1, 11)}  # 1-10번째 위치별 개수
    
    for dialogue in tqdm(hallucination_data, desc="Processing dialogues"):
        dialogue_id = dialogue['dialogue_id']
        
        for turn_data in dialogue['dialogue']:
            hallucination_type = turn_data.get('hallucination_type')
            
            # 할루시네이션이 주입된 경우만 처리
            if hallucination_type is not None:
                total_hallucinated_turns += 1
                turn_num = turn_data.get('turn')
                qid = turn_data.get('qid')
                
                # 현재 턴의 context_id를 passages에서 가져오기
                current_context_id = None
                if turn_data.get('passages') and len(turn_data['passages']) > 0:
                    current_context_id = turn_data['passages'][0].get('context_id')
                
                # positive_context_ids 추가: original_question 사용
                positive_context_ids = []
                original_question = turn_data.get('original_question', '')
                
                if original_question:
                    # 임베딩으로 유사한 context 찾기
                    similar_context_ids = embedder.find_similar_contexts(original_question, top_k=10)
                    
                    if similar_context_ids and current_context_id:
                        # similar_context_ids를 복사
                        positive_context_ids = similar_context_ids.copy()
                        
                        # context_id가 이미 포함되어 있는지 확인
                        if current_context_id in positive_context_ids:
                            context_id_already_included += 1
                            # 이미 포함되어 있으면 그대로 유지 (최대 10개)
                            positive_context_ids = positive_context_ids[:10]
                        else:
                            context_id_added += 1
                            # 포함되어 있지 않으면
                            # 1. 리스트가 10개 이상이면 마지막 항목 제거
                            if len(positive_context_ids) >= 10:
                                positive_context_ids = positive_context_ids[:9]
                            
                            # 2. context_id를 append
                            positive_context_ids.append(current_context_id)
                        
                        # 3. 전체 shuffle
                        random.shuffle(positive_context_ids)
                        
                        # 4. shuffle 후 위치 통계
                        pos = positive_context_ids.index(current_context_id) + 1
                        if pos <= 10:
                            position_stats[pos] += 1
                    elif current_context_id:
                        # similar_context_ids가 없지만 current_context_id는 있는 경우
                        positive_context_ids = [current_context_id]
                        position_stats[1] += 1
                else:
                    # original_question이 없는 경우
                    if current_context_id:
                        positive_context_ids = [current_context_id]
                        position_stats[1] += 1
                
                if positive_context_ids:
                    total_with_positive += 1
                
                turn_data['positive_context_ids'] = positive_context_ids
                
                # negative_context_ids 추가 (false_acceptance와 false_refusal만)
                negative_context_ids = []
                if hallucination_type in ['false_acceptance', 'false_refusal']:
                    negative_context_ids = get_negative_context_ids(
                        hardnegative_data, 
                        dialogue_id, 
                        turn_num, 
                        qid, 
                        num_samples=10
                    )
                    
                    if negative_context_ids:
                        total_with_negative += 1
                
                turn_data['negative_context_ids'] = negative_context_ids
    
    # 통계 출력
    print(f"\n=== 처리 통계 ===")
    print(f"총 할루시네이션 주입 턴: {total_hallucinated_turns}")
    print(f"Positive context IDs 추가된 턴: {total_with_positive}")
    print(f"Negative context IDs 추가된 턴 (false_acceptance/false_refusal만): {total_with_negative}")
    print(f"\n=== Positive Context ID 획득 방법 ===")
    print(f"임베딩 검색 결과에 이미 포함: {context_id_already_included}")
    print(f"임베딩 검색 결과에 새로 추가: {context_id_added}")
    print(f"\n=== Context ID 위치 통계 (shuffle 후, 10개 passage 중) ===")
    for pos in range(1, 11):
        print(f"{pos}번째 위치: {position_stats[pos]}개")
    
    return hallucination_data


def verify_context_ids(hallucination_data: List[Dict]) -> None:
    """
    모든 할루시네이션 턴에서 context_id가 positive_context_ids에 포함되어 있는지 검증합니다.
    """
    print(f"\n=== Context ID 포함 여부 검증 ===")
    
    total_hallucinated = 0
    context_id_included = 0
    context_id_missing = 0
    missing_cases = []
    
    for dialogue in hallucination_data:
        for turn_data in dialogue['dialogue']:
            if turn_data.get('hallucination_type') is not None:
                total_hallucinated += 1
                
                # passages에서 context_id 가져오기
                context_id = None
                if turn_data.get('passages') and len(turn_data['passages']) > 0:
                    context_id = turn_data['passages'][0].get('context_id')
                
                # positive_context_ids에 포함되어 있는지 확인
                positive_context_ids = turn_data.get('positive_context_ids', [])
                
                if context_id and context_id in positive_context_ids:
                    context_id_included += 1
                else:
                    context_id_missing += 1
                    missing_cases.append({
                        'dialogue_id': dialogue['dialogue_id'],
                        'turn': turn_data['turn'],
                        'qid': turn_data['qid'],
                        'context_id': context_id,
                        'positive_ids_count': len(positive_context_ids)
                    })
    
    print(f"총 할루시네이션 턴: {total_hallucinated}")
    print(f"Context ID가 positive_context_ids에 포함됨: {context_id_included} ✓")
    print(f"Context ID가 positive_context_ids에 없음: {context_id_missing} ✗")
    
    if missing_cases:
        print(f"\n⚠️  누락된 케이스:")
        for case in missing_cases[:5]:  # 최대 5개만 출력
            print(f"  - Dialogue {case['dialogue_id']}, Turn {case['turn']}, QID: {case['qid']}")
            print(f"    Context ID: {case['context_id']}, Positive IDs count: {case['positive_ids_count']}")
    else:
        print(f"\n✅ 모든 할루시네이션 턴에서 context_id가 positive_context_ids에 포함되어 있습니다!")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="할루시네이션 대화에 positive/negative context_ids 추가"
    )
    parser.add_argument("--hallucination-dialogue", type=str, default=None,
                        help="할루시네이션 주입 대화 파일")
    parser.add_argument("--hardnegative", type=str, default=None,
                        help="hard negative passages 파일")
    parser.add_argument("--output", type=str, default=None,
                        help="출력 파일 경로")
    parser.add_argument("--embedding-cache-dir", type=str, default=None,
                        help="임베딩 캐시 디렉토리")
    parser.add_argument("--embedding-model", type=str, default="nlpai-lab/KURE-v1",
                        help="임베딩 모델 이름")
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent

    # 입력 파일명에서 split 자동 감지 (train/test)
    hallucination_file = Path(args.hallucination_dialogue) if args.hallucination_dialogue else \
        base_dir / "benchmark_dataset" / "hallucination_injected_dialogue_results_test.json"

    # split 감지: 파일명에 'train'이 포함되면 train, 아니면 test
    input_filename = hallucination_file.name
    split = "train" if "train" in input_filename else "test"

    hardnegative_file = Path(args.hardnegative) if args.hardnegative else \
        base_dir / "benchmark_dataset" / f"hardnegative_passages_{split}.json"
    output_file = Path(args.output) if args.output else \
        base_dir / "benchmark_dataset" / f"hallucination_injected_dialogue_results_{split}_with_contexts.json"
    embedding_cache_dir = Path(args.embedding_cache_dir) if args.embedding_cache_dir else \
        base_dir / "vector_db_cache" / split

    # 파일 존재 확인
    if not hallucination_file.exists():
        print(f"Error: {hallucination_file} 파일을 찾을 수 없습니다.")
        return

    if not embedding_cache_dir.exists():
        print(f"Error: {embedding_cache_dir} 디렉토리를 찾을 수 없습니다.")
        print("먼저 preprocess.py를 실행하여 임베딩 캐시를 생성하세요.")
        return

    print("파일 로딩 중...")
    # 데이터 로드
    hallucination_data = load_json(hallucination_file)
    if hardnegative_file.exists():
        hardnegative_data = load_json(hardnegative_file)
    else:
        print(f"⚠️ {hardnegative_file} 없음 → negative_context_ids 비활성화")
        hardnegative_data = {}

    # conversations 형식이면 리스트로 변환
    if isinstance(hallucination_data, dict) and "conversations" in hallucination_data:
        hallucination_data = hallucination_data["conversations"]

    print(f"할루시네이션 데이터: {len(hallucination_data)}개 대화")
    print(f"Hard negative 데이터: {len(hardnegative_data)}개 항목")

    # 임베딩 검색기 초기화
    embedder = EmbeddingSearcher(
        embedding_model=args.embedding_model,
        cache_dir=str(embedding_cache_dir)
    )

    # Context IDs 추가
    print("\nContext IDs 추가 중...")
    modified_data = add_context_ids_to_hallucinated_turns(
        hallucination_data,
        embedder,
        hardnegative_data
    )

    # 검증
    verify_context_ids(modified_data)

    # 결과 저장
    print(f"\n결과를 {output_file}에 저장 중...")
    save_json(modified_data, output_file)
    print("완료!")


if __name__ == "__main__":
    main()
