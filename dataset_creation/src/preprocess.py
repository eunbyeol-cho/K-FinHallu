#!/usr/bin/env python3
"""
JSON 파일 전처리 스크립트
- raw_data: 문서 기준 전처리 (금융 문서 필터링, source filter, 토큰 길이 필터링, 빈 문서 제거)
- labeled_data: QA 기준 전처리 (질문-답변 추출)
- 질문 임베딩 기반 유사 컨텍스트 검색
- qid를 키로 하는 데이터 구조 생성
"""

import json
import os
import pickle
import re
import numpy as np
from typing import List, Dict, Any, Tuple, Set
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer


def clean_doc_source(doc_source: str) -> str:
    """doc_source에서 괄호와 그 안의 내용을 제거"""
    if not doc_source:
        return doc_source
    cleaned = re.sub(r'[\(（].*?[\)）]', '', doc_source)
    cleaned = cleaned.strip()
    cleaned = re.sub(r'^[,，\n\r\s]+', '', cleaned)
    cleaned = re.sub(r'[,，\n\r\s]+$', '', cleaned)
    return cleaned


# AI-Hub 배포 zip의 유형 번호 순서 (1.정답경계 추출형, 2.YesNo 단문형, 3.Table 정답 추출형,
# 4.다지선다형, 5.절차(방법)형). 논문 데이터는 이 순서로 파일을 읽어 만들어졌다.
AIHUB_TYPE_ORDER = ['span_extraction', 'text_entailment', 'tableqa', 'multiple_choice', 'span_extraction_how']


def aihub_file_order(filename: str):
    """'TS_span_extraction_how.json' -> (4, name): 유형 번호 순, 모르는 파일은 맨 뒤."""
    qtype = filename[3:-len('.json')] if filename[2:3] == '_' else filename
    rank = AIHUB_TYPE_ORDER.index(qtype) if qtype in AIHUB_TYPE_ORDER else len(AIHUB_TYPE_ORDER)
    return rank, filename


class DataPreprocessor:
    """데이터 전처리 클래스"""
    
    def __init__(self, 
                 embedding_model: str = "nlpai-lab/KURE-v1",
                 max_token_length: int = 1000,
                 top_k_similar: int = 10,
                 embedding_cache_dir: str = "vector_db_cache",
                 max_instances: int = None,
                 question_types: List[str] = None,
                 split_mode: str = "train",
                 train_sources: List[str] = None,
                 test_sources: List[str] = None):
        self.embedding_model = embedding_model
        self.max_token_length = max_token_length
        self.top_k_similar = top_k_similar
        self.max_instances = max_instances
        self.question_types = question_types
        self.split_mode = split_mode
        self.train_sources = train_sources
        self.test_sources = test_sources
        
        # 캐시 디렉토리 설정
        if split_mode in ["train", "test"]:
            self.embedding_cache_dir = os.path.join(embedding_cache_dir, split_mode)
        else:
            self.embedding_cache_dir = embedding_cache_dir
        os.makedirs(self.embedding_cache_dir, exist_ok=True)
        
        # 모델 초기화
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        print(f"🔧 Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)
        
        # 데이터 저장소
        self.context_embeddings = None
        self.context_ids = []
        self.faiss_index = None
        self.context_id_mapping = {}
    
    # ==================== 데이터 로드 ====================
    
    def load_data(self, labeled_data_dir, raw_data_dir) -> Tuple[List[Dict], List[Dict]]:
        """labeled_data와 raw_data 로드"""
        print("📁 Loading data...")
        
        def load_json_files(directories) -> List[Dict]:
            all_data = []
            if isinstance(directories, str):
                directories = [directories]
            
            for directory in directories:
                if not os.path.exists(directory):
                    print(f"⚠️ Directory not found: {directory}")
                    continue
                
                # 파일 순서가 중복 context 제거 결과(어느 context_id를 남길지)를 결정하므로
                # os.listdir 순서에 맡기지 않고 AI-Hub 유형 번호 순서로 고정한다.
                json_files = sorted((f for f in os.listdir(directory) if f.endswith('.json')),
                                    key=aihub_file_order)
                for json_file in json_files:
                    file_path = os.path.join(directory, json_file)
                    try:
                        with open(file_path, 'r', encoding='utf-8-sig') as f:
                            data = json.load(f)["data"]
                            all_data.extend(data)
                    except Exception as e:
                        print(f"Error loading {json_file}: {e}")
            
            return all_data
        
        labeled_data = load_json_files(labeled_data_dir)
        raw_data = load_json_files(raw_data_dir)
        
        print(f"✅ Loaded {len(labeled_data)} labeled documents")
        print(f"✅ Loaded {len(raw_data)} raw documents")
        
        return labeled_data, raw_data
    
    # ==================== 유틸리티 함수 ====================
    
    def _is_financial_document(self, doc: Dict) -> bool:
        """금융 문서인지 확인"""
        doc_class = doc.get('doc_class', {})
        doc_class_class = doc_class.get('class', '') if doc_class else ''
        if not doc_class_class:
            return False
        subject = doc_class_class.split("/")[0]
        return subject == "금융"
    
    def _should_include_doc(self, doc_source: str) -> bool:
        """doc_source를 기반으로 train/test split 여부 결정"""
        cleaned_source = clean_doc_source(doc_source)
        
        if self.split_mode == "test":
            return cleaned_source in self.test_sources
        else:  # train 모드
            return cleaned_source in self.train_sources
            
    def _calculate_token_length(self, text: str) -> int:
        """텍스트의 토큰 길이 계산"""
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        return len(tokens)
    
    def _convert_qa_type_to_string(self, qa_type_num: int) -> str:
        """qa_type 숫자를 문자열로 변환"""
        qa_type_mapping = {
            1: "span_extraction",
            5: "span_extraction_how", 
            4: "multiple_choice",
            2: "text_entailment"
        }
        return qa_type_mapping.get(qa_type_num, f"unknown_{qa_type_num}")
    
    def _extract_sentence_from_span(self, context: str, answer_start: int, answer_end: int) -> str:
        """answer_start와 answer_end를 포함하는 문장 추출 (\n 기준)"""
        if not context or answer_start is None or answer_end is None:
            return ""
        
        if answer_start < 0 or answer_end > len(context) or answer_start >= answer_end:
            return ""
        
        # answer_start 이전의 줄바꿈 위치 찾기
        sentence_start = 0
        for i in range(answer_start - 1, -1, -1):
            if context[i] == '\n':
                sentence_start = i + 1
                break
        
        # answer_end 이후의 줄바꿈 위치 찾기
        sentence_end = len(context)
        for i in range(answer_end, len(context)):
            if context[i] == '\n':
                sentence_end = i
                break
        
        return context[sentence_start:sentence_end].strip()
    
    # ==================== Raw Data 전처리 (문서 기준) ====================
    
    def _filter_paragraph(self, context: str, context_id: str) -> bool:
        """paragraph 필터링 (토큰 길이 체크)"""
        if not context or not context_id:
            return False
        
        token_length = self._calculate_token_length(context)
        return token_length <= self.max_token_length
    
    def _process_document_paragraphs(self, doc: Dict) -> List[Dict]:
        """문서의 paragraph들을 처리하여 유효한 paragraph만 반환"""
        valid_paragraphs = []
        
        for paragraph in doc.get('paragraphs', []):
            context = paragraph.get('context', '')
            context_id = paragraph.get('context_id', '')
            
            if not self._filter_paragraph(context, context_id):
                continue
            
            valid_paragraphs.append({
                'context_id': context_id,
                'context': context,
                'doc_id': doc.get('doc_id', ''),
                'doc_source': doc.get('doc_source', ''),
                'doc_class_class': doc.get('doc_class', {}).get('class', '') if doc.get('doc_class') else '',
                'doc_class_code': doc.get('doc_class', {}).get('code', '') if doc.get('doc_class') else '',
                'token_length': self._calculate_token_length(context)
            })
        
        return valid_paragraphs
    
    def process_raw_data(self, raw_data: List[Dict]) -> Tuple[Dict[str, Dict], Dict[str, str]]:
        """
        raw_data 전처리 (문서 기준)
        - 금융 문서 필터링
        - source filter (train/test split)
        - 토큰 길이 넘는 paragraph 제거
        - paragraph가 비어있는 문서 제거
        - context 내용 기준 중복 제거 및 매핑 생성
        
        Returns:
            paragraphs: {context_id: paragraph_info} - 유니크한 paragraph들
            context_id_mapping: {duplicate_context_id: first_context_id} - 중복 매핑
        """
        print(f"🔄 Processing raw data ({self.split_mode} mode)...")
        
        # 1. 금융 문서 필터링
        financial_docs = [doc for doc in raw_data if self._is_financial_document(doc)]
        print(f"   📊 Financial documents: {len(financial_docs)}")
        
        # 2. Source filter (train/test split)
        filtered_docs = [doc for doc in financial_docs if self._should_include_doc(doc.get('doc_source', ''))]
        print(f"   📊 After source filter: {len(filtered_docs)} documents")
        
        # 3. Paragraph 처리 및 빈 문서 제거
        # context 내용 기준으로 첫 번째 context_id를 저장
        paragraphs = {}  # {context_id: paragraph_info} - 유니크한 paragraph만 저장
        context_to_first_id = {}  # {context: first_context_id} - context 내용 -> 첫 번째 context_id
        context_id_mapping = {}  # {duplicate_context_id: first_context_id} - 중복 매핑
        
        for doc in tqdm(filtered_docs, desc="Processing documents"):
            valid_paragraphs = self._process_document_paragraphs(doc)
            
            if not valid_paragraphs:  # 유효한 paragraph가 없는 문서 제거 (빈 리스트)
                continue
            
            for para in valid_paragraphs:
                context_id = para['context_id']
                context = para['context']
                
                # context_id 중복 체크 (같은 context_id가 이미 있으면 스킵)
                if context_id in paragraphs:
                    continue
                
                # context 내용 기준 중복 체크
                if context in context_to_first_id:
                    # 같은 context 내용이 이미 있으면 매핑 생성
                    first_context_id = context_to_first_id[context]
                    context_id_mapping[context_id] = first_context_id
                    continue
                
                # 새로운 유니크한 paragraph
                paragraphs[context_id] = para
                context_to_first_id[context] = context_id
        
        # 통계 출력 (unique 개수만)
        unique_documents = set(para['doc_id'] for para in paragraphs.values())
        
        print(f"\n📊 Raw Data Statistics:")
        print(f"   Unique documents: {len(unique_documents)}")
        print(f"   Unique paragraphs: {len(paragraphs)}")
        
        return paragraphs, context_id_mapping
    
    # ==================== Labeled Data 전처리 (QA 기준) ====================
    
    def _extract_qa_from_paragraph(self, paragraph: Dict, doc: Dict, context_id_mapping: Dict[str, str] = None) -> List[Dict]:
        """paragraph에서 QA 정보 추출 (홀수 question_id만)"""
        if context_id_mapping is None:
            context_id_mapping = {}
        
        qa_list = []
        context = paragraph.get('context', '')
        context_id = paragraph.get('context_id', '')
        qas = paragraph.get('qas', [])
        
        if not context or not context_id:
            return qa_list
                
        # context_id 매핑 적용
        mapped_context_id = context_id_mapping.get(context_id, context_id)
        
        for qa in qas:
            question_id = qa.get('question_id', '')
            
            # 홀수 question_id만 필터링
            if not question_id or int(question_id) % 2 == 0:
                continue
            
            question = qa.get('question', '')
            answer = qa.get('answer', {})
            qa_type_num = qa.get('qa_type', '')
            
            if not question or not answer:
                continue
            
            # qa_type 변환
            qa_type_str = self._convert_qa_type_to_string(qa_type_num)
            
            # clue_text 처리
            clue_text = answer.get('clue_text')
            if qa_type_str in ['span_extraction', 'span_extraction_how']:
                answer_start = answer.get('answer_start')
                answer_end = answer.get('answer_end')
                if answer_start is not None and answer_end is not None:
                    clue_text = self._extract_sentence_from_span(context, answer_start, answer_end)
            
            qa_list.append({
                'question_id': question_id,
                'question': question,
                'answer': answer.get('text', ''),
                'clue_text': clue_text,
                'qa_type': qa_type_str,  # qa_type 저장
                'context_id': mapped_context_id,  # 매핑된 context_id 사용
                'doc_id': doc.get('doc_id', ''),
                'doc_source': doc.get('doc_source', ''),
                'doc_class_class': doc.get('doc_class', {}).get('class', '') if doc.get('doc_class') else '',
                'doc_class_code': doc.get('doc_class', {}).get('code', '') if doc.get('doc_class') else '',
                'context': context
            })
        
        return qa_list
    
    def process_labeled_data(self, labeled_data: List[Dict], context_id_mapping: Dict[str, str] = None, raw_paragraphs: Dict[str, Dict] = None) -> List[Dict]:
        """
        labeled_data 전처리 (QA 기준)
        - 금융 문서 필터링
        - source filter (train/test split)
        - 홀수 question_id만 추출
        - 토큰 길이 필터링
        
        Args:
            labeled_data: 전처리할 labeled data
            context_id_mapping: context_id 매핑 (raw_data에서 생성된 것)
            raw_paragraphs: raw_data에서 처리된 paragraphs (통계 계산용)
        """
        print(f"🔄 Processing labeled data ({self.split_mode} mode)...")
        
        if context_id_mapping is None:
            context_id_mapping = {}
        if raw_paragraphs is None:
            raw_paragraphs = {}
        
        # 1. 금융 문서 필터링
        financial_docs = [doc for doc in labeled_data if self._is_financial_document(doc)]
        print(f"   📊 Financial documents: {len(financial_docs)}")
        
        # 2. Source filter (train/test split)
        filtered_docs = [doc for doc in financial_docs if self._should_include_doc(doc.get('doc_source', ''))]
        print(f"   📊 After source filter: {len(filtered_docs)} documents")
        
        # 3. QA 추출
        qa_list = []
        
        for doc in tqdm(filtered_docs, desc="Processing QA"):
            for paragraph in doc.get('paragraphs', []):
                context_id = paragraph.get('context_id', '')
                if not context_id:
                    continue
                
                # context_id 매핑 적용
                mapped_context_id = context_id_mapping.get(context_id, context_id)
                
                # raw_paragraphs에 있는 paragraph만 처리
                if mapped_context_id not in raw_paragraphs:
                    continue
                
                # QA 추출 (context_id_mapping 적용)
                extracted_qas = self._extract_qa_from_paragraph(paragraph, doc, context_id_mapping)
                
                if not extracted_qas:
                    continue
                
                qa_list.extend(extracted_qas)
        
        # 통계 출력 (unique 개수만)
        unique_questions = set(qa['question_id'] for qa in qa_list)
        unique_paragraphs = set(qa['context_id'] for qa in qa_list)  # 모두 raw_paragraphs에 있는 것들
        # raw_paragraphs의 doc_id를 사용하여 일관성 유지
        unique_documents = set(raw_paragraphs.get(qa['context_id'], {}).get('doc_id', qa['doc_id']) for qa in qa_list)
        
        print(f"\n📊 Labeled Data Statistics:")
        print(f"   Unique questions: {len(unique_questions)}")
        print(f"   Unique paragraphs: {len(unique_paragraphs)}")
        print(f"   Unique documents: {len(unique_documents)}")
        
        return qa_list
    
    # ==================== 임베딩 인덱스 구축 ====================
    
    def _get_cache_paths(self):
        """캐시 파일 경로 생성"""
        embeddings_path = os.path.join(self.embedding_cache_dir, "context_embeddings.npy")
        context_ids_path = os.path.join(self.embedding_cache_dir, "context_ids.pkl")
        faiss_index_path = os.path.join(self.embedding_cache_dir, "faiss.index")
        return embeddings_path, context_ids_path, faiss_index_path
    
    def _save_embedding_cache(self, embeddings: np.ndarray, context_ids: List[str], faiss_index: faiss.Index):
        """임베딩 캐시 저장"""
        embeddings_path, context_ids_path, faiss_index_path = self._get_cache_paths()
        os.makedirs(self.embedding_cache_dir, exist_ok=True)
        
        print(f"💾 Saving embedding cache...")
        np.save(embeddings_path, embeddings)
        with open(context_ids_path, 'wb') as f:
            pickle.dump(context_ids, f)
        faiss.write_index(faiss_index, faiss_index_path)
        print(f"✅ Saved cache to {self.embedding_cache_dir}")
    
    def _load_embedding_cache(self) -> Tuple[np.ndarray, List[str], faiss.Index]:
        """임베딩 캐시 로드"""
        embeddings_path, context_ids_path, faiss_index_path = self._get_cache_paths()
        
        if not all(os.path.exists(path) for path in [embeddings_path, context_ids_path, faiss_index_path]):
            return None, None, None
        
        print(f"📂 Loading embedding cache...")
        embeddings = np.load(embeddings_path)
        with open(context_ids_path, 'rb') as f:
            context_ids = pickle.load(f)
        faiss_index = faiss.read_index(faiss_index_path)
        print(f"✅ Loaded cache with {len(context_ids)} embeddings")
        return embeddings, context_ids, faiss_index
    
    def build_embedding_index(self, raw_paragraphs: Dict[str, Dict], context_id_mapping: Dict[str, str]):
        """
        컨텍스트 임베딩 인덱스 구축 (캐시 사용)
        raw_paragraphs는 이미 중복이 제거된 상태이므로 그대로 사용
        """
        print(f"🔧 Building context embedding index ({self.split_mode} mode)...")
        
        # 매핑 저장 (나중에 사용하기 위해)
        self.context_id_mapping = context_id_mapping
        
        # 캐시에서 로드 시도
        cached_embeddings, cached_context_ids, cached_faiss_index = self._load_embedding_cache()
        
        if cached_embeddings is not None:
            self.context_embeddings = cached_embeddings
            self.context_ids = cached_context_ids
            self.faiss_index = cached_faiss_index
            # 캐시 로드 시에도 매핑 설정 (이미 설정되어 있지만 명시적으로)
            # context_id_mapping은 이미 파라미터로 받아서 설정됨
            return
        
        # 캐시가 없으면 새로 생성
        # raw_paragraphs는 이미 중복이 제거된 상태이므로 그대로 사용
        print("🔄 Cache not found, generating new embeddings...")
        contexts = []
        context_ids = []
        
        for context_id, paragraph in tqdm(raw_paragraphs.items(), desc="Building embeddings"):
            contexts.append(paragraph['context'])
            context_ids.append(context_id)
        
        # 임베딩 생성
        print("🔄 Generating context embeddings...")
        embeddings = self.embedder.encode(contexts, show_progress_bar=True)
        
        # FAISS 인덱스 구축
        print("🔄 Building FAISS index...")
        dimension = embeddings.shape[1]
        faiss_index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        faiss_index.add(embeddings.astype('float32'))
        
        self.context_embeddings = embeddings
        self.context_ids = context_ids
        self.faiss_index = faiss_index
        
        # 캐시 저장
        self._save_embedding_cache(embeddings, context_ids, faiss_index)
        print(f"✅ Built index with {len(context_ids)} unique context embeddings")
    
    def find_similar_contexts_by_question(self, question: str, top_k: int = None) -> List[str]:
        """질문 임베딩을 사용하여 유사한 컨텍스트들 찾기"""
        if top_k is None:
            top_k = self.top_k_similar
        
        if self.faiss_index is None:
            return []
        
        # 질문 임베딩 생성
        question_embedding = self.embedder.encode([question])
        faiss.normalize_L2(question_embedding)
        
        # 유사도 검색
        scores, indices = self.faiss_index.search(question_embedding.astype('float32'), top_k)
        
        # 유사한 context_id 반환
        # self.context_ids는 이미 중복이 제거된 최종 context_id들이므로 그대로 사용
        similar_context_ids = []
        seen_context_ids = set()
        
        for i in indices[0]:
            if i < len(self.context_ids):
                context_id = self.context_ids[i]
                # 중복 제거 (같은 context_id가 여러 번 나올 수 있음)
                if context_id not in seen_context_ids:
                    similar_context_ids.append(context_id)
                    seen_context_ids.add(context_id)
        
        return similar_context_ids[:top_k]
    
    # ==================== 최종 데이터 생성 ====================
    
    def create_processed_data(self, qa_list: List[Dict], raw_paragraphs: Dict[str, Dict]) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
        """최종 전처리된 데이터 생성"""
        print(f"🔄 Creating processed data ({self.split_mode} mode)...")
        
        # 1. raw_paragraphs의 모든 context에 대해 context_mapping 초기화
        context_mapping = {}
        for context_id, paragraph in raw_paragraphs.items():
            context_mapping[context_id] = {
                'context': paragraph['context'],
                'doc_id': paragraph.get('doc_id', ''),
                'doc_source': paragraph.get('doc_source', ''),
                'doc_class_class': paragraph.get('doc_class_class', ''),
                'doc_class_code': paragraph.get('doc_class_code', ''),
                'q_id': []
            }
        
        # 2. QA 처리
        qid_data = {}
        skipped_count = 0
        duplicate_question_ids = 0
        
        for qa in tqdm(qa_list, desc="Creating processed instances"):
            question_id = qa['question_id']
            context_id = qa['context_id']  # 이미 매핑된 context_id
            
            # raw_data에서 해당 paragraph 확인
            if context_id not in raw_paragraphs:
                skipped_count += 1
                continue
            
            # question_id 중복 체크
            if question_id in qid_data:
                duplicate_question_ids += 1
                continue
            
            # 질문 기반으로 유사한 컨텍스트 찾기
            similar_context_ids = self.find_similar_contexts_by_question(qa['question'])
            
            # qid 데이터 생성
            qid_data[question_id] = {
                'question': qa['question'],
                'answer': qa['answer'],
                'clue_text': qa['clue_text'],
                'qa_type': qa.get('qa_type', ''),  # qa_type 저장
                'context_id': context_id,  # 이미 매핑된 context_id
                'similar_context_id': similar_context_ids,
                'doc_id': qa['doc_id'],
                'doc_source': qa['doc_source'],
                'doc_class_class': qa['doc_class_class'],
                'doc_class_code': qa['doc_class_code']
            }
            
            # context_mapping에 q_id 추가
            if context_id in context_mapping:
                context_mapping[context_id]['q_id'].append(question_id)
        
        print(f"✅ Created {len(qid_data)} question instances")
        print(f"✅ Created {len(context_mapping)} context mappings (all raw_paragraphs)")
        if skipped_count > 0:
            print(f"⚠️ Skipped {skipped_count} QAs (context_id not found in raw_paragraphs)")
        if duplicate_question_ids > 0:
            print(f"⚠️ Skipped {duplicate_question_ids} duplicate question_ids")
        
        return qid_data, context_mapping
    
    def save_processed_data(self, qid_data: Dict[str, Dict], context_mapping: Dict[str, Dict], output_path: str):
        """전처리된 데이터 저장"""
        print(f"💾 Saving processed data to {output_path}")
        
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        output_data = {
            'qid_data': qid_data,
            'context_mapping': context_mapping
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Saved {len(qid_data)} question instances and {len(context_mapping)} context mappings")
    
    def clear_cache(self):
        """임베딩 캐시 삭제"""
        import shutil
        if os.path.exists(self.embedding_cache_dir):
            shutil.rmtree(self.embedding_cache_dir)
            print(f"🗑️ Cleared embedding cache: {self.embedding_cache_dir}")
        else:
            print(f"ℹ️ No cache found at: {self.embedding_cache_dir}")
    
    # ==================== 메인 파이프라인 ====================
    
    def run_preprocessing(self, labeled_data_dir: str, raw_data_dir: str, output_path: str, clear_cache: bool = False):
        """전체 전처리 파이프라인 실행"""
        print(f"🚀 Starting data preprocessing ({self.split_mode} mode)...")
        
        if clear_cache:
            self.clear_cache()
        
        # 1. 데이터 로드
        labeled_data, raw_data = self.load_data(labeled_data_dir, raw_data_dir)
        
        # 2. Raw data 전처리 (문서 기준) - 먼저 처리
        print("\n" + "="*60)
        print("STEP 1: Processing Raw Data (Document-based)")
        print("="*60)
        raw_paragraphs, context_id_mapping = self.process_raw_data(raw_data)
        
        # 3. Labeled data 전처리 (QA 기준)
        print("\n" + "="*60)
        print("STEP 2: Processing Labeled Data (QA-based)")
        print("="*60)
        qa_list = self.process_labeled_data(labeled_data, context_id_mapping, raw_paragraphs)
        
        # 4. 컨텍스트 임베딩 인덱스 구축
        print("\n" + "="*60)
        print("STEP 3: Building Embedding Index")
        print("="*60)
        self.build_embedding_index(raw_paragraphs, context_id_mapping)
        
        # 5. 전처리된 데이터 생성
        print("\n" + "="*60)
        print("STEP 4: Creating Processed Data")
        print("="*60)
        qid_data, context_mapping = self.create_processed_data(qa_list, raw_paragraphs)
        
        # 6. 저장
        print("\n" + "="*60)
        print("STEP 5: Saving Processed Data")
        print("="*60)
        self.save_processed_data(qid_data, context_mapping, output_path)
        
        print("\n🎉 Preprocessing completed!")
        return qid_data, context_mapping


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Data preprocessing with embedding cache")
    parser.add_argument("--split", type=str, choices=["train", "test"], required=True,
                       help="Split mode: 'train' or 'test'")
    parser.add_argument("--clear-cache", action="store_true", 
                       help="Clear embedding cache before processing")
    parser.add_argument("--question-type", type=str, nargs="+", 
                       default=["span_extraction", "multiple_choice", "text_entailment", "span_extraction_how"],
                       help="Question types to process")
    parser.add_argument("--max-tokens", type=int, default=1000,
                       help="Maximum token length for paragraphs")
    parser.add_argument("--top-k", type=int, default=10,
                       help="Number of similar contexts to retrieve")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file path (default: data/processed_data_{split}.json)")
    parser.add_argument("--max-instances", type=int, default=None,
                       help="Maximum number of instances to process (for testing)")
    parser.add_argument("--train-sources", type=str, nargs="+",
                       default=["금융위원회", "한국은행", "한국금융연구원", "금융위원회, 서민금융진흥원", "금융위원회, 한국주택금융공사", "고용노동부"],
                       help="List of train sources")
    parser.add_argument("--test-sources", type=str, nargs="+",
                       default=["중소기업벤처부", "금융감독원", "한국소비자원"],
                       help="List of test sources")
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = f"data/processed_data_{args.split}.json"
    
    # 경로 설정
    data_root = "./data/aihub_data"
    training_dir = os.path.join(data_root, "Training")
    validation_dir = os.path.join(data_root, "Validation")
    
    labeled_data_dir = [
        os.path.join(training_dir, "labeled_data"),
        os.path.join(validation_dir, "labeled_data")
    ]
    raw_data_dir = [
        os.path.join(training_dir, "raw_data"),
        os.path.join(validation_dir, "raw_data")
    ]
    
    # 전처리기 초기화
    preprocessor = DataPreprocessor(
        embedding_model="nlpai-lab/KURE-v1",
        max_token_length=args.max_tokens,
        top_k_similar=args.top_k,
        max_instances=args.max_instances,
        question_types=args.question_type,
        split_mode=args.split,
        train_sources=args.train_sources,
        test_sources=args.test_sources
    )
    
    # 전처리 실행
    qid_data, context_mapping = preprocessor.run_preprocessing(
        labeled_data_dir=labeled_data_dir,
        raw_data_dir=raw_data_dir,
        output_path=args.output,
        clear_cache=args.clear_cache
    )
    
    # 최종 통계 출력
    print("\n" + "="*60)
    print("📊 Final Statistics")
    print("="*60)
    print(f"Total questions: {len(qid_data)}")
    print(f"Total contexts: {len(context_mapping)}")
    
    similar_counts = [len(item['similar_context_id']) for item in qid_data.values()]
    if similar_counts:
        print(f"Average similar contexts: {np.mean(similar_counts):.1f}")
        print(f"Max similar contexts: {max(similar_counts)}")
        print(f"Min similar contexts: {min(similar_counts)}")
    
    question_counts = [len(item['q_id']) for item in context_mapping.values()]
    if question_counts:
        print(f"Average questions per context: {np.mean(question_counts):.2f}")
        print(f"Max questions per context: {max(question_counts)}")


if __name__ == "__main__":
    main()
