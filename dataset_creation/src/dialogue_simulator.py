#!/usr/bin/env python3
"""
Dialogue Simulator: RAG 기반 대화 시뮬레이션
"""

import random
from typing import List, Dict, Any, Optional, Callable
from retriever import Retriever
from llm_components import (
    QuestionSelector,
    ReconfirmationGenerator,
    QuestionAnswerExtractor,
    QuestionRewriter,
    AnswerGenerator
)


class DialogueSimulator:
    """RAG 기반 대화 시뮬레이터"""
    
    def __init__(
        self,
        retriever: Retriever,
        question_selector: QuestionSelector,
        reconfirmation_generator: ReconfirmationGenerator,
        question_answer_extractor: QuestionAnswerExtractor,
        question_rewriter: QuestionRewriter,
        answer_generator: AnswerGenerator,
        qid_data: Dict[str, Dict[str, Any]],
        context_mapping: Dict[str, Dict[str, Any]],
        get_next_qid: Optional[Callable[[], str]] = None,
        preprocessor: Optional[Any] = None
    ):
        """
        Args:
            retriever: 검색기 인스턴스
            question_selector: 질문 선택 LLM
            reconfirmation_generator: 재확인 질문 생성 LLM
            question_answer_extractor: 질문/답변 추출 LLM
            question_rewriter: 질문 재작성 LLM
            answer_generator: 답변 생성 LLM
            qid_data: {qid: {question, answer, clue_text, context_id, similar_context_id, doc_id}}
            context_mapping: {context_id: {context, doc_id, q_id: [list]}}
            get_next_qid: 다음 qid를 반환하는 함수 (새 질문 생성 시 사용)
            preprocessor: DataPreprocessor 인스턴스 (retrieval을 위해)
        """
        self.retriever = retriever
        self.question_selector = question_selector
        self.reconfirmation_generator = reconfirmation_generator
        self.question_answer_extractor = question_answer_extractor
        self.question_rewriter = question_rewriter
        self.answer_generator = answer_generator
        self.qid_data = qid_data
        self.context_mapping = context_mapping
        self.get_next_qid = get_next_qid
        self.preprocessor = preprocessor
        
        # doc_id별 질문 매핑 생성 (빠른 조회를 위해)
        self._build_doc_id_to_questions()
    
    def _build_doc_id_to_questions(self):
        """doc_id별 질문 리스트 매핑 생성"""
        self.doc_id_questions = {}  # {doc_id: [{qid, question, context_id}, ...]}
        for qid, data in self.qid_data.items():
            doc_id = data.get('doc_id')
            if doc_id:
                if doc_id not in self.doc_id_questions:
                    self.doc_id_questions[doc_id] = []
                self.doc_id_questions[doc_id].append({
                    'qid': qid,
                    'question': data.get('question', ''),
                    'context_id': data.get('context_id', '')
                })
    
    def _get_similar_context_questions(self, current_qid: str) -> List[Dict[str, Any]]:
        """현재 qid의 유사한 context에서 나온 질문들 가져오기"""
        if current_qid not in self.qid_data:
            return []
        
        current_data = self.qid_data[current_qid]
        similar_context_ids = current_data.get('similar_context_id', [])
        
        similar_questions = []
        for context_id in similar_context_ids:
            if context_id in self.context_mapping:
                q_ids = self.context_mapping[context_id].get('q_id', [])
                for qid in q_ids:
                    if qid != current_qid and qid in self.qid_data:
                        similar_questions.append({
                            'qid': qid,
                            'question': self.qid_data[qid].get('question', ''),
                            'context_id': context_id,
                            'doc_id': self.qid_data[qid].get('doc_id', '')
                        })
        
        return similar_questions
    
    def _get_doc_id_questions(self, qid: str) -> List[Dict[str, Any]]:
        """같은 doc_id에 포함된 질문들 가져오기"""
        if qid not in self.qid_data:
            return []
        
        doc_id = self.qid_data[qid].get('doc_id')
        if not doc_id or doc_id not in self.doc_id_questions:
            return []
        
        # 현재 qid 제외
        return [
            q for q in self.doc_id_questions[doc_id]
            if q['qid'] != qid
        ]
    
    def _get_gold(self, qid: Optional[str]) -> Optional[Dict[str, Any]]:
        """
        질문의 context_id로부터 gold passage 준비
        
        Args:
            qid: 질문의 qid (None이면 None 반환)
            
        Returns:
            gold: 현재 질문의 context 또는 None
        """
        if not qid or qid not in self.qid_data:
            return None
        
        qid_data = self.qid_data[qid]
        context_id = qid_data.get('context_id')
        
        if context_id and context_id in self.context_mapping:
            gold_context = self.context_mapping[context_id]
            return {
                'context_id': context_id,
                'context': gold_context.get('context', ''),
                'doc_id': gold_context.get('doc_id', '')
            }
        
        return None
    
    def _get_topk(self, qid: Optional[str]) -> List[Dict[str, Any]]:
        """
        질문의 similar_context_ids로부터 topk passages 준비
        
        Args:
            qid: 질문의 qid (None이면 빈 리스트 반환)
            
        Returns:
            topk: similar contexts 리스트
        """
        topk = []
        
        if not qid or qid not in self.qid_data:
            return topk
        
        qid_data = self.qid_data[qid]
        similar_context_ids = qid_data.get('similar_context_id', [])
        
        for similar_context_id in similar_context_ids:
            if similar_context_id in self.context_mapping:
                similar_context = self.context_mapping[similar_context_id]
                topk.append({
                    'context_id': similar_context_id,
                    'context': similar_context.get('context', ''),
                    'doc_id': similar_context.get('doc_id', '')
                })
        
        return topk
    
    def simulate_conversation(
        self,
        seed_qid: str,
        min_turns: int = 4,
        max_turns: int = 8,
        evidence_text: Optional[str] = None,
        retrieval_mode: str = "gold"
    ) -> tuple:
        """
        seed question으로 시작하여 대화 시뮬레이션
        
        Args:
            seed_qid: 시작할 seed question의 qid
            min_turns: 최소 대화 턴 수
            max_turns: 최대 대화 턴 수
            evidence_text: 정답을 유추할 수 있는 evidence 텍스트 (옵션)
            retrieval_mode: "gold" 또는 "topk" (기본값: "gold")
            
        Returns:
            (dialogue_log, created_questions):
            - dialogue_log: 대화 기록 리스트
            - created_questions: 새로 생성된 질문-답변 리스트
        """
        if seed_qid not in self.qid_data:
            raise ValueError(f"Seed QID {seed_qid} not found in qid_data")
        
        conversation_history = []
        dialogue_log = []
        current_qid = seed_qid
        created_questions = []  # 새로 생성된 질문들 저장
        
        # Seed question으로 첫 턴 시작
        seed_data = self.qid_data[seed_qid]
        initial_question = seed_data.get('question', '')
        original_answer = seed_data.get('answer', '')  # 원본 답변 저장
        qa_type = seed_data.get('qa_type')  # qa_type 가져오기
        qid_history = [{'qid': seed_qid, 'question': initial_question}]  # 각 턴의 qid와 question 추적
        
        # 첫 질문 처리
        user_question = initial_question
        
        # retrieval_mode에 따라 필요한 것만 준비
        if retrieval_mode == "gold":
            gold = self._get_gold(seed_qid)
            topk = None
        elif retrieval_mode == "topk":
            gold = None
            topk = self._get_topk(seed_qid)
        else:
            raise ValueError(f"Invalid retrieval_mode: {retrieval_mode}. Must be 'gold' or 'topk'")
        
        passages = self.retriever.retrieve(
            user_question,
            conversation_history,
            gold=gold,
            topk=topk
        )
        
        # Evidence text 가져오기 (clue_text 사용)
        current_evidence = evidence_text or seed_data.get('clue_text', '')
        
        # 답변 생성
        assistant_answer = self.answer_generator.generate_answer(
            question=user_question,
            passages=passages,
            conversation_history=conversation_history,
            evidence_text=current_evidence
        )
        
        # 대화 히스토리 업데이트
        conversation_history.append({'role': 'user', 'content': user_question})
        conversation_history.append({'role': 'assistant', 'content': assistant_answer})
        
        # 로그 기록
        dialogue_log.append({
            'turn': 1,
            'qid': seed_qid,
            'original_question': initial_question,
            'original_answer': original_answer,
            'user_question': user_question,
            'assistant_answer': assistant_answer,
            'passages': passages,
            'action': 'seed_question',
            'clue_text': current_evidence,
            'qa_type': qa_type
        })
        
        # 총 턴 수 랜덤 샘플링 (min_turns ~ max_turns)
        total_turns = random.randint(min_turns, max_turns)
        
        # reconfirm 턴 랜덤 샘플링 (min_turns ~ total_turns)
        reconfirm_turn = random.randint(min_turns, total_turns)
        
        print(f"📊 Total turns: {total_turns}, Reconfirm turn: {reconfirm_turn}")
        
        # 다음 턴들 처리
        for turn in range(2, total_turns + 1):
            # 유사한 context 질문들과 같은 doc_id 질문들 가져오기
            similar_questions = self._get_similar_context_questions(current_qid)
            doc_questions = self._get_doc_id_questions(current_qid)
            
            # 이미 사용된 질문 제외 (qid_history에서 qid 추출)
            used_qids = {item['qid'] for item in qid_history if item.get('qid')}
            similar_questions = [q for q in similar_questions if q['qid'] not in used_qids]
            doc_questions = [q for q in doc_questions if q['qid'] not in used_qids]
            
            # 중복 제거: similar_questions를 우선하여 doc_questions에서 중복 제거
            similar_qids = {q['qid'] for q in similar_questions}
            doc_questions = [q for q in doc_questions if q['qid'] not in similar_qids]
            
            # reconfirm 턴인지 확인
            if turn == reconfirm_turn:
                action = 'reconfirm'
                selection_result = {'action': 'reconfirm', 'reason': 'Scheduled reconfirm turn'}
            else:
                # 다음 질문 선택 (select_question 또는 new_question만)
                selection_result = self.question_selector.select_next_question(
                    conversation_history=conversation_history,
                    similar_context_questions=similar_questions,
                    doc_id_questions=doc_questions,
                    current_qid=current_qid
                )
                action = selection_result['action']
            
            original_answer = None  # 원본 답변 초기화
            qa_type = None  # qa_type 초기화
            
            if action == 'select_question':
                # 기존 질문 선택
                selected_qid = selection_result.get('selected_qid')
                # 선택된 질문의 출처 추적
                if selected_qid:
                    similar_qids = {q['qid'] for q in similar_questions}
                    selection_result['question_source'] = 'similar_context_question' if selected_qid in similar_qids else 'same_doc_question'

                if selected_qid and selected_qid in self.qid_data and selected_qid not in used_qids:
                    selected_data = self.qid_data[selected_qid]
                    user_question = selected_data.get('question', '')
                    original_answer = selected_data.get('answer', '')  # 원본 답변 저장
                    qa_type = selected_data.get('qa_type')  # qa_type 가져오기
                    current_qid = selected_qid
                    current_evidence = selected_data.get('clue_text', '')
                else:
                    # QID를 찾지 못했거나 이미 사용된 경우 새 질문 생성
                    action = 'new_question'
            
            elif action == 'reconfirm':
                # 재확인할 질문 선택 및 재확인 질문 생성
                reconfirm_result = self.reconfirmation_generator.select_and_generate_reconfirmation(
                    conversation_history=conversation_history,
                    qid_history=qid_history
                )
                
                user_question = reconfirm_result['reconfirmation_question']
                selected_qid = reconfirm_result.get('selected_qid')
                
                if selected_qid and selected_qid in self.qid_data:
                    # 선택된 qid의 evidence 사용
                    current_qid = selected_qid
                    current_evidence = self.qid_data[selected_qid].get('clue_text', '')
                    original_answer = self.qid_data[selected_qid].get('answer', '')  # 원본 답변 저장
                    qa_type = self.qid_data[selected_qid].get('qa_type')  # qa_type 가져오기
                else:
                    # qid가 선택되지 않은 경우 (새로 생성된 질문이었거나)
                    # 가장 최근 질문의 evidence 재사용
                    current_qid = None
                    current_evidence = evidence_text or ''
                    qa_type = None  # 새로 생성된 질문이므로 qa_type 없음
            
            elif action == 'new_question':
                # 새 질문 생성 (현재 passage에서 추출)
                if passages:
                    # 가장 관련성 높은 passage 사용
                    paragraph = passages[0].get('context', '')
                    context_id = passages[0].get('context_id', '')
                    doc_id = passages[0].get('doc_id', '')
                    
                    extract_result = self.question_answer_extractor.extract_question_answer(
                        paragraph=paragraph,
                        conversation_history=conversation_history
                    )
                    user_question = extract_result.get('question', '')
                    original_answer = extract_result.get('answer', '')  # 원본 답변 저장
                    current_evidence = extract_result.get('reference_sentence', '')
                    
                    # 새로 생성된 질문에 qid 할당
                    if self.get_next_qid:
                        current_qid = self.get_next_qid()
                    else:
                        current_qid = None  # qid 생성 함수가 없으면 None
                    
                    # 유사한 컨텍스트 찾기 (retrieval 수행)
                    similar_context_ids = []
                    if self.preprocessor:
                        try:
                            similar_context_ids = self.preprocessor.find_similar_contexts_by_question(
                                user_question, 
                                top_k=10
                            )
                        except Exception as e:
                            print(f"⚠️ Warning: Failed to find similar contexts: {e}")
                            similar_context_ids = []
                    
                    # context_mapping에서 doc_source 등 추가 정보 가져오기
                    context_info = self.context_mapping.get(context_id, {})
                    
                    # 새로 생성된 질문을 qid_data에 동적으로 추가 (다음 턴에서 사용하기 위해)
                    if current_qid:
                        self.qid_data[current_qid] = {
                            'question': user_question,
                            'answer': extract_result.get('answer', ''),
                            'clue_text': current_evidence,
                            'qa_type': None,
                            'context_id': context_id,
                            'similar_context_id': similar_context_ids,
                            'doc_id': doc_id,
                            'doc_source': context_info.get('doc_source', ''),
                            'doc_class_class': context_info.get('doc_class_class', ''),
                            'doc_class_code': context_info.get('doc_class_code', '')
                        }
                        
                        # context_mapping에 q_id 추가
                        if context_id not in self.context_mapping:
                            self.context_mapping[context_id] = {
                                'context': paragraph,
                                'doc_id': doc_id,
                                'doc_source': context_info.get('doc_source', ''),
                                'doc_class_class': context_info.get('doc_class_class', ''),
                                'doc_class_code': context_info.get('doc_class_code', ''),
                                'q_id': []
                            }
                        
                        if current_qid not in self.context_mapping[context_id].get('q_id', []):
                            self.context_mapping[context_id].setdefault('q_id', []).append(current_qid)
                        
                        # doc_id_to_questions에도 추가
                        if doc_id:
                            if doc_id not in self.doc_id_questions:
                                self.doc_id_questions[doc_id] = []
                            self.doc_id_questions[doc_id].append({
                                'qid': current_qid,
                                'question': user_question,
                                'context_id': context_id
                            })
                    
                    # 새로 생성된 질문 저장
                    created_questions.append({
                        'question': user_question,
                        'answer': extract_result.get('answer', ''),
                        'reference_sentence': current_evidence,
                        'context': paragraph,
                        'context_id': context_id,
                        'doc_id': doc_id,
                        'turn': turn,
                        'qid': current_qid,  # 할당된 qid 저장
                        'similar_context_id': similar_context_ids  # 유사한 컨텍스트 ID 리스트
                    })
                else:
                    # Passage가 없으면 대화 종료
                    break
            
            # 원본 질문 저장 (재작성 전)
            original_question = user_question
            
            # 질문 재작성 (자연스럽게)
            user_question = self.question_rewriter.rewrite_question(
                question=user_question,
                conversation_history=conversation_history
            )
            
            # retrieval_mode에 따라 필요한 것만 준비
            if retrieval_mode == "gold":
                gold = self._get_gold(current_qid)
                topk = None
            elif retrieval_mode == "topk":
                gold = None
                topk = self._get_topk(current_qid)
            else:
                raise ValueError(f"Invalid retrieval_mode: {retrieval_mode}. Must be 'gold' or 'topk'")
            
            passages = self.retriever.retrieve(
                user_question,
                conversation_history,
                gold=gold,
                topk=topk
            )
            
            # 답변 생성
            assistant_answer = self.answer_generator.generate_answer(
                question=user_question,
                passages=passages,
                conversation_history=conversation_history,
                evidence_text=current_evidence
            )
            
            # 대화 히스토리 업데이트
            conversation_history.append({'role': 'user', 'content': user_question})
            conversation_history.append({'role': 'assistant', 'content': assistant_answer})
            
            # qid_history 업데이트 (재작성된 질문 저장)
            qid_history.append({'qid': current_qid, 'question': user_question})
            
            # 로그 기록
            log_entry = {
                'turn': turn,
                'qid': current_qid,
                'original_question': original_question,
                'original_answer': original_answer,
                'user_question': user_question,
                'assistant_answer': assistant_answer,
                'passages': passages,
                'action': action,
                'clue_text': current_evidence,
                'selection_reason': selection_result.get('reason', ''),
                'qa_type': qa_type
            }
            if action == 'select_question':
                log_entry['question_source'] = selection_result.get('question_source')
            dialogue_log.append(log_entry)
            
            # 목표 턴 수에 도달한 경우 종료 (턴 처리 후 체크)
            if turn >= total_turns:
                break
        
        return dialogue_log, created_questions

