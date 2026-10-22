#!/usr/bin/env python3
from __future__ import annotations
"""
Hallucination Injection Script
대화에 환각을 주입하는 스크립트

환각 유형:
1. Contradictory: 답변을 수정하여 문서 컨텍스트와 모순되게 만듦
2. Inconsistency: reconfirm 질문에서 이전 답변과 모순되게 만듦 (별도 타입)
3. Irrelevance: 답변을 수정하여 관련 없게 만듦
4. Unverifiable: 근거 없는 정보 추가
5. False Acceptance: 문서를 Similar 문서로 교체
6. False Refusal: 거절 답변 (태깅만, 답변 수정 없음)
"""

import json
import argparse
import random
import re
from typing import List, Dict, Any, Optional, Set
from qa_model import QAModel
from hallucination_prompts import HallucinationPrompts


class HallucinationInjector:
    """환각 주입 클래스"""
    
    # 환각 유형별 비율 (전체 환각 대비)
    # HALLUCINATION_RATIOS = {
    #     'contradictory': 0.30,
    #     'inconsistency': 0.10,
    #     'unverifiable': 0.20,
    #     'irrelevance': 0.20,
    #     'false_acceptance': 0.10,
    #     'false_refusal': 0.10,
    # }
    # false_refusal은 거절 대화 수로 자동 결정되며, 초과분은 irrelevance/inconsistency에서 차감
    HALLUCINATION_RATIOS = {
        'contradictory': 0.20,
        'inconsistency': 0.20,
        'unverifiable': 0.20,
        'irrelevance': 0.20,
        'false_acceptance': 0.10,
        'false_refusal': 0.10,
    }
    HALLUCINATION_TYPES = {
        'contradictory': {
            'description': '주어진 컨텍스트(문서와 이전 대화)와 충돌을 만드는 유형',
            'hint': '핵심 단어/한정어/수식어/숫자/전문금융용어 중 하나를 변경/누락하여 모순을 만들기 쉬운 턴',
        },
        'inconsistency': {
            'description': '이전 대화와 모순되도록 충돌을 만드는 유형',
            'hint': '이전 답변을 재확인하는 턴(reconfirm)에만 주입',
        },
        'unverifiable': {
            'description': '주어진 컨텍스트(문서와 이전 대화)에서 검증 불가한 내용을 추가하는 유형',
            'hint': '원본 답변이 문서 기반으로 구체적이며, 짧은 정보(절차/조건/예외)나 주관적 의견을 끼워 넣기 쉬운 턴'
        },
        'irrelevance': {
            'description': '질문 의도와 어긋나게 답변하도록 만드는 유형',
            'hint': '질문에 의문사(누가/무엇)가 있거나, 조건(대상/기간)이나 핵심 정보(금리/한도)를 묻는 턴'
        },
        'false_acceptance': {
            'description': '문서 조합을 변경하여 주어진 문서로 대답이 불가능한 쿼리로 만드는 유형',
            'hint': '대답만 놓고 봤을 때, LLM이 내부 지식으로 충분히 답할 수 있을 것 같은 턴'
        },
        'false_refusal': {
            'description': '이미 거절한 답변인데 실제로는 문서에 답이 있어서 답할 수 있었던 경우 (태깅만)',
            'hint': '이미 거절 답변("주어진 정보만으로는 대답 불가능합니다")이 있는 턴'
        }
    }
    
    # 거절 답변 패턴
    REFUSAL_PHRASES = [
        "주어진 정보만으로는 대답 불가능합니다",
        "주어진 정보만으로는 대답이 불가능합니다",
        "주어진 정보만으로는 답변이 어렵습니다",
    ]
    
    def __init__(self, provider: str = "openai", model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        """
        Args:
            provider: "openai" or "google"
            model: 모델 이름
            api_key: API 키
        """
        self.llm = QAModel(provider=provider, model=model, api_key=api_key)
    
    def determine_injection_positions(
        self,
        dialogue: List[Dict[str, Any]],
        hallucination_types: List[str],
        prioritize_turns: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """
        LLM을 사용하여 환각 주입 위치 결정
        
        Args:
            dialogue: 대화 턴 리스트
            hallucination_types: [hallucination_type, ...] - 이미 샘플링된 정확한 값
            prioritize_turns: 사용하지 않음 (하위 호환성 유지용)
            
        Returns:
            [{'turn_idx': int, 'hallucination_type': str}, ...]
        """
        # 대화 텍스트 생성
        dialogue_text = self._format_dialogue_text(dialogue)
        
        # reconfirm 턴 찾기 (선호 힌트용)
        reconfirm_turns = [i+1 for i, turn in enumerate(dialogue)
                          if turn.get('action') == 'reconfirm']

        # 제약/선호 조건 노트 생성
        constraint_note = self._get_constraint_note(hallucination_types, reconfirm_turns)

        # 환각 유형 정의/주입 가이드 생성 (LLM이 더 적절한 턴을 고르도록 보조)
        hallucination_definitions = self._build_hallucination_definitions(hallucination_types)
        
        # 프롬프트 생성 (이미 샘플링된 정확한 type/subtype 전달)
        prompt = HallucinationPrompts.get_injection_position_prompt(
            dialogue_text, hallucination_types, hallucination_definitions, constraint_note
        )
        
        messages = [
            {
                "role": "system",
                "content": HallucinationPrompts.get_injection_position_system_prompt()
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        response_text = self.llm.call_llm(messages, temperature=0.3)
        
        # JSON 파싱 및 fallback 처리
        return self._parse_injection_response_with_index(response_text, dialogue, hallucination_types)

    def _build_hallucination_definitions(
        self,
        hallucination_types: List[str],
    ) -> str:
        """
        샘플링된 환각 유형의 정의와 주입 위치 선정 힌트를 생성
        """
        lines = []
        for i, h_type in enumerate(hallucination_types, 1):
            type_info = self.HALLUCINATION_TYPES.get(h_type, {})
            desc = type_info.get('description', '')
            hint = type_info.get('hint', '')
            
            # 포맷팅
            if hint:
                lines.append(f"- {i}. {h_type}: {desc} / 힌트: {hint}")
            else:
                lines.append(f"- {i}. {h_type}: {desc}")
        
        return "\n".join(lines)
    
    def _assign_to_prioritized_turns(
        self,
        dialogue: List[Dict[str, Any]],
        hallucination_types: List[str],
        prioritize_turns: List[int]
    ) -> List[Dict[str, Any]]:
        """
        우선 순위 턴(is_correct=false)에 환각을 배정
        
        Args:
            dialogue: 대화 턴 리스트
            hallucination_types: 환각 유형 리스트
            prioritize_turns: 우선 배정해야 할 턴 인덱스 리스트
            
        Returns:
            injection 정보 리스트
        """
        # prioritize_turns를 복사해서 사용
        available_turns = prioritize_turns.copy()
        random.shuffle(available_turns)
        
        injections = []
        for h_type in hallucination_types:
            # 모든 유형은 available_turns에서 순서대로 배정
            if available_turns:
                turn_idx = available_turns.pop(0)
                injections.append({
                    'turn_idx': turn_idx,
                    'hallucination_type': h_type
                })
        
        return injections
    
    def _format_dialogue_text(self, dialogue: List[Dict[str, Any]]) -> str:
        """대화를 텍스트 형식으로 변환"""
        dialogue_text = ""
        for i, turn in enumerate(dialogue, 1):
            user_q = turn.get('user_question', '')
            assistant_a = turn.get('assistant_answer', '')
            dialogue_text += f"턴 {i}:\n질문: {user_q}\n답변: {assistant_a}\n\n"
        return dialogue_text
    
    def _get_constraint_note(
        self, 
        hallucination_types: List[str], 
        reconfirm_turns: List[int]
    ) -> str:
        """제약/선호 조건 노트 생성"""
        notes = []
        
        has_inconsistency = any(h_type == 'inconsistency' for h_type in hallucination_types)
        has_non_inconsistency = any(h_type != 'inconsistency' for h_type in hallucination_types)
        
        if has_inconsistency:
            if reconfirm_turns:
                notes.append(f"\n\n중요: \"Inconsistency\" 유형은 반드시 reconfirm 질문(action이 \"reconfirm\"인 턴)에만 주입해야 합니다. reconfirm 턴: {reconfirm_turns}")
            else:
                notes.append("\n\n중요: \"Inconsistency\" 유형이 있지만 reconfirm 질문이 없습니다. 이 경우 inconsistency는 일반 contradictory로 처리해야 합니다.")
        
        if has_non_inconsistency and reconfirm_turns:
            notes.append(f"\n\n중요: Inconsistency를 제외한 다른 모든 환각 유형은 reconfirm 턴(action이 \"reconfirm\"인 턴)에 주입할 수 없습니다. reconfirm 턴 {reconfirm_turns}은 피해야 합니다.")
        
        return "".join(notes)
    
    def _parse_injection_response_with_index(
        self, 
        response_text: str, 
        dialogue: List[Dict[str, Any]], 
        hallucination_types: List[str]
    ) -> List[Dict[str, Any]]:
        """
        LLM 응답에서 injection 정보 파싱 (순서 기반)
        
        Args:
            response_text: LLM 응답
            dialogue: 대화 턴 리스트
            hallucination_types: 샘플링된 정확한 환각 유형 리스트 (순서대로)
        """
        # JSON 파싱 시도
        try:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                turn_indices = result.get('turn_indices', [])
                
                # 턴 번호 리스트를 환각 유형과 순서대로 매핑
                if len(turn_indices) == len(hallucination_types):
                    # 거절 답변 턴 찾기
                    refusal_turns_set = set([
                        i + 1 for i, turn in enumerate(dialogue)
                        if self._is_refusal_answer(turn.get('assistant_answer', ''))
                    ])
                    reconfirm_turns = [
                        i + 1 for i, turn in enumerate(dialogue) 
                        if turn.get('action') == 'reconfirm' and (i + 1) not in refusal_turns_set
                    ]
                    refusal_turns = list(refusal_turns_set)

                    injections = []
                    # 중복 제거를 위한 set
                    used_turns = set()
                    
                    for turn_idx, h_type in zip(turn_indices, hallucination_types):
                        # 중복 턴 체크
                        if turn_idx in used_turns:
                            continue
                        
                        # false_refusal은 거절 답변 턴에만 배정 가능
                        if h_type == 'false_refusal':
                            if turn_idx not in refusal_turns_set:
                                # 거절 답변 턴이 아니면 다른 거절 턴 찾기
                                available_refusal = [t for t in refusal_turns if t not in used_turns]
                                if available_refusal:
                                    turn_idx = available_refusal[0]
                                else:
                                    # 거절 턴이 없으면 건너뛰기
                                    continue
                        
                        # 다른 환각은 거절 답변 턴에 배정 금지
                        elif turn_idx in refusal_turns_set:
                            # 거절 답변 턴이면 다른 일반 턴 찾기
                            available_turns = [
                                i + 1 for i in range(len(dialogue)) 
                                if (i + 1) not in used_turns and (i + 1) not in refusal_turns_set
                            ]
                            if available_turns:
                                turn_idx = available_turns[0]
                            else:
                                # 가용 턴이 없으면 건너뛰기
                                continue
                        
                        # inconsistency 외 다른 환각은 reconfirm 턴에 배정 금지
                        elif h_type != 'inconsistency' and turn_idx in reconfirm_turns:
                            # reconfirm 턴이면 non-reconfirm 턴으로 찾기
                            available_turns = [
                                i + 1 for i in range(len(dialogue))
                                if (i + 1) not in used_turns 
                                and (i + 1) not in refusal_turns_set
                                and (i + 1) not in reconfirm_turns
                            ]
                            if available_turns:
                                turn_idx = available_turns[0]
                            else:
                                # 가용 턴이 없으면 건너뛰기
                                continue
                            
                        # inconsistency는 반드시 reconfirm 턴이어야 함
                        if h_type == 'inconsistency':
                            if turn_idx not in reconfirm_turns:
                                if reconfirm_turns:
                                    # 가장 가까운 reconfirm으로 강제
                                    available_reconfirm = [t for t in reconfirm_turns if t not in used_turns]
                                    if available_reconfirm:
                                        turn_idx = available_reconfirm[0]
                                    else:
                                        turn_idx = reconfirm_turns[0]
                                else:
                                    # reconfirm이 없으면 contradictory로 다운그레이드
                                    h_type = 'contradictory'

                        used_turns.add(turn_idx)
                        injections.append({
                            'turn_idx': turn_idx,
                            'hallucination_type': h_type
                        })
                    return injections
        except Exception as e:
            print(f"JSON 파싱 실패: {e}")
        
        # 파싱 실패 시 랜덤 할당
        return self._random_assign_injections(dialogue, hallucination_types)
    
    def _random_assign_injections(
        self, 
        dialogue: List[Dict[str, Any]], 
        hallucination_types: List[str]
    ) -> List[Dict[str, Any]]:
        """파싱 실패 시 랜덤으로 injection 위치 할당"""
        # 거절 답변 턴 찾기
        refusal_turns_set = set([
            i + 1 for i, turn in enumerate(dialogue)
            if self._is_refusal_answer(turn.get('assistant_answer', ''))
        ])
        
        # reconfirm 턴 찾기 (거절 답변 제외)
        reconfirm_turns = [
            i + 1 for i, turn in enumerate(dialogue) 
            if turn.get('action') == 'reconfirm' and (i + 1) not in refusal_turns_set
        ]
        
        # 가용 턴 풀을 "한 번" 만들고, 뽑힌 턴은 항상 풀에서 제거해서 중복을 방지한다.
        # 중요: 거절 답변 턴은 false_refusal 전용이므로 다른 환각에서 제외
        reconfirm_pool = reconfirm_turns.copy()
        non_reconfirm_pool = [
            i + 1 for i, turn in enumerate(dialogue) 
            if turn.get('action') != 'reconfirm' and (i + 1) not in refusal_turns_set
        ]
        refusal_pool = list(refusal_turns_set)

        random.shuffle(reconfirm_pool)
        random.shuffle(non_reconfirm_pool)
        random.shuffle(refusal_pool)

        # 거절 답변 턴을 제외한 일반 턴 풀
        available_turns: List[int] = [t for t in range(1, len(dialogue) + 1) if t not in refusal_turns_set]
        random.shuffle(available_turns)

        used: Set[int] = set()
        injections: List[Dict[str, Any]] = []

        def _pick_from(candidates: List[int]) -> Optional[int]:
            for t in candidates:
                if t in used:
                    continue
                if t in available_turns:
                    return t
            return None

        for h_type in hallucination_types:
            chosen: Optional[int] = None

            if h_type == 'inconsistency':
                # inconsistency는 reconfirm 턴에서만
                chosen = _pick_from(reconfirm_pool)
                if chosen is None:
                    # reconfirm이 없으면 contradictory로 다운그레이드 후 non_reconfirm 턴
                    h_type = 'contradictory'
                    chosen = _pick_from(non_reconfirm_pool)

            elif h_type == 'false_refusal':
                # 요구사항: false_refusal은 "원래 거절 답변" 턴에만 태깅 (필수)
                chosen = _pick_from(refusal_pool)
                if chosen is None:
                    # refusal 턴이 없으면 이 환각은 건너뛰기 (환각 손실)
                    continue
            
            else:
                # 다른 모든 환각은 reconfirm 턴 제외
                chosen = _pick_from(non_reconfirm_pool)

            if chosen is None and h_type != 'false_refusal':
                # 마지막 수단: non-reconfirm 턴에서라도 배정
                for t in non_reconfirm_pool:
                    if t not in used:
                        chosen = t
                        break

            if chosen is not None:
                used.add(chosen)
                injections.append({
                    'turn_idx': chosen,
                    'hallucination_type': h_type
                })

        return injections
    
    def _is_refusal_answer(self, answer: Optional[str]) -> bool:
        """
        이미 거절 답변인지 여부 판단
        (예: "주어진 정보만으로는 대답 불가능합니다" 등)
        """
        if not answer:
            return False
        return any(phrase in answer for phrase in self.REFUSAL_PHRASES)
    
    def _parse_llm_response(self, response: str) -> tuple[str, str]:
        """
        LLM 응답에서 답변과 근거 파싱
        
        Args:
            response: LLM 응답 텍스트 (형식: "답변: [답변]\n근거: [근거]")
            
        Returns:
            (답변, 근거) 튜플
        """
        answer = ""
        reason = ""
        
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('답변:'):
                answer = line.replace('답변:', '').strip()
                # 대괄호 제거 (있는 경우)
                answer = answer.lstrip('[').rstrip(']').strip()
            elif line.startswith('근거:'):
                reason = line.replace('근거:', '').strip()
                # 대괄호 제거 (있는 경우)
                reason = reason.lstrip('[').rstrip(']').strip()
        
        # 파싱 실패 시 전체 응답을 답변으로 사용
        if not answer:
            answer = response.strip()
        
        return answer, reason
    
    def inject_contradictory(
        self,
        turn: Dict[str, Any],
        dialogue_history: List[Dict[str, str]]
    ) -> tuple[str, str]:
        """
        Contradictory 환각 주입
        
        Returns:
            (modified_answer, change_description)
        """
        original_answer = turn.get('assistant_answer', '')
        user_question = turn.get('user_question', '')
        passages = turn.get('passages', [])
        
        # 컨텍스트 준비
        passage_text = "\n\n".join([p.get('context', '') for p in passages])
        
        # 환각 정의 생성
        definition = self._get_hallucination_definition('contradictory')
        
        # 프롬프트 생성
        prompt = HallucinationPrompts.get_contradictory_prompt(
            user_question, original_answer, passage_text, definition
        )
        
        messages = [
            {
                "role": "system",
                "content": HallucinationPrompts.get_contradictory_system_prompt()
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        response = self.llm.call_llm(messages, temperature=0.7)
        modified_answer, reason = self._parse_llm_response(response)
        
        # 변경 내용 상세 설명
        change_description = f"원본 답변: {original_answer} → 수정된 답변: {modified_answer} (근거: {reason})"
        
        return modified_answer, change_description
    
    def _format_history(self, dialogue_history: List[Dict[str, str]]) -> str:
        """대화 히스토리를 텍스트로 포맷"""
        if not dialogue_history:
            return "없음"
        return "\n".join([
            f"{'사용자' if t['role'] == 'user' else '어시스턴트'}: {t['content']}"
            for t in dialogue_history[-3:]
        ])
    
    def _get_hallucination_definition(self, h_type: str) -> str:
        """환각 유형 정의 문자열 생성"""
        type_info = self.HALLUCINATION_TYPES.get(h_type, {})
        desc = type_info.get('description', '')
        return desc
    
    def inject_inconsistency(
        self,
        turn: Dict[str, Any],
        dialogue_history: List[Dict[str, str]]
    ) -> tuple[str, str]:
        """
        Inconsistency 환각 주입 (이전 답변과 모순)
        
        Returns:
            (modified_answer, change_description)
        """
        original_answer = turn.get('assistant_answer', '')
        user_question = turn.get('user_question', '')
        passages = turn.get('passages', [])
        
        # 컨텍스트 준비
        passage_text = "\n\n".join([p.get('context', '') for p in passages])
        history_text = self._format_history(dialogue_history)
        
        # 환각 정의 생성
        definition = self._get_hallucination_definition('inconsistency')
        
        # Inconsistency 프롬프트 사용
        prompt = HallucinationPrompts.get_contradictory_inconsistency_prompt(
            history_text, user_question, original_answer, passage_text, definition
        )
        
        messages = [
            {
                "role": "system",
                "content": HallucinationPrompts.get_contradictory_system_prompt()
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        response = self.llm.call_llm(messages, temperature=0.7)
        modified_answer, reason = self._parse_llm_response(response)
        
        # 변경 내용 상세 설명
        change_description = f"원본 답변: {original_answer} → 수정된 답변(이전 대화와 모순): {modified_answer} (근거: {reason})"
        
        return modified_answer, change_description
    
    def inject_irrelevance(
        self,
        turn: Dict[str, Any],
        dialogue_history: List[Dict[str, str]]
    ) -> tuple[str, str]:
        """
        Irrelevance 환각 주입 (subtype 없음)
        
        Returns:
            (modified_answer, change_description)
        """
        original_answer = turn.get('assistant_answer', '')
        user_question = turn.get('user_question', '')
        passages = turn.get('passages', [])
        
        passage_text = "\n\n".join([p.get('context', '') for p in passages])
        history_text = self._format_history(dialogue_history)
        
        # 환각 정의 생성
        definition = self._get_hallucination_definition('irrelevance')
        
        # 단일 프롬프트 사용
        prompt = HallucinationPrompts.get_irrelevance_prompt(
            history_text, user_question, original_answer, passage_text, definition
        )
        
        messages = [
            {
                "role": "system",
                "content": HallucinationPrompts.get_irrelevance_system_prompt()
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        response = self.llm.call_llm(messages, temperature=0.7)
        modified_answer, reason = self._parse_llm_response(response)
        
        # 변경 내용 상세 설명
        change_description = f"원본 답변: {original_answer} → 수정된 답변: {modified_answer} (근거: {reason})"
        
        return modified_answer, change_description
    
    def inject_unverifiable(
        self,
        turn: Dict[str, Any],
        dialogue_history: List[Dict[str, str]]
    ) -> tuple[str, str]:
        """
        Unverifiable 환각 주입 (subtype 없음, 근거 없는 정보 추가)
        
        Returns:
            (modified_answer, change_description)
        """
        original_answer = turn.get('assistant_answer', '')
        user_question = turn.get('user_question', '')
        passages = turn.get('passages', [])
        
        passage_text = "\n\n".join([p.get('context', '') for p in passages])
        
        # 환각 정의 생성
        definition = self._get_hallucination_definition('unverifiable')
        
        # 단일 프롬프트 사용 (subtype 없음)
        prompt = HallucinationPrompts.get_unverifiable_prompt(
            user_question, original_answer, passage_text, definition
        )
        
        messages = [
            {
                "role": "system",
                "content": HallucinationPrompts.get_unverifiable_system_prompt()
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        response = self.llm.call_llm(messages, temperature=0.7)
        modified_answer, reason = self._parse_llm_response(response)
        
        # 변경 내용 상세 설명
        change_description = f"원본 답변: {original_answer} → 수정된 답변(근거 없는 정보 추가): {modified_answer} (근거: {reason})"
        
        return modified_answer, change_description
    
    def inject_false_acceptance(
        self,
        turn: Dict[str, Any],
        similar_passages: List[Dict[str, Any]]
    ) -> tuple[List[Dict[str, Any]], str]:
        """
        False Acceptance 환각 주입 (Positive 문서를 Similar 문서로 교체)
        
        Args:
            turn: 현재 턴
            similar_passages: 유사한 문서 리스트 (사용 가능한 경우)
            
        Returns:
            (modified_passages, change_description)
        """
        original_passages = turn.get('passages', [])
        
        if not original_passages:
            return original_passages, "Passage가 없어 변경하지 않음"
        
        if similar_passages:
            # 유사한 문서가 제공된 경우: 교체
            modified_passages = original_passages.copy()
            replacement = random.choice(similar_passages)
            original_context_id = modified_passages[0].get('context_id', '')
            modified_passages[0] = replacement
            modified_context_id = replacement.get('context_id', '')
            change_description = f"Positive 문서(context_id: {original_context_id})를 Similar 문서(context_id: {modified_context_id})로 교체"
            return modified_passages, change_description
        else:
            # 유사한 문서가 없는 경우: placeholder로 표시
            modified_passages = original_passages.copy()
            original_context_id = modified_passages[0].get('context_id', '')
            
            # Placeholder 문서 생성
            placeholder_passage = {
                'context_id': f"{original_context_id}_similar_placeholder",
                'context': "[TODO] 이 위치에 정답이 아닌 유사 문서로 교체 예정",
                'doc_id': modified_passages[0].get('doc_id', '')
            }
            
            modified_passages[0] = placeholder_passage
            change_description = f"Positive 문서(context_id: {original_context_id})를 Similar 문서로 교체 예정 (현재 placeholder)"
            
            return modified_passages, change_description
    
    def inject_false_refusal(
        self,
        turn: Dict[str, Any],
        dialogue_history: List[Dict[str, str]]
    ) -> tuple[str, str]:
        """
        False Refusal 환각 태깅 (이미 거절한 답변인데 실제로는 답할 수 있었던 경우)
        
        주의: 이 함수는 답변을 변경하지 않고, 이미 거절한 답변을 false_refusal로 태깅만 함
        
        Returns:
            (original_answer, change_description)
        """
        original_answer = turn.get('assistant_answer', '')

        # 이미 거절 답변인지 확인
        if not self._is_refusal_answer(original_answer):
            raise ValueError(
                f"False Refusal은 이미 거절한 답변에만 적용 가능합니다. "
                f"현재 답변: {original_answer}"
            )

        # 답변을 변경하지 않고 그대로 사용하며, false_refusal로만 태깅
        change_description = (
            f"원본 답변이 이미 거절 답변이므로 그대로 사용하고 "
            f"false_refusal로 태깅함 (답변: {original_answer})"
        )
        
        return original_answer, change_description
    
    def inject_hallucination(
        self,
        turn: Dict[str, Any],
        dialogue_history: List[Dict[str, str]],
        hallucination_type: str,
        similar_passages: Optional[List[Dict[str, Any]]] = None
    ) -> tuple[Dict[str, Any], str]:
        """
        환각 주입 실행
        
        Args:
            turn: 현재 턴
            dialogue_history: 대화 히스토리
            hallucination_type: 환각 유형
            similar_passages: 유사 문서 리스트 (False Acceptance용)
        
        Returns:
            (modified_turn, change_description) 튜플
        """
        modified_turn = turn.copy()
        change_description = ""
        
        if hallucination_type == 'contradictory':
            modified_answer, change_description = self.inject_contradictory(
                turn, dialogue_history
            )
            modified_turn['assistant_answer'] = modified_answer

        elif hallucination_type == 'inconsistency':
            # inconsistency는 reconfirm 맥락에서 이전 답변과 모순되도록 수정
            modified_answer, change_description = self.inject_inconsistency(
                turn, dialogue_history
            )
            modified_turn['assistant_answer'] = modified_answer
        
        elif hallucination_type == 'irrelevance':
            modified_answer, change_description = self.inject_irrelevance(
                turn, dialogue_history
            )
            modified_turn['assistant_answer'] = modified_answer
        
        elif hallucination_type == 'unverifiable':
            modified_answer, change_description = self.inject_unverifiable(
                turn, dialogue_history
            )
            modified_turn['assistant_answer'] = modified_answer
        
        elif hallucination_type == 'false_acceptance':
            modified_passages, change_description = self.inject_false_acceptance(
                turn, similar_passages or []
            )
            modified_turn['passages'] = modified_passages
        
        elif hallucination_type == 'false_refusal':
            modified_answer, change_description = self.inject_false_refusal(
                turn, dialogue_history
            )
            modified_turn['assistant_answer'] = modified_answer
        
        return modified_turn, change_description


def process_dialogue(
    dialogue: List[Dict[str, Any]],
    injector: HallucinationInjector,
    similar_passages_map: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    dialogue_id: Optional[int] = None,
    assigned_hallucinations: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    하나의 대화에 환각 주입
    
    Args:
        dialogue: 대화 턴 리스트
        injector: HallucinationInjector 인스턴스
        similar_passages_map: {context_id: [similar_passages]} 매핑 (False Acceptance용)
        dialogue_id: 대화 ID
        assigned_hallucinations: 이 대화에 배정된 환각 유형 리스트 [type, ...]
        
    Returns:
        환각이 주입된 대화 턴 리스트 (원본 구조 유지, assistant_answer만 수정, 필드 추가)
    """
    num_turns = len(dialogue)
    assigned_hallucinations = assigned_hallucinations or []
    
    # 모든 거절 답변 턴 찾기
    refusal_turns_set = set([
        i + 1 for i, turn in enumerate(dialogue)
        if injector._is_refusal_answer(turn.get('assistant_answer', ''))
    ])
    
    # 1. 배정된 환각이 없으면 원본 구조 유지하고 필드만 추가
    if not assigned_hallucinations:
        # 첫 번째 거절 턴만 false_refusal로, 나머지는 null로
        output_turns = []
        first_refusal_tagged = False
        for turn_idx, turn in enumerate(dialogue, 1):
            is_refusal = turn_idx in refusal_turns_set

            # 원본 턴 복사
            output_turn = turn.copy()
            # 첫 번째 거절 턴만 태깅
            if is_refusal and not first_refusal_tagged:
                output_turn['hallucination_type'] = 'false_refusal'
                output_turn['change_description'] = f"거절 답변을 false_refusal로 태깅 (답변: {turn.get('assistant_answer', '')})"
                first_refusal_tagged = True
            else:
                output_turn['hallucination_type'] = None
                output_turn['change_description'] = None

            output_turns.append(output_turn)
        return output_turns
    
    # 2. 주입 위치 결정
    injection_positions = injector.determine_injection_positions(
        dialogue, 
        assigned_hallucinations,
        prioritize_turns=None
    )

    # 3. injection_positions 후처리:
    # - 턴 중복 제거/재배정 (중복이면 실제 주입 개수가 줄어드는 문제 방지)
    # - inconsistency는 항상 reconfirm 턴으로 강제
    # - inconsistency 외 다른 환각은 reconfirm 턴에 배정 금지
    # - false_refusal은 거절 답변 턴에만 할당
    # - 거절 답변 턴은 false_refusal 전용이므로 다른 환각 배정에서 제외
    reconfirm_turns = [
        i + 1 for i, turn in enumerate(dialogue) 
        if turn.get('action') == 'reconfirm' and (i + 1) not in refusal_turns_set
    ]
    refusal_turns = list(refusal_turns_set)
    # non-reconfirm 턴만 포함 (거절 답변 제외)
    non_reconfirm_turns = [
        t for t in range(1, len(dialogue) + 1) 
        if t not in refusal_turns_set and t not in reconfirm_turns
    ]

    used_turns: Set[int] = set()
    fixed_positions: List[Dict[str, Any]] = []

    def _first_available(preferred: List[int]) -> Optional[int]:
        for t in preferred:
            if t not in used_turns:
                return t
        return None

    for inj in injection_positions:
        h_type = inj.get('hallucination_type')
        turn_idx = inj.get('turn_idx')

        # false_refusal 제약: 거절 답변 턴에만 태깅 가능 (필수)
        if h_type == 'false_refusal':
            if turn_idx not in refusal_turns:
                # 현재 턴이 거절 답변이 아니면 다른 거절 턴 찾기
                alt = _first_available(refusal_turns)
                if alt is not None:
                    turn_idx = alt
                else:
                    # 거절 턴이 없으면 이 환각은 건너뛰기
                    continue
        
        # inconsistency 제약: reconfirm 턴에만 배정 가능
        elif h_type == 'inconsistency':
            if turn_idx not in reconfirm_turns:
                alt = _first_available(reconfirm_turns)
                if alt is not None:
                    turn_idx = alt
                else:
                    # reconfirm이 없으면 contradictory로 다운그레이드하고 non-reconfirm 턴에 배정
                    h_type = 'contradictory'
                    if turn_idx in reconfirm_turns or turn_idx in refusal_turns_set:
                        alt = _first_available(non_reconfirm_turns)
                        if alt is not None:
                            turn_idx = alt
                        else:
                            continue
        
        # 다른 환각은 거절 답변 턴 및 reconfirm 턴에 배정 금지
        else:
            if turn_idx in refusal_turns_set or turn_idx in reconfirm_turns:
                # non-reconfirm 턴으로 재배정
                alt = _first_available(non_reconfirm_turns)
                if alt is not None:
                    turn_idx = alt
                else:
                    # 가용 턴이 없으면 건너뛰기
                    continue

        # 턴 중복 방지: 이미 사용된 턴이면 유형별로 가능한 후보에서 재배정
        if turn_idx in used_turns:
            if h_type == 'inconsistency':
                alt = _first_available(reconfirm_turns)
                if alt is None:
                    # reconfirm이 없으면 contradictory로 다운그레이드하고 non-reconfirm 턴에 배정
                    h_type = 'contradictory'
                    alt = _first_available(non_reconfirm_turns)
            elif h_type == 'false_refusal':
                alt = _first_available(refusal_turns)
                if alt is None:
                    # 거절 답변 턴이 없으면 이 환각은 건너뛰기
                    continue
            else:
                alt = _first_available(non_reconfirm_turns)
            
            if alt is not None:
                turn_idx = alt
            else:
                # 최후의 수단: 아직 사용되지 않은 첫 번째 non-reconfirm 턴에 강제 배정 (환각 손실 방지)
                # false_refusal은 이미 위에서 continue로 건너뛰었으므로 여기 도달하지 않음
                for t in non_reconfirm_turns:
                    if t not in used_turns:
                        turn_idx = t
                        break
                else:
                    # 정말 가용 턴이 없으면 건너뛰기
                    continue

        used_turns.add(turn_idx)
        fixed_positions.append({
            'turn_idx': turn_idx,
            'hallucination_type': h_type
        })

    injection_positions = fixed_positions
    
    # 대화 히스토리 생성
    dialogue_history = []
    output_turns = []
    
    for turn_idx, turn in enumerate(dialogue, 1):
        # 이 턴에 환각 주입이 필요한지 확인
        injection = None
        for inj in injection_positions:
            if inj['turn_idx'] == turn_idx:
                injection = inj
                break
        
        # 원본 턴 복사 (모든 필드 유지)
        output_turn = turn.copy()
        
        if injection:
            # 환각 주입
            hallucination_type = injection['hallucination_type']
            
            # False Acceptance의 경우 similar passages 가져오기
            similar_passages = None
            if hallucination_type == 'false_acceptance' and similar_passages_map:
                # 현재 turn의 passage context_id로 similar passages 찾기
                original_passages = turn.get('passages', [])
                if original_passages:
                    context_id = original_passages[0].get('context_id')
                    if context_id in similar_passages_map:
                        similar_passages = similar_passages_map[context_id]
            
            modified_turn, change_description = injector.inject_hallucination(
                turn, dialogue_history, hallucination_type, similar_passages
            )
            
            # assistant_answer는 원본 유지, hallucinated_answer에 수정본 저장
            hallucinated_answer = modified_turn.get('assistant_answer', turn.get('assistant_answer', ''))
            output_turn['hallucinated_answer'] = hallucinated_answer
            # assistant_answer는 원본 그대로 유지 (이미 turn.copy()에서 복사됨)
            
            # passages는 직접 수정 (False Acceptance의 경우)
            output_turn['passages'] = modified_turn.get('passages', turn.get('passages', []))
            
            # hallucination 관련 필드 추가
            output_turn['hallucination_type'] = hallucination_type
            output_turn['change_description'] = change_description
            
            # 히스토리 업데이트 (할루시네이션된 답변 사용 - 이후 대화가 이를 기반으로 진행됨)
            dialogue_history.append({'role': 'user', 'content': output_turn.get('user_question', '')})
            dialogue_history.append({'role': 'assistant', 'content': hallucinated_answer})
        else:
            # 환각이 없는 경우
            # 거절 답변이면 자동으로 false_refusal 태깅
            is_refusal = turn_idx in refusal_turns_set
            
            # hallucination 관련 필드 추가
            output_turn['hallucination_type'] = 'false_refusal' if is_refusal else None
            output_turn['change_description'] = f"거절 답변을 false_refusal로 태깅 (답변: {turn.get('assistant_answer', '')})" if is_refusal else None
            
            # 히스토리 업데이트 (원본 답변 사용)
            dialogue_history.append({'role': 'user', 'content': output_turn.get('user_question', '')})
            dialogue_history.append({'role': 'assistant', 'content': output_turn.get('assistant_answer', '')})
        
        output_turns.append(output_turn)
    
    return output_turns


def generate_hallucinations(
    hallucination_counts: Dict[str, int]
) -> List[str]:
    """
    환각 유형을 생성
    
    Args:
        hallucination_counts: {hallucination_type: count} 딕셔너리
        
    Returns:
        [hallucination_type, ...] 리스트
    """
    all_hallucinations = []
    for h_type, count in hallucination_counts.items():
        for _ in range(count):
            all_hallucinations.append(h_type)
    
    return all_hallucinations


def main():
    parser = argparse.ArgumentParser(description="Inject hallucinations into dialogue results")
    parser.add_argument(
        '--input',
        type=str,
        default='dialogue_results.json',
        help='Input dialogue results JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output.json',
        help='Output JSON file with hallucinations'
    )
    parser.add_argument(
        '--provider',
        type=str,
        default='openai',
        choices=['openai', 'google'],
        help='LLM provider'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o-mini',
        help='LLM model name'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='API key (if not provided, will use environment variable)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of conversations/dialogues to process (for debugging)'
    )
    
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
    
    print(f"Loading dialogue results from {args.input}...")
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    injector = HallucinationInjector(
        provider=args.provider,
        model=args.model,
        api_key=args.api_key
    )
    
    # 데이터 구조에 따라 처리
    all_output_turns = []
    
    if 'conversations' in data:
        # dialogue_results.json 형식
        conversations = data.get('conversations', [])
        if args.max_samples is not None:
            conversations = conversations[:args.max_samples]
        print(f"Processing {len(conversations)} conversations...")

        # 0. reconfirm 턴에 거절 답변이 있는 대화 제거
        filtered_conversations = []
        removed_count = 0
        for conv in conversations:
            dialogue = conv.get('dialogue', [])
            has_refusal_reconfirm = any(
                turn.get('action') == 'reconfirm' and injector._is_refusal_answer(turn.get('assistant_answer', ''))
                for turn in dialogue
            )
            if has_refusal_reconfirm:
                removed_count += 1
            else:
                filtered_conversations.append(conv)
        if removed_count > 0:
            print(f"⚠️ Removed {removed_count} dialogues with refusal answers on reconfirm turns")
        conversations = filtered_conversations

        # 1. 대화 정보 수집 + 거절 대화 분리
        dialogue_info = []
        refusal_dialogue_ids = set()
        for conv_idx, conversation in enumerate(conversations, 1):
            dialogue = conversation.get('dialogue', [])
            conv_dialogue_id = conversation.get('dialogue_id', conv_idx)
            dialogue_info.append((conv_dialogue_id, dialogue))
            if any(injector._is_refusal_answer(turn.get('assistant_answer', '')) for turn in dialogue):
                refusal_dialogue_ids.add(conv_dialogue_id)

        # 거절 대화는 배정 없이 자동 false_refusal 태깅 (process_dialogue에서 처리)
        dialogue_hallucinations = {}
        for d_id in refusal_dialogue_ids:
            dialogue_hallucinations[d_id] = []

        # 거절 없는 대화만 환각 배정 대상
        injectable_dialogue_ids = [d_id for d_id, _ in dialogue_info if d_id not in refusal_dialogue_ids]
        num_injectable = len(injectable_dialogue_ids)

        print(f"Total dialogues: {len(dialogue_info)}")
        print(f"  Dialogues with refusal (auto false_refusal): {len(refusal_dialogue_ids)}")
        print(f"  Dialogues for hallucination injection: {num_injectable}")

        # 2. 전체 대화 수 기준 비율 계산
        num_total = len(dialogue_info)
        num_refusal = len(refusal_dialogue_ids)
        hallucination_counts: Dict[str, int] = {}
        for h_type, ratio in injector.HALLUCINATION_RATIOS.items():
            hallucination_counts[h_type] = int(num_total * ratio)

        # false_refusal은 실제 거절 대화 수로 대체
        ideal_false_refusal = hallucination_counts.pop('false_refusal', 0)
        excess = num_refusal - ideal_false_refusal  # 양수면 거절이 비율보다 많음

        # 초과분을 irrelevance, inconsistency에서 균등 차감
        if excess != 0:
            half = excess // 2
            remainder = excess - half
            hallucination_counts['irrelevance'] -= half
            hallucination_counts['inconsistency'] -= remainder
            print(f"  false_refusal 조정: 비율={ideal_false_refusal}, 실제={num_refusal}, 차이={excess}")
            print(f"    → irrelevance에서 {half} 차감, inconsistency에서 {remainder} 차감")

        # 반올림 오차 보정 (5개 타입 합 = injectable 수)
        current_sum = sum(hallucination_counts.values())
        diff = num_injectable - current_sum
        if diff != 0:
            max_type = max(hallucination_counts, key=hallucination_counts.get)
            hallucination_counts[max_type] += diff
            print(f"  (반올림 보정: {max_type}에 {diff} 추가)")

        print(f"  Injectable type counts:")
        for h_type, count in hallucination_counts.items():
            print(f"    {h_type}: {count}")

        # 음수 검증
        for h_type, count in hallucination_counts.items():
            if count < 0:
                raise ValueError(f"{h_type} count is negative ({count}). Too many refusal dialogues.")

        # 3. 환각 생성 및 셔플
        all_hallucinations = generate_hallucinations(hallucination_counts)
        random.shuffle(all_hallucinations)

        # 4. 거절 없는 대화에 1:1 배정
        random.shuffle(injectable_dialogue_ids)
        for d_id, h_type in zip(injectable_dialogue_ids, all_hallucinations):
            dialogue_hallucinations[d_id] = [h_type]

        # 분배 결과 출력
        total_assigned = sum(len(h) for h in dialogue_hallucinations.values())
        print(f"\nHallucinations distributed: {total_assigned} injectable + {num_refusal} auto false_refusal = {total_assigned + num_refusal} total")
        
        # 7. 각 대화 처리 (원본 구조 유지)
        output_conversations = []
        for conv_idx, conversation in enumerate(conversations, 1):
            print(f"Processing conversation {conv_idx}/{len(conversations)}...")
            dialogue = conversation.get('dialogue', [])
            conv_dialogue_id = conversation.get('dialogue_id', conv_idx)
            
            assigned = dialogue_hallucinations.get(conv_dialogue_id, [])
            
            # TODO: similar_passages_map을 실제로 로드해야 함 (현재는 None)
            modified_dialogue = process_dialogue(
                dialogue,
                injector,
                similar_passages_map=None,
                dialogue_id=conv_dialogue_id,
                assigned_hallucinations=assigned,
            )
            
            # 원본 conversation 구조 복사하고 dialogue만 교체
            output_conversation = conversation.copy()
            output_conversation['dialogue'] = modified_dialogue
            output_conversations.append(output_conversation)
    
    else:
        raise ValueError("Unknown data format. Expected 'conversations' key.")
    
    # 최종 출력 형식 (원본 구조 유지)
    output_data = data.copy()
    output_data['conversations'] = output_conversations
    
    print(f"Processed {len(output_conversations)} conversations")
    print(f"\nSaving results to {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print("\nHallucination injection completed!")


if __name__ == '__main__':
    main()

