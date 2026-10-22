#!/usr/bin/env python3
"""
Hallucination Evaluation Script
할루시네이션이 주입된 답변을 평가합니다:
- 적절성 (Appropriateness) (1-3): 지시된 할루시네이션 유형이 정의에 맞게 정확히 구현되었는가?
- 자연스러움 (Naturalness) (1-3): 할루시네이션이 문맥과 자연스럽게 어울리는가?
"""

import json
import argparse
import re
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from qa_model import QAModel


class HallucinationEvaluator:
    """할루시네이션 평가 클래스"""
    
    def __init__(self, provider: str = "openai", model: str = "gpt-4o", api_key: Optional[str] = None):
        """
        Args:
            provider: "openai" or "google"
            model: 모델 이름
            api_key: API 키 (없으면 환경변수에서 가져옴)
        """
        self.llm = QAModel(provider=provider, model=model, api_key=api_key)
    
    def evaluate_appropriateness(
        self,
        query: str,
        hallucinated_answer: str,
        hallucination_type: str,
        original_answer: str,
        documents: List[Dict[str, Any]]
    ) -> int:
        """
        할루시네이션 주입 적절성 평가 (1-3)
        지시된 할루시네이션 유형이 정의에 맞게 정확히 구현되었는가?
        
        Args:
            query: 사용자 질문
            hallucinated_answer: 할루시네이션이 주입된 답변
            hallucination_type: 지시된 할루시네이션 유형
            original_answer: 원본 정답
            documents: 관련 문서
        
        Returns:
            1, 2, 또는 3 (높을수록 좋음)
        """
        # 문서 텍스트 추출
        doc_texts = []
        for doc in documents:
            doc_texts.append(f"[문서 {doc.get('context_id', '')}]\n{doc.get('context', '')}")
        documents_text = "\n\n".join(doc_texts)
        
        # 할루시네이션 유형별 정의
        # NOTE:
        # - inject_hallucinations.py에서 생성되는 taxonomy(contradictory/irrelevance/unverifiable/false_acceptance/false_refusal + subtypes)만 지원합니다.
        # - 알 수 없는 hallucination_type이 들어오면 조용히 넘어가지 않고 에러를 발생시켜(빠르게 실패) 데이터/파이프라인 문제를 드러냅니다.

        def _parse_injected_type(ht: str) -> tuple[str, Optional[str]]:
            """
            injected hallucination_type 예:
              - contradictory
              - inconsistency
              - unverifiable
              - irrelevance
              - false_acceptance
              - false_refusal
            """
            if not ht:
                return "", None
            # base type에 underscore가 포함된 케이스를 먼저 처리
            for base in ("false_acceptance", "false_refusal", "true_refusal"):
                if ht == base:
                    return base, None
            if ht.startswith("contradictory_"):
                return "contradictory", ht[len("contradictory_"):]
            if ht.startswith("unverifiable_"):
                return "unverifiable", ht[len("unverifiable_"):]
            return ht, None

        base_type, subtype = _parse_injected_type(hallucination_type)

        injected_definitions = {
            "contradictory": "답변을 최소 수정하여 (문서/원본 정답/이전 대화와) 충돌·모순되게 만드는 유형",
            "inconsistency": "재확인(reconfirm) 맥락에서 이전 답변과 모순되도록 수정하는 유형",
            "irrelevance": "질문 의도와 어긋나게 관련 없는 내용으로 답하도록 만드는 유형",
            "unverifiable": "문서에 근거가 없는 정보를 추가하거나(또는 주관적 의견을) 사실처럼 제시하는 유형",
            "false_acceptance": "관련 문서(근거)를 유사 문서로 바꿔 답변이 문서로부터 지지받지 않게(또는 잘못 지지받게) 만드는 유형",
            "false_refusal": "답할 수 있는 질문인데도 고정된 거절 답변(예: '주어진 정보만으로는...')으로 바꾸는 유형",
        }

        injected_subtype_definitions = {
            # contradictory subtypes
            "contradictory": {
                "inconsistency": "재확인(reconfirm) 맥락에서 이전 답변과 모순되도록 수정",
                "financial_term_misunderstanding": "전문 금융 용어를 잘못 이해한 것처럼 다른 용어로 바꾸어 의미를 왜곡",
                "modifier_change": "한정어/수식어(예: 최소/최대/이상/이하 등)를 바꿔 조건을 뒤집거나 왜곡",
                "number_error": "기간/금리/한도 등 핵심 숫자를 다른 숫자로 바꿔 답변을 틀리게 만듦",
            },
            # unverifiable subtypes
            "unverifiable": {
                "fabricated_detail": "문서에 없는 구체적 디테일(절차/예외/조건/기간 등)을 그럴듯하게 추가",
                "subjective_opinion": "문서에 없는 주관적 평가/추천을 객관적 사실처럼 섞어 넣음",
            },
        }

        hal_def = ""
        if base_type in injected_definitions:
            hal_def = injected_definitions[base_type]
            if subtype:
                sub_map = injected_subtype_definitions.get(base_type, {})
                sub_def = sub_map.get(subtype)
                if sub_def:
                    hal_def = f"{hal_def} (subtype: {subtype} - {sub_def})"
        else:
            raise ValueError(
                f"Unknown hallucination_type='{hallucination_type}'. "
                "Expected inject_hallucinations.py taxonomy (e.g., contradictory, inconsistency, irrelevance, unverifiable, false_acceptance, false_refusal; "
                "also supports legacy forms like contradictory_* or unverifiable_*)."
            )
        
        prompt = f"""다음 정보가 주어집니다:

질문: {query}

원본 정답: {original_answer}

할루시네이션이 주입된 답변: {hallucinated_answer}

지시된 할루시네이션 유형: {hallucination_type}
정의: {hal_def}

관련 문서:
{documents_text}

지시된 할루시네이션 유형({hallucination_type})이 답변에 정의에 맞게 정확히 구현되었는지 평가하세요. (점수가 높을수록 좋습니다)

평가 기준:
1: 부적절함 - 지시된 할루시네이션 유형과 다르거나, 환각을 제대로 주입하지 않음
2: 보통 - 지시된 할루시네이션 유형의 정의를 부분적으로 충족하거나 약간 모호한 부분이 있음
3: 적절함 - 지시된 할루시네이션 유형의 정의를 정확하게 구현함

답변 형식: 숫자만 출력하세요 (1, 2, 또는 3)"""
        
        messages = [
            {
                "role": "system",
                "content": "당신은 할루시네이션 평가 전문가입니다. 주입된 할루시네이션이 지시된 유형에 맞게 적절히 구현되었는지 정확하게 판단하세요."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        response_text = self.llm.call_llm(messages, temperature=0.1)
        
        # 숫자만 추출
        numbers = re.findall(r'\d+', response_text)
        if numbers:
            try:
                score = int(numbers[0])
                return max(1, min(3, score))  # 1-3 범위로 제한
            except:
                pass
        return 2  # 기본값
    
    def evaluate_naturalness(
        self,
        query: str,
        hallucinated_answer: str,
        original_answer: str,
        dialogue_history: List[Dict[str, str]]
    ) -> int:
        """
        환각 주입의 자연스러움 평가 (1-3)
        할루시네이션이 문맥과 자연스럽게 어울리고, 인위적으로 삽입된 것처럼 보이지 않는가?
        
        Args:
            query: 사용자 질문
            hallucinated_answer: 할루시네이션이 주입된 답변
            original_answer: 원본 정답
            dialogue_history: 대화 히스토리 (문맥 파악용)
        
        Returns:
            1, 2, 또는 3 (높을수록 좋음)
        """
        # 대화 히스토리 텍스트 생성
        history_text = ""
        if dialogue_history:
            history_lines = []
            for i, turn in enumerate(dialogue_history[-3:], 1):  # 최근 3턴만 사용
                role = turn.get('role', '')
                content = turn.get('content', '')
                history_lines.append(f"{role}: {content}")
            history_text = "\n".join(history_lines)
        
        prompt = f"""다음 정보가 주어집니다:

대화 히스토리:
{history_text if history_text else "(없음)"}

질문: {query}

원본 정답: {original_answer}

할루시네이션이 주입된 답변: {hallucinated_answer}

할루시네이션이 주입된 답변이 문맥과 자연스럽게 어울리고, 인위적으로 삽입된 것처럼 보이지 않는지 평가하세요. (점수가 높을수록 좋습니다)

평가 기준:
1: 어색함 - 문법적 오류가 발생하거나, 문장의 가독성을 해치고 환각 주입 사실이 즉각적으로 노출됨
2: 보통 - 주변 문맥과 어느 정도 어울리지만, 약간 어색한 부분이 있어 인위적으로 삽입된 것처럼 보임
3: 자연스러움 - 주변 문맥과 자연스럽게 어울리고, 인위적으로 삽입된 것 같지 않게 보임

답변 형식: 숫자만 출력하세요 (1, 2, 또는 3)"""
        
        messages = [
            {
                "role": "system",
                "content": "당신은 문장 자연스러움 평가 전문가입니다. 할루시네이션이 주입된 답변이 얼마나 자연스럽게 작성되었는지 판단하세요."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        response_text = self.llm.call_llm(messages, temperature=0.1)
        
        # 숫자만 추출
        numbers = re.findall(r'\d+', response_text)
        if numbers:
            try:
                score = int(numbers[0])
                return max(1, min(3, score))  # 1-3 범위로 제한
            except:
                pass
        return 2  # 기본값
    
    def evaluate_turn(
        self,
        turn: Dict[str, Any],
        dialogue_history: List[Dict[str, str]],
        parallel: bool = False,
        num_workers: int = 2
    ) -> Dict[str, int]:
        """
        한 턴에 대한 할루시네이션 평가 수행 (적절성, 자연스러움)
        
        Args:
            turn: 평가할 턴 데이터
            dialogue_history: 대화 히스토리 (자연스러움 평가용)
            parallel: 병렬 처리 여부
            num_workers: 병렬 처리 시 워커 수
            
        Returns:
            {
                'appropriateness': int (1-3),
                'naturalness': int (1-3)
            }
        """
        query = turn.get('user_question', '')
        original_answer = turn.get('original_answer', '')
        hallucinated_answer = turn.get('hallucinated_answer', turn.get('assistant_answer', ''))
        hallucination_type = turn.get('hallucination_type', '')
        documents = turn.get('passages', [])
        
        if parallel:
            # 병렬로 2가지 평가 수행
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_to_metric = {
                    executor.submit(
                        self.evaluate_appropriateness, 
                        query, 
                        hallucinated_answer, 
                        hallucination_type,
                        original_answer,
                        documents
                    ): 'appropriateness',
                    executor.submit(
                        self.evaluate_naturalness, 
                        query, 
                        hallucinated_answer,
                        original_answer,
                        dialogue_history
                    ): 'naturalness'
                }
                
                results = {}
                for future in as_completed(future_to_metric):
                    metric = future_to_metric[future]
                    try:
                        results[metric] = future.result()
                    except Exception as exc:
                        print(f"평가 {metric} 중 오류 발생: {exc}")
                        results[metric] = 2  # 기본값
                
                return {
                    'appropriateness': results.get('appropriateness', 2),
                    'naturalness': results.get('naturalness', 2)
                }
        else:
            # 순차 처리
            appropriateness = self.evaluate_appropriateness(
                query, 
                hallucinated_answer, 
                hallucination_type,
                original_answer,
                documents
            )
            naturalness = self.evaluate_naturalness(
                query, 
                hallucinated_answer,
                original_answer,
                dialogue_history
            )
            
            return {
                'appropriateness': appropriateness,
                'naturalness': naturalness
            }


def evaluate_single_conversation(
    conversation: Dict[str, Any],
    conv_idx: int,
    total_conversations: int,
    evaluator: HallucinationEvaluator,
    parallel_turn: bool,
    num_workers: int
) -> Dict[str, Any]:
    """
    단일 대화의 할루시네이션을 평가하는 함수 (병렬 처리용)
    
    Returns:
        평가된 대화 딕셔너리
    """
    print(f"Processing conversation {conv_idx}/{total_conversations}...")
    
    dialogue = conversation.get('dialogue', [])
    evaluated_dialogue = []
    dialogue_history = []
    
    if not dialogue:
        print(f"  Warning: No dialogue turns found in conversation {conv_idx}")
        # 대화가 비어있으면 원본 대화 반환
        evaluated_conversation = conversation.copy()
        evaluated_conversation['average_hallucination_evaluations'] = {
            'appropriateness': 0,
            'naturalness': 0
        }
        return evaluated_conversation
    
    # 각 턴에 대해 적절성, 자연스러움 평가
    for turn_idx, turn in enumerate(dialogue, 1):
        # LLM이 평가 (병렬 처리 옵션 포함)
        evaluations = evaluator.evaluate_turn(
            turn=turn,
            dialogue_history=dialogue_history,
            parallel=parallel_turn,
            num_workers=num_workers
        )
        
        # 평가 결과를 턴에 추가
        evaluated_turn = turn.copy()
        evaluated_turn['hallucination_evaluations'] = evaluations
        evaluated_dialogue.append(evaluated_turn)
        
        # 히스토리 업데이트
        dialogue_history.append({'role': 'user', 'content': turn.get('user_question', '')})
        hallucinated_answer = turn.get('hallucinated_answer', turn.get('assistant_answer', ''))
        dialogue_history.append({'role': 'assistant', 'content': hallucinated_answer})
    
    # 대화 전체 평가 통계 계산
    all_evaluations = [turn['hallucination_evaluations'] for turn in evaluated_dialogue]
    avg_evaluations = {
        'appropriateness': sum(e['appropriateness'] for e in all_evaluations) / len(all_evaluations),
        'naturalness': sum(e['naturalness'] for e in all_evaluations) / len(all_evaluations)
    }
    
    evaluated_conversation = conversation.copy()
    evaluated_conversation['dialogue'] = evaluated_dialogue
    evaluated_conversation['average_hallucination_evaluations'] = avg_evaluations
    return evaluated_conversation


def evaluate_hallucination_results(
    input_file: str,
    output_file: str,
    provider: str = "openai",
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    parallel_conversations: bool = False,
    parallel_turn: bool = True,
    num_workers: int = 4,
    max_samples: Optional[int] = None
):
    """
    할루시네이션이 주입된 대화 데이터를 읽어서 평가하고 결과를 저장
    
    Args:
        input_file: 입력 파일 경로 (hallucination_results.json)
        output_file: 출력 파일 경로
        provider: LLM provider
        model: LLM 모델 이름
        api_key: API 키
        parallel_conversations: 대화 간 병렬 처리 여부
        parallel_turn: 턴 내 평가 메트릭 병렬 처리 여부
        num_workers: 병렬 처리 시 워커 수
        max_samples: 평가할 최대 대화 수 (None이면 전체)
    """
    print(f"Loading hallucination results from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 평가에서 제외할 hallucination_type
    # - true_refusal: 원본이 거절 답변인 케이스 태깅(주입/환각 아님)
    # - false_acceptance: 문서(근거) 평가를 다른 파이프라인에서 수행하기로 했으므로 여기서는 스킵
    SKIP_HALLUCINATION_TYPES = {"true_refusal", "false_acceptance"}

    def _should_evaluate_turn(ht: Any) -> bool:
        return not (ht is None or ht == '' or ht in SKIP_HALLUCINATION_TYPES)

    def _convert_injected_turn_to_eval_turn(turn: Dict[str, Any]) -> Dict[str, Any]:
        """
        inject_hallucinations.py의 출력(turn) 또는 그와 유사한 구조를
        HallucinationEvaluator.evaluate_turn이 기대하는 구조로 변환.
        """
        hallucination_type = turn.get('hallucination_type')
        modified = turn.get('modified', {}) or {}
        # false_acceptance는 passage만 바뀌는 경우가 많아서 modified.passage를 우선 사용
        passage_obj = modified.get('passage', turn.get('passage'))
        return {
            'user_question': turn.get('user_question', ''),
            'original_answer': turn.get('assistant_answer', turn.get('original_answer', '')),
            'hallucinated_answer': (
                modified.get('assistant_answer')
                or turn.get('hallucinated_answer')
                or turn.get('assistant_answer', '')
            ),
            'hallucination_type': hallucination_type,
            'passages': [passage_obj] if passage_obj else turn.get('passages', []),
            'turn': turn.get('turn'),
            'dialogue_id': turn.get('dialogue_id'),
            'change_description': turn.get('change_description'),
        }

    # turns 형식인지 conversations 형식인지 확인
    if 'turns' in data and 'conversations' not in data:
        # turns 형식을 conversations 형식으로 변환 (환각이 주입된 턴만)
        print("Converting turns format to conversations format (hallucinated turns only)...")
        turns = data.get('turns', [])
        
        # dialogue_id로 그룹화 (환각이 주입된 턴만)
        conversations_dict = {}
        hallucinated_count = 0
        for turn in turns:
            dialogue_id = turn.get('dialogue_id')
            hallucination_type = turn.get('hallucination_type')
            
            # 평가 대상만 포함
            if not _should_evaluate_turn(hallucination_type):
                continue
            
            hallucinated_count += 1
            
            if dialogue_id not in conversations_dict:
                conversations_dict[dialogue_id] = {
                    'dialogue_id': dialogue_id,
                    'dialogue': []
                }
            
            converted_turn = _convert_injected_turn_to_eval_turn(turn)
            
            conversations_dict[dialogue_id]['dialogue'].append(converted_turn)
        
        # dialogue_id 순서대로 정렬
        conversations = [conversations_dict[did] for did in sorted(conversations_dict.keys())]
        print(f"Converted {hallucinated_count} hallucinated turns from {len(turns)} total turns into {len(conversations)} conversations")
    else:
        # 이미 conversations 형식
        conversations = data.get('conversations', [])

        # conversations 형식이라도 inject_hallucinations.py의 원본 출력 구조인 경우가 많아서
        # (assistant_answer/passage/modified 기반) 평가용 구조로 변환 + 환각 턴만 필터링
        normalized_conversations = []
        injected_like = False
        for conv in conversations:
            dialogue = conv.get('dialogue', []) or []
            normalized_dialogue = []
            for turn in dialogue:
                ht = turn.get('hallucination_type')
                if not _should_evaluate_turn(ht):
                    continue
                # injected output structure detection
                if 'assistant_answer' in turn or 'modified' in turn or 'passage' in turn:
                    injected_like = True
                normalized_dialogue.append(_convert_injected_turn_to_eval_turn(turn))
            if normalized_dialogue:
                new_conv = conv.copy()
                new_conv['dialogue'] = normalized_dialogue
                normalized_conversations.append(new_conv)

        # 변환/필터 결과가 존재하면 그걸 사용 (기존 "hallucinated turns only" 가정 유지)
        if normalized_conversations:
            if injected_like:
                print("Detected injected conversations format. Converting to evaluation format (hallucinated turns only)...")
            conversations = normalized_conversations
    
    total_conversations = len(conversations)
    
    # max_samples가 지정되면 해당 개수만큼만 평가
    if max_samples is not None and max_samples > 0:
        original_count = len(conversations)
        conversations = conversations[:max_samples]
        total_conversations = len(conversations)
        print(f"Limited to {max_samples} conversations (out of {original_count} total)")
    
    print(f"Evaluating {total_conversations} conversations...")
    if parallel_conversations:
        print(f"  병렬 처리 모드: 대화 간 병렬 처리 활성화 (워커 수: {num_workers})")
    if parallel_turn:
        print(f"  턴 내 병렬 처리: 각 턴의 평가 메트릭 병렬 처리 활성화 (워커 수: 2)")
    
    evaluated_conversations = []
    
    if parallel_conversations:
        # 대화 간 병렬 처리
        # 각 워커마다 별도의 evaluator 인스턴스 생성
        def create_evaluator():
            return HallucinationEvaluator(provider=provider, model=model, api_key=api_key)
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # 각 대화에 대해 평가 작업 제출
            future_to_conv = {
                executor.submit(
                    evaluate_single_conversation,
                    conversation,
                    conv_idx,
                    total_conversations,
                    create_evaluator(),
                    parallel_turn,
                    2  # 턴 내 병렬 처리용 워커 수
                ): conv_idx
                for conv_idx, conversation in enumerate(conversations, 1)
            }
            
            # 결과 수집 (순서 유지)
            results = [None] * total_conversations
            for future in as_completed(future_to_conv):
                conv_idx = future_to_conv[future]
                try:
                    result = future.result()
                    results[conv_idx - 1] = result
                    print(f"✅ Completed conversation {conv_idx}/{total_conversations}")
                except Exception as exc:
                    print(f"❌ Conversation {conv_idx} 평가 중 오류 발생: {exc}")
                    # 오류 발생 시 원본 대화 유지
                    results[conv_idx - 1] = conversations[conv_idx - 1]
            
            evaluated_conversations = [r for r in results if r is not None]
    else:
        # 순차 처리
        evaluator = HallucinationEvaluator(provider=provider, model=model, api_key=api_key)
        
        for conv_idx, conversation in enumerate(conversations, 1):
            evaluated_conversation = evaluate_single_conversation(
                conversation,
                conv_idx,
                total_conversations,
                evaluator,
                parallel_turn,
                2  # 턴 내 병렬 처리용 워커 수
            )
            evaluated_conversations.append(evaluated_conversation)
    
    # 전체 통계 계산
    all_avg_evaluations = []
    for conv in evaluated_conversations:
        if 'average_hallucination_evaluations' in conv:
            all_avg_evaluations.append(conv['average_hallucination_evaluations'])
    
    if all_avg_evaluations:
        overall_avg = {
            'appropriateness': sum(e['appropriateness'] for e in all_avg_evaluations) / len(all_avg_evaluations),
            'naturalness': sum(e['naturalness'] for e in all_avg_evaluations) / len(all_avg_evaluations)
        }
    else:
        overall_avg = {}
    
    # 결과 저장
    output_data = {
        'total_conversations': total_conversations,
        'overall_average_hallucination_evaluations': overall_avg,
        'conversations': evaluated_conversations
    }
    
    print(f"\nSaving evaluated results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print("\nHallucination Evaluation completed!")
    print(f"\nOverall Average Hallucination Evaluations:")
    print(f"  Appropriateness (적절성): {overall_avg.get('appropriateness', 0):.3f} (1-3)")
    print(f"  Naturalness (자연스러움): {overall_avg.get('naturalness', 0):.3f} (1-3)")


def main():
    parser = argparse.ArgumentParser(description="Evaluate hallucination in dialogue results using LLM")
    parser.add_argument(
        '--input',
        type=str,
        default='hallucination_results.json',
        help='Input hallucination results JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='hallucination_results_evaluated.json',
        help='Output evaluated results JSON file'
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
        '--parallel-conversations',
        action='store_true',
        help='Enable parallel processing across conversations'
    )
    parser.add_argument(
        '--no-parallel-turn',
        action='store_true',
        help='Disable parallel processing within each turn (default: enabled)'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of worker threads for parallel processing (default: 4)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of conversations to evaluate (default: all)'
    )
    
    args = parser.parse_args()
    
    evaluate_hallucination_results(
        input_file=args.input,
        output_file=args.output,
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        parallel_conversations=args.parallel_conversations,
        parallel_turn=not args.no_parallel_turn,
        num_workers=args.num_workers,
        max_samples=args.max_samples
    )


if __name__ == '__main__':
    main()

