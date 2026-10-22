#!/usr/bin/env python3
"""
Dialogue Evaluation Script
5가지 기준으로 대화를 평가합니다:
- Answerability (0-1): 답변이 문서에 있는지
- Plausibility (1-3): 실제 사용자가 질문할 가능성
- Correctness (0-1): 답변이 문서 정보에 기반해 정확한지
- Diversity (1-3): 각 질문-답변이 서로 충분히 다른지
- Coherency (1-3): 대화 흐름이 자연스럽고 일관성 있는지
"""

import json
import argparse
import re
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from qa_model import QAModel


class DialogueEvaluator:
    """대화 평가 클래스"""
    
    def __init__(self, provider: str = "openai", model: str = "gpt-4o", api_key: Optional[str] = None):
        """
        Args:
            provider: "openai" or "google"
            model: 모델 이름
            api_key: API 키 (없으면 환경변수에서 가져옴)
        """
        self.llm = QAModel(provider=provider, model=model, api_key=api_key)
    
    def evaluate_answerability(
        self,
        query: str,
        original_answer: str,
        documents: List[Dict[str, Any]]
    ) -> int:
        """
        Answerability 평가 (0-1)
        질문에 대한 답변(original_answer)이 주어진 문서에 있는지 평가
        
        Returns:
            0 또는 1
        """
        # 문서 텍스트 추출
        doc_texts = []
        for doc in documents:
            doc_texts.append(f"[문서 {doc.get('context_id', '')}]\n{doc.get('context', '')}")
        documents_text = "\n\n".join(doc_texts)
        
        prompt = f"""다음 질문과 답변, 그리고 관련 문서가 주어집니다.

질문: {query}
답변: {original_answer}

관련 문서:
{documents_text}

질문에 대한 답변이 위 문서들에 포함되어 있는지 평가하세요.
- 답변이 문서에 명시적으로 있거나 문서 내용으로부터 추론 가능하면 1
- 답변이 문서에 없거나 문서 내용으로 답할 수 없으면 0

답변 형식: 숫자만 출력하세요 (0 또는 1)"""
        
        messages = [
            {
                "role": "system",
                "content": "당신은 문서 기반 질문-답변 시스템의 평가자입니다. 답변이 문서에 있는지 정확하게 판단하세요."
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
                return 1 if score == 1 else 0
            except:
                pass
        # 파싱 실패 시 기본값
        return 0
    
    def evaluate_plausibility(
        self,
        query: str,
        dialogue_history: List[Dict[str, str]]
    ) -> int:
        """
        Plausibility 평가 (1-3)
        실제 사용자가 질문할 가능성 평가
        
        Returns:
            1, 2, 또는 3
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
        
        prompt = f"""다음은 대화 히스토리와 현재 질문입니다.

대화 히스토리:
{history_text if history_text else "(없음)"}

현재 질문: {query}

이 질문이 실제 사용자가 자연스럽게 할 수 있는 질문인지 평가하세요. (점수가 높을수록 좋습니다)
1: 부자연스럽거나 실제 사용자가 하기 어려운 질문
2: 어느 정도 자연스럽지만 약간 부자연스러운 부분이 있음
3: 매우 자연스럽고 실제 사용자가 할 가능성이 높음

답변 형식: 숫자만 출력하세요 (1, 2, 또는 3)"""
        
        messages = [
            {
                "role": "system",
                "content": "당신은 대화 평가자입니다. 질문이 실제 사용자가 할 수 있는 자연스러운 질문인지 판단하세요."
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
    
    def evaluate_correctness(
        self,
        query: str,
        response: str,
        documents: List[Dict[str, Any]]
    ) -> int:
        """
        Correctness 평가 (0-1)
        답변이 문서 정보에 기반해 정확한지 평가
        
        Returns:
            0 또는 1
        """
        # 문서 텍스트 추출
        doc_texts = []
        for doc in documents:
            doc_texts.append(f"[문서 {doc.get('context_id', '')}]\n{doc.get('context', '')}")
        documents_text = "\n\n".join(doc_texts)
        
        prompt = f"""다음 질문과 답변, 그리고 관련 문서가 주어집니다.

질문: {query}
답변: {response}

관련 문서:
{documents_text}

답변이 문서의 정보에 기반하여 정확한지 평가하세요.
- 답변이 문서 정보에 기반하여 정확하면 1
- 답변이 문서 정보와 다르거나 잘못되었으면 0

답변 형식: 숫자만 출력하세요 (0 또는 1)"""
        
        messages = [
            {
                "role": "system",
                "content": "당신은 문서 기반 질문-답변 시스템의 평가자입니다. 답변이 문서 정보에 기반해 정확한지 판단하세요."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        response_text = self.llm.call_llm(messages, temperature=0.1)
        
        # 숫자만 추출
        try:
            score = int(response_text.strip().split()[0])
            return 1 if score == 1 else 0
        except:
            return 0  # 기본값
    
    def evaluate_diversity(
        self,
        dialogue_history: List[Dict[str, str]]
    ) -> int:
        """
        Diversity 평가 (1-3)
        전체 대화에서 각 질문-답변이 서로 충분히 다른지 평가
        
        Returns:
            1, 2, 또는 3 (높을수록 좋음)
        """
        
        # 대화 히스토리에서 질문-답변 쌍 추출
        qa_pairs = []
        for i in range(0, len(dialogue_history), 2):
            if i + 1 < len(dialogue_history):
                user_msg = dialogue_history[i]
                assistant_msg = dialogue_history[i + 1]
                if user_msg.get('role') == 'user' and assistant_msg.get('role') == 'assistant':
                    qa_pairs.append({
                        'question': user_msg.get('content', ''),
                        'answer': assistant_msg.get('content', '')
                    })
        
        if len(qa_pairs) < 2:
            return 3  # 턴이 적으면 다양성 최고 (점수가 높을수록 좋음)
        
        # 질문-답변 쌍들을 텍스트로 변환
        qa_text = ""
        for idx, qa in enumerate(qa_pairs, 1):
            qa_text += f"턴 {idx}:\n질문: {qa['question']}\n답변: {qa['answer']}\n\n"
        
        prompt = f"""다음은 전체 대화 히스토리입니다.
        
[유의사항]
본 대화에는 일관성 검증을 위한 '재확인 질문'이 의도적으로 한 번 포함되어 있습니다.
평가 시 해당 질문과 그에 따른 답변 턴은 내용 중복으로 간주하지 않으며, 평가 대상에서 제외합니다.
이를 제외한 나머지 대화 턴들을 바탕으로 아래 기준에 따라 점수를 부여하세요.

대화 히스토리:
{qa_text}

전체 대화에서 각 질문-답변이 서로 충분히 다른지 평가하세요. (점수가 높을수록 좋습니다)
1: 새로운 정보 없이 이전 내용을 단순히 되풀이함
2: 정보의 진전은 있으나, 특정 대화 내용이 반복됨
3: 대부분의 턴에서 새로운 정보가 추가되며, 대화 내용이 서로 다름

답변 형식: 숫자만 출력하세요 (1, 2, 또는 3)"""
        
        messages = [
            {
                "role": "system",
                "content": "당신은 대화 평가자입니다. 전체 대화에서 질문-답변이 얼마나 서로 다른지 판단하세요."
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
    
    def evaluate_coherency(
        self,
        dialogue_history: List[Dict[str, str]]
    ) -> int:
        """
        Coherency 평가 (1-3)
        전체 대화 흐름이 자연스럽고 일관성 있는지 평가
        
        Returns:
            1, 2, 또는 3 (높을수록 좋음)
        """
        
        # 대화 히스토리 텍스트 생성
        history_lines = []
        for turn in dialogue_history:
            role = turn.get('role', '')
            content = turn.get('content', '')
            if role == 'user':
                history_lines.append(f"사용자: {content}")
            elif role == 'assistant':
                history_lines.append(f"어시스턴트: {content}")
        history_text = "\n".join(history_lines)
        
        prompt = f"""다음은 전체 대화 히스토리입니다.

대화 히스토리:
{history_text}

전체 대화 흐름이 자연스럽고 일관성 있는지 평가하세요. (점수가 높을수록 좋습니다)
1: 이전 맥락과 관련이 없거나 논리적 모순이 있어 대화의 흐름이 부자연스러움
2: 문맥과 관련은 있으나, 답변이 모호하거나 화제 전환이 매끄럽지 않아 흐름이 다소 어색함
3: 이전 맥락을 정확히 반영하고 논리적으로 모순이 없으며, 대화가 자연스럽고 유기적으로 이어짐

답변 형식: 숫자만 출력하세요 (1, 2, 또는 3)"""
        
        messages = [
            {
                "role": "system",
                "content": "당신은 대화 평가자입니다. 전체 대화 흐름이 자연스럽고 일관성 있는지 판단하세요."
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
        num_workers: int = 3
    ) -> Dict[str, int]:
        """
        한 턴에 대한 평가 수행 (Answerability, Plausibility, Correctness만)
        
        Args:
            turn: 평가할 턴 데이터
            dialogue_history: 대화 히스토리 (plausibility 평가용)
            parallel: 병렬 처리 여부
            num_workers: 병렬 처리 시 워커 수
            
        Returns:
            {
                'answerability': int (0-1),
                'plausibility': int (1-3),
                'correctness': int (0-1)
            }
        """
        query = turn.get('user_question', '')
        original_answer = turn.get('original_answer', '')
        assistant_answer = turn.get('assistant_answer', '')
        documents = turn.get('passages', [])
        
        if parallel:
            # 병렬로 3가지 평가 수행
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_to_metric = {
                    executor.submit(self.evaluate_answerability, query, original_answer, documents): 'answerability',
                    executor.submit(self.evaluate_plausibility, query, dialogue_history): 'plausibility',
                    executor.submit(self.evaluate_correctness, query, assistant_answer, documents): 'correctness'
                }
                
                results = {}
                for future in as_completed(future_to_metric):
                    metric = future_to_metric[future]
                    try:
                        results[metric] = future.result()
                    except Exception as exc:
                        print(f"평가 {metric} 중 오류 발생: {exc}")
                        # 기본값 설정
                        if metric == 'answerability' or metric == 'correctness':
                            results[metric] = 0
                        else:
                            results[metric] = 2
                
                return {
                    'answerability': results.get('answerability', 0),
                    'plausibility': results.get('plausibility', 2),
                    'correctness': results.get('correctness', 0)
                }
        else:
            # 순차 처리
            answerability = self.evaluate_answerability(query, original_answer, documents)
            plausibility = self.evaluate_plausibility(query, dialogue_history)
            correctness = self.evaluate_correctness(query, assistant_answer, documents)
            
            return {
                'answerability': answerability,
                'plausibility': plausibility,
                'correctness': correctness
            }
        # return {
        #     'answerability': 0,
        #     'plausibility': 0,
        #     'correctness': 0
        # }

def get_human_input_for_turn(
    turn: Dict[str, Any],
    turn_idx: int,
    total_turns: int
) -> Dict[str, int]:
    """
    사람이 한 턴에 대해 평가 입력을 받음
    
    Returns:
        {
            'answerability': int (0-1),
            'plausibility': int (1-3),
            'correctness': int (0-1)
        }
    """
    print("\n" + "="*80)
    print(f"턴 {turn_idx}/{total_turns} 평가")
    print("="*80)
    
    query = turn.get('user_question', '')
    original_answer = turn.get('original_answer', '')
    assistant_answer = turn.get('assistant_answer', '')
    documents = turn.get('passages', [])
    
    print(f"\n질문: {query}")
    print(f"\n원본 답변 (original_answer): {original_answer}")
    print(f"\n모델 답변 (assistant_answer): {assistant_answer}")
    
    if documents:
        print(f"\n관련 문서 ({len(documents)}개):")
        for i, doc in enumerate(documents[:2], 1):  # 최대 2개만 표시
            context = doc.get('context', '')[:200]  # 처음 200자만
            print(f"  [{i}] {context}...")
        if len(documents) > 2:
            print(f"  ... 외 {len(documents) - 2}개 문서")
    
    print("\n" + "-"*80)
    print("평가 기준:")
    print("  Answerability (0-1): 질문에 대한 답변(original_answer)이 문서에 있는지")
    print("    - 0: 답변이 문서에 없거나 문서 내용으로 답할 수 없음")
    print("    - 1: 답변이 문서에 명시적으로 있거나 문서 내용으로부터 추론 가능")
    print("\n  Plausibility (1-3): 실제 사용자가 질문할 가능성 (높을수록 좋음)")
    print("    - 1: 부자연스럽거나 실제 사용자가 하기 어려운 질문")
    print("    - 2: 어느 정도 자연스럽지만 약간 부자연스러운 부분이 있음")
    print("    - 3: 매우 자연스럽고 실제 사용자가 할 가능성이 높음")
    print("\n  Correctness (0-1): 답변(assistant_answer)이 문서 정보에 기반해 정확한지")
    print("    - 0: 답변이 문서 정보와 다르거나 잘못됨")
    print("    - 1: 답변이 문서 정보에 기반하여 정확함")
    print("-"*80)
    
    while True:
        try:
            answerability = input("\nAnswerability (0 또는 1): ").strip()
            answerability = int(answerability)
            if answerability in [0, 1]:
                break
            else:
                print("0 또는 1만 입력하세요.")
        except ValueError:
            print("숫자를 입력하세요.")
    
    while True:
        try:
            plausibility = input("Plausibility (1, 2, 또는 3): ").strip()
            plausibility = int(plausibility)
            if plausibility in [1, 2, 3]:
                break
            else:
                print("1, 2, 또는 3만 입력하세요.")
        except ValueError:
            print("숫자를 입력하세요.")
    
    while True:
        try:
            correctness = input("Correctness (0 또는 1): ").strip()
            correctness = int(correctness)
            if correctness in [0, 1]:
                break
            else:
                print("0 또는 1만 입력하세요.")
        except ValueError:
            print("숫자를 입력하세요.")
    
    return {
        'answerability': answerability,
        'plausibility': plausibility,
        'correctness': correctness
    }


def get_human_input_for_dialogue(
    dialogue_history: List[Dict[str, str]],
    conv_idx: int,
    total_conversations: int
) -> Dict[str, int]:
    """
    사람이 전체 대화에 대해 Diversity와 Coherency 평가 입력을 받음
    
    Returns:
        {
            'diversity': int (1-3),
            'coherency': int (1-3)
        }
    """
    print("\n" + "="*80)
    print(f"대화 {conv_idx}/{total_conversations} 전체 평가")
    print("="*80)
    
    print("\n전체 대화 히스토리:")
    print("-"*80)
    for i, turn in enumerate(dialogue_history, 1):
        role = turn.get('role', '')
        content = turn.get('content', '')
        if role == 'user':
            print(f"\n[{i}] 사용자: {content}")
        elif role == 'assistant':
            print(f"    어시스턴트: {content}")
    print("-"*80)
    
    print("\n평가 기준:")
    print("  Diversity (1-3): 전체 대화에서 각 질문-답변이 서로 충분히 다른지 (높을수록 좋음)")
    print("    - 1: 매우 유사하거나 거의 동일한 내용이 반복됨")
    print("    - 2: 어느 정도 다양하지만 일부 중복이 있음")
    print("    - 3: 매우 다양하고 각 턴이 새로운 주제나 관점을 다룸")
    print("\n  Coherency (1-3): 전체 대화 흐름이 자연스럽고 일관성 있는지 (높을수록 좋음)")
    print("    - 1: 부자연스럽거나 일관성이 부족함")
    print("    - 2: 대체로 자연스럽지만 약간 어색한 부분이 있음")
    print("    - 3: 매우 자연스럽고 일관성 있음")
    print("-"*80)
    
    while True:
        try:
            diversity = input("\nDiversity (1, 2, 또는 3): ").strip()
            diversity = int(diversity)
            if diversity in [1, 2, 3]:
                break
            else:
                print("1, 2, 또는 3만 입력하세요.")
        except ValueError:
            print("숫자를 입력하세요.")
    
    while True:
        try:
            coherency = input("Coherency (1, 2, 또는 3): ").strip()
            coherency = int(coherency)
            if coherency in [1, 2, 3]:
                break
            else:
                print("1, 2, 또는 3만 입력하세요.")
        except ValueError:
            print("숫자를 입력하세요.")
    
    return {
        'diversity': diversity,
        'coherency': coherency
    }


def evaluate_single_conversation(
    conversation: Dict[str, Any],
    conv_idx: int,
    total_conversations: int,
    evaluator: Optional[DialogueEvaluator],
    human_eval: bool,
    parallel_turn: bool,
    num_workers: int
) -> Dict[str, Any]:
    """
    단일 대화를 평가하는 함수 (병렬 처리용)
    
    Returns:
        평가된 대화 딕셔너리
    """
    if not human_eval:
        print(f"Processing conversation {conv_idx}/{total_conversations}...")
    
    dialogue = conversation.get('dialogue', [])
    evaluated_dialogue = []
    dialogue_history = []
    
    # 각 턴에 대해 Answerability, Plausibility, Correctness 평가
    for turn_idx, turn in enumerate(dialogue, 1):
        if human_eval:
            # 사람이 평가
            evaluations = get_human_input_for_turn(turn, turn_idx, len(dialogue))
        else:
            print(f"  Evaluating turn {turn_idx}/{len(dialogue)}...")
            # LLM이 평가 (병렬 처리 옵션 포함)
            evaluations = evaluator.evaluate_turn(
                turn=turn,
                dialogue_history=dialogue_history,
                parallel=parallel_turn,
                num_workers=num_workers
            )
        
        # 평가 결과를 턴에 추가
        evaluated_turn = turn.copy()
        evaluated_turn['evaluations'] = evaluations
        evaluated_dialogue.append(evaluated_turn)
        
        # 히스토리 업데이트
        dialogue_history.append({'role': 'user', 'content': turn.get('user_question', '')})
        dialogue_history.append({'role': 'assistant', 'content': turn.get('assistant_answer', '')})
    
    # 대화 전체에 대해 Diversity와 Coherency 평가 (한 번만)
    if human_eval:
        # 사람이 평가
        dialogue_eval = get_human_input_for_dialogue(dialogue_history, conv_idx, total_conversations)
        diversity = dialogue_eval['diversity']
        coherency = dialogue_eval['coherency']
    else:
        print(f"  Evaluating diversity and coherency for entire conversation...")
        diversity = evaluator.evaluate_diversity(dialogue_history)
        coherency = evaluator.evaluate_coherency(dialogue_history)
    
    # 대화 전체 평가 통계 계산
    all_evaluations = [turn['evaluations'] for turn in evaluated_dialogue]
    avg_evaluations = {
        'answerability': sum(e['answerability'] for e in all_evaluations) / len(all_evaluations),
        'plausibility': sum(e['plausibility'] for e in all_evaluations) / len(all_evaluations),
        'correctness': sum(e['correctness'] for e in all_evaluations) / len(all_evaluations),
        'diversity': diversity,  # 대화 전체에 대한 단일 점수
        'coherency': coherency,  # 대화 전체에 대한 단일 점수
    }
    
    evaluated_conversation = conversation.copy()
    evaluated_conversation['dialogue'] = evaluated_dialogue
    evaluated_conversation['average_evaluations'] = avg_evaluations
    return evaluated_conversation


def evaluate_dialogue_results(
    input_file: str,
    output_file: str,
    provider: str = "openai",
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    human_eval: bool = False,
    parallel_conversations: bool = False,
    parallel_turn: bool = True,
    num_workers: int = 4
):
    """
    dialogue_results.json을 읽어서 평가하고 결과를 저장
    
    Args:
        input_file: 입력 파일 경로 (dialogue_results.json)
        output_file: 출력 파일 경로
        provider: LLM provider
        model: LLM 모델 이름
        api_key: API 키
        human_eval: True이면 사람이 평가, False이면 LLM이 평가
        parallel_conversations: 대화 간 병렬 처리 여부
        parallel_turn: 턴 내 평가 메트릭 병렬 처리 여부
        num_workers: 병렬 처리 시 워커 수
    """
    print(f"Loading dialogue results from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    conversations = data.get('conversations', [])
    total_conversations = len(conversations)
    
    if human_eval:
        print(f"\n사람 평가 모드: {total_conversations}개의 대화를 평가합니다.")
        print("각 턴과 대화 전체에 대해 평가를 입력해주세요.\n")
        parallel_conversations = False  # 사람 평가는 병렬 처리 불가
        parallel_turn = False
    else:
        print(f"Evaluating {total_conversations} conversations...")
        if parallel_conversations:
            print(f"  병렬 처리 모드: 대화 간 병렬 처리 활성화 (워커 수: {num_workers})")
        if parallel_turn:
            print(f"  턴 내 병렬 처리: 각 턴의 평가 메트릭 병렬 처리 활성화 (워커 수: 3)")
    
    evaluated_conversations = []
    
    if parallel_conversations and not human_eval:
        # 대화 간 병렬 처리
        # 각 워커마다 별도의 evaluator 인스턴스 생성
        def create_evaluator():
            return DialogueEvaluator(provider=provider, model=model, api_key=api_key)
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # 각 대화에 대해 평가 작업 제출
            future_to_conv = {
                executor.submit(
                    evaluate_single_conversation,
                    conversation,
                    conv_idx,
                    total_conversations,
                    create_evaluator(),
                    human_eval,
                    parallel_turn,
                    3  # 턴 내 병렬 처리용 워커 수
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
        evaluator = None if human_eval else DialogueEvaluator(provider=provider, model=model, api_key=api_key)
        
        for conv_idx, conversation in enumerate(conversations, 1):
            evaluated_conversation = evaluate_single_conversation(
                conversation,
                conv_idx,
                total_conversations,
                evaluator,
                human_eval,
                parallel_turn,
                3  # 턴 내 병렬 처리용 워커 수
            )
            evaluated_conversations.append(evaluated_conversation)
    
    # 전체 통계 계산
    all_avg_evaluations = []
    for conv in evaluated_conversations:
        if 'average_evaluations' in conv:
            all_avg_evaluations.append(conv['average_evaluations'])
    
    if all_avg_evaluations:
        overall_avg = {
            'answerability': sum(e['answerability'] for e in all_avg_evaluations) / len(all_avg_evaluations),
            'plausibility': sum(e['plausibility'] for e in all_avg_evaluations) / len(all_avg_evaluations),
            'correctness': sum(e['correctness'] for e in all_avg_evaluations) / len(all_avg_evaluations),
            'diversity': sum(e['diversity'] for e in all_avg_evaluations) / len(all_avg_evaluations),
            'coherency': sum(e['coherency'] for e in all_avg_evaluations) / len(all_avg_evaluations),
        }
    else:
        overall_avg = {}
    
    # 결과 저장
    output_data = {
        'total_conversations': total_conversations,
        'overall_average_evaluations': overall_avg,
        'conversations': evaluated_conversations
    }
    
    print(f"\nSaving evaluated results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print("\nEvaluation completed!")
    print(f"\nOverall Average Evaluations:")
    print(f"  Answerability: {overall_avg.get('answerability', 0):.3f} (0-1)")
    print(f"  Plausibility: {overall_avg.get('plausibility', 0):.3f} (1-3)")
    print(f"  Correctness: {overall_avg.get('correctness', 0):.3f} (0-1)")
    print(f"  Diversity: {overall_avg.get('diversity', 0):.3f} (1-3)")
    print(f"  Coherency: {overall_avg.get('coherency', 0):.3f} (1-3)")


def main():
    parser = argparse.ArgumentParser(description="Evaluate dialogue results using LLM")
    parser.add_argument(
        '--input',
        type=str,
        default='dialogue_results.json',
        help='Input dialogue results JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='dialogue_results_evaluated.json',
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
        '--human',
        action='store_true',
        help='Use human evaluation instead of LLM'
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
    
    args = parser.parse_args()
    
    evaluate_dialogue_results(
        input_file=args.input,
        output_file=args.output,
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        human_eval=args.human,
        parallel_conversations=args.parallel_conversations,
        parallel_turn=not args.no_parallel_turn,
        num_workers=args.num_workers
    )


if __name__ == '__main__':
    main()

