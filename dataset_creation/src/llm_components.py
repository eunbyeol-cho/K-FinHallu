#!/usr/bin/env python3
"""
LLM Components for Dialogue Simulation
- QuestionSelector: 다음 질문 선택 및 액션 결정
- ReconfirmationGenerator: 재확인 질문 생성
- QuestionAnswerExtractor: paragraph에서 reference 문장 지정 후 질문/답변 추출
- QuestionRewriter: 질문을 자연스럽게 변환 (coreference, 문장 간소화)
- AnswerGenerator: RAG 기반 답변 생성
"""

from typing import List, Dict, Any, Optional, Tuple
from qa_model import QAModel


class QuestionSelector:
    """LLM_1: 다음 대화에 가장 적절한 질문 선택 및 액션 결정"""
    
    def __init__(self, provider: str = "openai", model: str = "gpt-4o-mini"):
        self.llm = QAModel(provider=provider, model=model)
    
    def select_next_question(
        self,
        conversation_history: List[Dict[str, str]],
        similar_context_questions: List[Dict[str, Any]],
        doc_id_questions: List[Dict[str, Any]],
        current_qid: str
    ) -> Dict[str, Any]:
        """
        유사한 context의 질문들과 같은 doc_id의 질문들 중에서 다음 질문 선택
        
        Args:
            conversation_history: 대화 히스토리
            similar_context_questions: 유사한 context에서 나온 질문들
                [{qid, question, context_id, doc_id}, ...]
            doc_id_questions: 같은 doc_id에 포함된 질문들
                [{qid, question, context_id}, ...]
            current_qid: 현재 질문의 qid
            
        Returns:
            {
                'action': 'select_question' | 'new_question',
                'selected_qid': Optional[str],  # action이 'select_question'일 때
                'selected_question': Optional[str],
                'reason': str
            }
        """
        # 대화 히스토리 요약
        history_text = "\n".join([
            f"{'사용자' if turn['role'] == 'user' else '어시스턴트'}: {turn['content']}"
            for turn in conversation_history
        ])
        
        # 유사 context 질문 포맷팅
        similar_questions_text = ""
        if similar_context_questions:
            similar_questions_text = "\n".join([
                f"{i+1}. [QID: {q['qid']}] {q['question']} (Context: {q.get('context_id', 'N/A')})"
                for i, q in enumerate(similar_context_questions)
            ])
        else:
            similar_questions_text = "없음"
        
        # 같은 doc_id 질문 포맷팅
        doc_questions_text = ""
        if doc_id_questions:
            doc_questions_text = "\n".join([
                f"{i+1}. [QID: {q['qid']}] {q['question']}"
                for i, q in enumerate(doc_id_questions)
            ])
        else:
            doc_questions_text = "없음"
        
        prompt = f"""당신은 대화의 자연스러운 흐름을 위해 다음 질문을 선택하는 역할입니다.

대화 히스토리:
{history_text}

유사한 context에서 나온 질문들:
{similar_questions_text}

같은 문서(doc_id)에 포함된 질문들:
{doc_questions_text}

다음 액션 중 하나를 선택하세요: (우선순위: select_question > new_question)

1. select_question: 위의 질문 목록에서 대화 히스토리와 겹치지 않으면서, 직전 대화의 키워드/맥락과 밀접하게 관련되거나 내용을 심화/확장할 수 있는 적절한 질문이 있는 경우
2. new_question: 질문 목록에 대화 흐름에 자연스럽게 이어갈 수 있는 적절한 질문이 없는 경우

select_question 선택 시 고려사항:
- 직전 대화의 핵심 주제/키워드를 유지하면서 이어가는 질문 우선 선택
- 급격한 주제 전환을 피하고, 현재 주제를 심화하거나 관련 세부사항을 묻는 질문 선호
- 직전 대화와 논리적으로 자연스럽게 연결되는 질문 선택

응답 형식:
액션: [select_question/new_question]
선택한 QID: [select_question일 경우 해당 QID, new_question일 경우 'N/A']
이유: [액션 선택 및 질문 연결의 상세한 추론 과정 및 근거 설명]"""
        
        messages = [
            {
                "role": "system",
                "content": "당신은 대화의 자연스러운 흐름을 위해 가장 적절한 다음 질문을 선택하는 AI입니다."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        response = self.llm.call_llm(messages, temperature=0.3)
        
        # 응답 파싱
        result = self._parse_response(response, similar_context_questions, doc_id_questions)
        return result
    
    def _parse_response(
        self,
        response: str,
        similar_context_questions: List[Dict[str, Any]],
        doc_id_questions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """LLM 응답을 파싱하여 구조화된 결과 반환"""
        result = {
            'action': 'new_question',
            'selected_qid': None,
            'selected_question': None,
            'reason': response
        }
        
        response_lower = response.lower()
        
        # 액션 추출 (select_question 또는 new_question만)
        if 'select_question' in response_lower:
            result['action'] = 'select_question'
            # QID 추출 시도
            for q in similar_context_questions + doc_id_questions:
                if q['qid'] in response:
                    result['selected_qid'] = q['qid']
                    result['selected_question'] = q['question']
                    break
        else:
            result['action'] = 'new_question'
        
        return result


class ReconfirmationGenerator:
    """LLM_2: 재확인할 질문 선택 및 재확인 질문 작성"""
    
    def __init__(self, provider: str = "openai", model: str = "gpt-4o-mini"):
        self.llm = QAModel(provider=provider, model=model)
    
    def select_and_generate_reconfirmation(
        self,
        conversation_history: List[Dict[str, str]],
        qid_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        qid_history와 conversation_history를 보고 재확인할만한 질문을 선택하고 재확인 질문 생성
        
        Args:
            conversation_history: 대화 히스토리 (질문-답변 쌍)
            qid_history: 각 턴의 qid와 question을 포함하는 리스트
                [{'qid': Optional[str], 'question': str}, ...]
            
        Returns:
            {
                'selected_qid': Optional[str],  # 재확인할 질문의 qid
                'reconfirmation_question': str  # 생성된 재확인 질문
            }
        """
               
        # 간단하게 히스토리 텍스트 생성
        history_items = []
        start_idx = max(0, len(qid_history) - 10)  # 최근 10개만
        
        for turn_num, qid_item in enumerate(qid_history[start_idx:], start=start_idx):
            # conversation_history에서 답변 가져오기 (user: 2*turn_num, assistant: 2*turn_num+1)
            answer_idx = 2 * turn_num + 1
            answer = ""
            if answer_idx < len(conversation_history):
                answer = conversation_history[answer_idx]['content'][:300]
            
            qid = qid_item.get('qid')
            question = qid_item.get('question', '')
            
            history_items.append(
                f"턴 {turn_num + 1}: {'[QID: ' + qid + ']' if qid else '[새로 생성된 질문]'}\n질문: {question}\n답변: {answer}"
            )
        
        history_text = "\n\n".join(history_items)
        
        prompt = f"""다음 대화 히스토리를 보고, LLM 응답의 일관성을 가장 효과적으로 평가할 수 있는 질문을 선택하고, 그에 대한 재확인 질문을 작성하세요.

대화 히스토리 (질문-답변 쌍):
{history_text}

재확인할만한 질문 선택 기준:
1. 핵심적이고 중요한 정보를 다루는 질문
2. 시간적 간격이 있는 질문 (최근 질문보다는 이전 대화에서 다뤘던 질문 중에서 선택)
3. 이전 답변과 비교하여 일관성을 검증하기 적합한 질문
4. 후속 질문들의 답변에 근거가 된, 대화의 초석이 되는 정보를 묻는 질문

재확인 질문 작성 원칙:
1. 원래 질문의 핵심 의도를 유지하면서 다른 표현으로 재구성 (단순 반복 금지)
2. 답변의 일관성을 효과적으로 검증할 수 있도록 작성
3. 사람이 실제 대화에서 재확인하는 것처럼 자연스럽게 작성
   - '아까', '이전에', '방금', '앞서' 등의 시간 지칭어를 적절히 활용
   - 재확인하는 톤으로 작성

응답 형식:
선택한 QID: [qid]
선택한 질문: [선택한 질문]
재확인 질문: [생성된 재확인 질문]"""
        
        messages = [
            {
                "role": "system",
                "content": "당신은 대화의 일관성을 확인하기 위해 재확인할 질문을 선택하고 재확인 질문을 작성하는 AI입니다."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        response = self.llm.call_llm(messages, temperature=0.7)
        
        # 응답 파싱
        result = self._parse_response(response, qid_history)
        
        # 재확인 질문이 없으면 생성
        if not result['reconfirmation_question']:
            selected_question = result.get('selected_question') or (qid_history[-1]['question'] if qid_history else '')
            if selected_question:
                prompt = f"""다음 질문을 자연스럽게 재확인하는 질문으로 변환하세요.

원본 질문: {selected_question}

재확인 질문만 출력하세요 (설명 없이)."""
                
                messages = [
                    {
                        "role": "system",
                        "content": "당신은 질문을 재확인하는 질문으로 변환하는 AI입니다."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
                
                reconfirm_response = self.llm.call_llm(messages, temperature=0.7)
                result['reconfirmation_question'] = reconfirm_response.strip()
            else:
                result['reconfirmation_question'] = '이전 답변에 대해 다시 확인하고 싶은데요.'
        
        return result
    
    def _parse_response(
        self,
        response: str,
        qid_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """LLM 응답 파싱"""
        result = {
            'selected_qid': None,
            'reconfirmation_question': '',
            'selected_question': None
        }
        
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('선택한 QID:') or line.startswith('QID:'):
                qid_str = line.split(':', 1)[1].strip()
                if qid_str and qid_str != '없음':
                    result['selected_qid'] = qid_str
                    # qid_history에서 해당 질문 찾기
                    for item in qid_history:
                        if item.get('qid') == qid_str:
                            result['selected_question'] = item['question']
                            break
            elif line.startswith('선택한 질문:'):
                question = line.split(':', 1)[1].strip()
                if question and question != '없음':
                    result['selected_question'] = question
            elif line.startswith('재확인 질문:'):
                reconfirm = line.split(':', 1)[1].strip()
                result['reconfirmation_question'] = reconfirm
        
        # QID가 있지만 질문을 찾지 못한 경우, qid_history에서 찾기
        if result['selected_qid'] and not result['selected_question']:
            for item in qid_history:
                if item.get('qid') == result['selected_qid']:
                    result['selected_question'] = item['question']
                    break
        
        # 질문이 없으면 선택된 질문 사용
        if not result['selected_question'] and qid_history:
            # 응답에서 질문 번호 찾기 시도
            for item in qid_history:
                if item['question'] in response:
                    result['selected_qid'] = item.get('qid')
                    result['selected_question'] = item['question']
                    break
            
            # 여전히 없으면 가장 최근 질문 사용
            if not result['selected_question']:
                latest = qid_history[-1]
                result['selected_qid'] = latest.get('qid')
                result['selected_question'] = latest['question']
        
        return result


class QuestionAnswerExtractor:
    """LLM_3: paragraph에서 reference 문장 지정 후 질문과 정답 추출"""
    
    def __init__(self, provider: str = "openai", model: str = "gpt-4o-mini"):
        self.llm = QAModel(provider=provider, model=model)
    
    def extract_question_answer(
        self,
        paragraph: str,
        conversation_history: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        주어진 paragraph에서 reference 문장을 지정하고 질문과 정답을 함께 추출
        
        Args:
            paragraph: 추출할 paragraph 텍스트
            conversation_history: 대화 히스토리 (맥락 파악용)
            
        Returns:
            {
                'reference_sentence': str,  # 참조 문장
                'question': str,  # 생성된 질문
                'answer': str  # 정답
            }
        """
        history_text = "\n".join([
            f"{'사용자' if turn['role'] == 'user' else '어시스턴트'}: {turn['content']}"
            for turn in conversation_history[-3:]  # 최근 3턴만
        ]) if conversation_history else "없음"
        
        prompt = f"""다음 paragraph를 읽고, 대화에 이어질 자연스러운 질문과 정답을 추출하세요.

대화 히스토리:
{history_text}

Paragraph:
{paragraph}

질문-정답 추출 규칙
1. 질문은 대화 히스토리에 포함된 모든 이전 질문 및 내용과 절대 겹치지 않아야 하며, 반드시 새로운 정보를 묻는 내용이어야 합니다.
2. 추출된 [정답]의 근거가 되는 [Paragraph] 내의 문장 하나를 반드시 그대로 지정해야 합니다.
3. 추출된 [질문]은 지정된 [참조 문장]만으로도 명확하고 완전하게 답변 가능해야 합니다.
4. 질문은 이전 대화 내용의 핵심 키워드나 맥락과 가장 밀접하게 관련되어야 합니다.

다음 형식으로 응답하세요:
참조문장: [paragraph에서 답변의 근거가 되는 문장]
질문: [paragraph를 기반으로 한 자연스러운 질문]
정답: [질문에 대한 정답]"""
        
        messages = [
            {
                "role": "system",
                "content": "당신은 paragraph에서 참조 문장을 지정하고 질문과 정답을 함께 추출하는 AI입니다."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        response = self.llm.call_llm(messages, temperature=0.5)
        
        # 응답 파싱
        result = self._parse_response(response)
        
        return result
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """LLM 응답 파싱"""
        result = {
            'reference_sentence': '',
            'question': '',
            'answer': ''
        }
        
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('참조문장:'):
                result['reference_sentence'] = line.replace('참조문장:', '').strip()
            elif line.startswith('질문:'):
                result['question'] = line.replace('질문:', '').strip()
            elif line.startswith('정답:'):
                result['answer'] = line.replace('정답:', '').strip()
        
        return result


class QuestionRewriter:
    """LLM_4: 질문을 자연스럽게 변환 (coreference, 문장 간소화)"""
    
    def __init__(self, provider: str = "openai", model: str = "gpt-4o-mini"):
        self.llm = QAModel(provider=provider, model=model)
    
    def rewrite_question(
        self,
        question: str,
        conversation_history: List[Dict[str, str]]
    ) -> str:
        """
        질문을 진짜 사람처럼 자연스럽게 변환
        
        Args:
            question: 원본 질문
            conversation_history: 대화 히스토리 (coreference 해결용)
            
        Returns:
            재작성된 질문
        """
        history_text = "\n".join([
            f"{'사용자' if turn['role'] == 'user' else '어시스턴트'}: {turn['content']}"
            for turn in conversation_history[-5:]  # 최근 5턴만
        ]) if conversation_history else "없음"
        
        prompt = f"""아래 대화 히스토리를 바탕으로, 원본 질문을 실제 사람이 대화를 이어나가는 것처럼 자연스러운 구어체로 패러프레이징(paraphrasing)하세요.
        

대화 히스토리:
{history_text}

원본 질문: {question}

변환 시 고려사항 (우선순위 순):
1. 원본 의미 완전 보존: 질문의 핵심 의미와 정보를 절대 변경하거나 훼손하지 마세요. 이는 패러프레이징이지 의미 변경이 아닙니다. 원본 질문이 묻는 바를 정확히 유지해야 합니다.

2. 대명사 사용: 
   - 앞서 언급되어 이미 알고 있는 대상(상품, 개념, 장소, 시간, 사람, 기관 등)은 '거기', '그거', '그 기관', '그 해', '걔네' 등으로 자연스럽게 바꾸세요.
   - 단, 질문의 핵심 대상은 대명사로 뭉뚱그리지 말고 원래 단어를 유지하거나 최소한으로만 줄이세요.
   - Bad: "그거(삼성전자)의 그거(영업이익)는 어때?" (너무 모호함)
   - Good: "거기(삼성전자) 영업이익은 어때?" (주체는 대명사, 핵심 소재는 유지)

3. 생략 활용: 
   - 대화 맥락에서 100% 명확히 유추 가능한 경우에만 생략하세요.
   - 생략으로 인해 무엇을 묻는지 불분명해지면 생략하지 마세요.
   - Bad: "그 프로젝트의 예산 규모는 얼마야?" -> "얼마야?" (무엇을 묻는지 불명확)
   - Good: "그 프로젝트의 예산 규모는 얼마야?" -> "예산 규모는?" (핵심 정보 유지)
   - Good: "삼성전자의 2023년 영업이익은 얼마인가요?" -> "2023년 영업이익은?" (앞서 삼성전자 언급됨)

4. 구어체 표현: 
   - 문어체를 피하고 자연스러운 구어체를 사용하세요.
   - 실제 사람처럼 간결하고 자연스럽게 표현하세요.
   - "그럼,", "근데," 같은 접속사를 적절히 사용하여 부드럽게 연결하세요.
   - "아,", "음," 같은 감탄사는 자연스러운 상황에서만 사용하세요.

5. 대화의 일관성을 위해, 대화 히스토리에서 시작된 존댓말/반말 스타일을 반드시 유지하세요.

원칙: 원본 의미 보존 > 자연스러운 표현 > 대명사/생략 사용. 대명사나 생략으로 인해 의미가 모호해지거나 핵심 정보가 손실될 수 있다면, 원본 표현을 그대로 사용하세요.

변환된 질문만 출력하세요 (설명 없이)."""
        
        messages = [
            {
                "role": "system",
                "content": "당신은 질문을 실제 사람이 대화를 이어나가는 것처럼 자연스럽게 대화 맥락에 맞게 패러프레이징(paraphrasing)하는 AI입니다."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        response = self.llm.call_llm(messages, temperature=0.7)
        return response.strip()


class AnswerGenerator:
    """LLM_5: RAG 기반 답변 생성"""
    
    def __init__(self, provider: str = "openai", model: str = "gpt-4o-mini"):
        self.llm = QAModel(provider=provider, model=model)
    
    def generate_answer(
        self,
        question: str,
        passages: List[Dict[str, Any]],
        conversation_history: List[Dict[str, str]],
        evidence_text: Optional[str] = None
    ) -> str:
        """
        RAG 기반으로 답변 생성
        
        Args:
            question: 현재 질문
            passages: retriever가 반환한 passage 리스트
            conversation_history: 대화 히스토리
            evidence_text: 정답을 유추할 수 있는 evidence 텍스트 (옵션)
            
        Returns:
            생성된 답변
        """
        # 대화 히스토리 포맷팅
        history_text = "\n".join([
            f"{'사용자' if turn['role'] == 'user' else '어시스턴트'}: {turn['content']}"
            for turn in conversation_history
        ]) if conversation_history else "없음"
        
        # Passages 포맷팅
        passages_text = "\n\n".join([
            f"[Passage {i+1}]\nContext ID: {p.get('context_id', 'N/A')}\n{p.get('context', '')}"
            for i, p in enumerate(passages)
        ])
        
        # Evidence text 추가
        evidence_section = ""
        if evidence_text:
            evidence_section = f"\n\n[Evidence Text]\n{evidence_text}"
        
        prompt = f"""다음은 사용자와의 대화입니다. 주어진 passage와 evidence만을 사용하여 정확하게 답변하세요.

대화 히스토리:
{history_text}

현재 질문: {question}

관련 Passage:
{passages_text}{evidence_section}

지침:
1. 주어진 Passage와 Evidence Text만을 사용하여 답변하세요.
2. 외부 지식(parametric knowledge)은 사용하지 마세요.
3. 주어진 정보만으로 답변할 수 없다면 "주어진 정보만으로는 대답 불가능합니다."라고 명시적으로 응답하세요.
4. 답변은 간결하고 정확하게 작성하세요.

답변:"""
        
        messages = [
            {
                "role": "system",
                "content": "당신은 주어진 passage와 evidence만을 사용하여 정확한 답변을 제공하는 RAG 기반 AI 어시스턴트입니다."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        response = self.llm.call_llm(messages, temperature=0.3)
        return response.strip()

