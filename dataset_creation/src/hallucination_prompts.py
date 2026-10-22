#!/usr/bin/env python3
"""
Hallucination Injection Prompts
환각 주입을 위한 프롬프트 템플릿 모음
"""


class HallucinationPrompts:
    """환각 주입 프롬프트 템플릿"""
    
    # ========== 위치 결정 프롬프트 ==========
    
    @staticmethod
    def get_injection_position_system_prompt() -> str:
        """위치 결정 시스템 프롬프트"""
        return "당신은 대화에 환각을 주입할 최적 위치를 결정하는 전문가입니다. 각 환각 유형에 가장 적합한 턴을 선택하세요."
    
    @staticmethod
    def get_injection_position_prompt(
        dialogue_text: str, 
        hallucination_types: list,
        hallucination_definitions: str = "",
        constraint_note: str = ""
    ) -> str:
        """환각 주입 위치 결정 프롬프트"""
        type_list = [
            f"{i}. {h_type}"
            for i, h_type in enumerate(hallucination_types, 1)
        ]
        example_turns = list(range(1, len(hallucination_types) + 1))
        definitions_block = f"\n환각 유형 정의:\n{hallucination_definitions}\n" if hallucination_definitions else ""

        return f"""대화:
{dialogue_text}

주입할 환각 유형 (순서대로):
{chr(10).join(type_list)}
{definitions_block}
각 환각을 서로 다른 턴에 주입하세요. 반환 순서가 유형 순서와 매핑됩니다.{constraint_note}

JSON 형식:
{{
  "turn_indices": {example_turns}
}}"""
    
    # ========== Contradictory 프롬프트 ==========
    
    @staticmethod
    def get_contradictory_system_prompt() -> str:
        """Contradictory 시스템 프롬프트"""
        return "답변을 미묘하게 수정하여 컨텍스트와 충돌을 만드는 전문가입니다. 최소 수정으로 자연스러운 환각을 생성하세요."

    @staticmethod
    def get_contradictory_prompt(
        user_question: str,
        original_answer: str,
        passage_text: str,
        hallucination_definition: str = ""
    ) -> str:
        """Contradictory"""
        definition_block = f"\n환각 유형 정의:\n{hallucination_definition}\n" if hallucination_definition else ""

        return f"""다음 질문과 답변이 주어집니다.

질문: {user_question}
답변: {original_answer}

관련 문서:
{passage_text}
{definition_block}
답변을 최소한으로만 수정하여, 관련 문서/컨텍스트와 미묘하게 충돌(모순)되도록 환각을 만드세요.
- 전체 답변을 다시 작성하지 말고, 핵심 금융전문용어/단어/수식어/숫자 중 1~2개만 바꿔서 의미가 틀어지게 하세요. 
- 검증 불가능한 정보(문장이나 구)를 추가하지말고, 기존 답변 구조와 길이를 유지하세요.
- 알아차리기 어려울 정도로 미묘하게 수정하여, 답변이 자연스럽게 보이도록 하세요.
- 답변의 주어는 최대한 바꾸지마세요. 미묘하게 수정한다고 해서 질문과 관련 없는 대답을 하면 안 됩니다. 

다음과 같은 방법을 참고할 수 있습니다:
- 금융 전문 용어 오해: 답변에 있는 어려운 금융 전문 용어를 유사하지만 다른 금융 전문 용어로 바꿔 의미를 왜곡 (예: "신용대출"↔"담보대출", "원리금상환"→"원금균등상환", 신용대출"→"담보대출", "이자율"→"금리")
- 한정어/수식어 변경하거나 누락: 오직/만/최소/최대/이상/이하/초과/미만/부분적으로 등 한정어/수식어를 변경/누락
- 숫자 오류: 기간/금리/한도/횟수 등 핵심 숫자를 비슷한 범위의 다른 숫자로 교체하거나, 컨텍스트 내 다른 관련 숫자로 교체 (예: "금리"를 묻는데 문서의 "이율" 관련 숫자를 가져와서 모순을 만듦)

다음 형식으로 답변하세요 (먼저 근거를 설명하고, 그 다음 최종 답변을 작성하세요):
근거: [어떤 부분을 왜 수정했는지 설명]
답변: [수정된 답변]"""
    
    @staticmethod
    def get_contradictory_inconsistency_prompt(history_text: str, user_question: str, 
                                               original_answer: str, passage_text: str,
                                               hallucination_definition: str = "") -> str:
        """Contradictory - Inconsistency"""
        definition_block = f"\n환각 유형 정의:\n{hallucination_definition}\n" if hallucination_definition else ""
        
        return f"""다음 대화 히스토리와 현재 답변이 주어집니다.

대화 히스토리:
{history_text}

현재 질문: {user_question}
현재 답변: {original_answer}

관련 문서:
{passage_text}
{definition_block}
현재 답변을 이전 대화와 모순되도록 미묘하고 알아차리기 쉽지않은 환각을 만들도록 최소한으로 수정하세요.
- 전체 답변을 다시 작성하지 말고, 핵심 금융전문용어/단어/수식어/숫자 중 1~2개만 바꿔서 의미가 틀어지게 하세요.
- 검증 불가능한 정보(문장이나 구)를 추가하지말고, 기존 답변 구조와 길이를 유지하세요.
- 알아차리기 어려울 정도로 미묘하게 수정하여, 답변이 자연스럽게 보이도록 하세요.
- 답변의 주어는 최대한 바꾸지마세요. 미묘하게 수정한다고 해서 질문과 관련 없는 대답을 하면 안 됩니다. 

다음과 같은 방법을 참고할 수 있습니다:
- 금융 전문 용어 오해: 답변에 있는 어려운 금융 전문 용어를 유사하지만 다른 금융 전문 용어로 바꿔 의미를 왜곡 (예: "신용대출"↔"담보대출", "원리금상환"→"원금균등상환", 신용대출"→"담보대출", "이자율"→"금리")
- 한정어/수식어 변경하거나 누락: 오직/만/최소/최대/이상/이하/초과/미만/부분적으로 등 한정어/수식어를 변경/누락
- 숫자 오류: 기간/금리/한도/횟수 등 핵심 숫자를 비슷한 범위의 다른 숫자로 교체하거나, 컨텍스트 내 다른 관련 숫자로 교체 (예: "금리"를 묻는데 문서의 "이율" 관련 숫자를 가져와서 모순을 만듦)

다음 형식으로 답변하세요 (먼저 근거를 설명하고, 그 다음 최종 답변을 작성하세요):
근거: [어떤 부분을 왜 수정했는지 설명]
답변: [수정된 답변]"""
    
    # ========== Irrelevance 프롬프트 ==========
    
    @staticmethod
    def get_irrelevance_system_prompt() -> str:
        """Irrelevance 시스템 프롬프트"""
        return "질문 의도와 어긋나는 답변을 만드는 전문가입니다. 핵심 토픽은 유지하되 질문과 관련 없는 답을 하세요."
    
    @staticmethod
    def get_irrelevance_prompt(history_text: str, user_question: str, 
                               original_answer: str, passage_text: str,
                               hallucination_definition: str = "") -> str:
        """Irrelevance"""
        definition_block = f"\n환각 유형 정의:\n{hallucination_definition}\n" if hallucination_definition else ""
        
        return f"""다음 대화 히스토리와 현재 질문, 답변이 주어집니다.

대화 히스토리:
{history_text}

현재 질문: {user_question}
현재 답변: {original_answer}

관련 문서:
{passage_text}
{definition_block}
같은 주제 내에서 질문의 핵심 키워드를 유지하면서 질문 의도와 어긋나도록 답변을 최소한으로만 수정하세요.

다음과 같은 방법을 참고할 수 있습니다:
- 조건 불만족: 질문의 조건(대상/기간/범위/자격)을 만족하지 않도록 수정 (예: "3개월"을 묻는데 "1년"으로 답하도록 변경)
- 주제 이탈: 같은 주제 내에서 질문과 다른 측면에 대해 답변 (예: "신청 방법"을 묻는데 "신청 기간"에 대해 답변)
- 질문 오해: 문법적 요소(조사/의문사)를 오해하여 다르게 답변 (예: "부터"를 "까지"로 오해하여 "언제까지"에 대해 답변)

다음 형식으로 답변하세요 (먼저 근거를 설명하고, 그 다음 최종 답변을 작성하세요):
근거: [어떤 방법을 사용하여 어떤 부분을 왜 수정했는지 설명]
답변: [수정된 답변]"""
    
    # ========== Unverifiable 프롬프트 ==========
    
    @staticmethod
    def get_unverifiable_system_prompt() -> str:
        """Unverifiable 시스템 프롬프트"""
        return "근거 없는 정보를 자연스럽게 추가하는 전문가입니다. 3-5어절의 짧은 phrase만 삽입하세요."
    
    @staticmethod
    def get_unverifiable_prompt(
        user_question: str,
        original_answer: str,
        passage_text: str,
        hallucination_definition: str = ""
    ) -> str:
        """Unverifiable (subtype 없음)"""
        definition_block = f"\n환각 유형 정의:\n{hallucination_definition}\n" if hallucination_definition else ""
        
        return f"""다음 질문과 답변이 주어집니다.

질문: {user_question}
원본 답변: {original_answer}

관련 문서:
{passage_text}
{definition_block}
원본 답변은 올바릅니다. 이 답변에 문서/컨텍스트로 검증 불가한 정보를 미묘하고 알아차리기 쉽지않게 추가하세요.
- 한 문장 전체를 추가하지 말고, 5어절 미만의 짧은 내용을 추가하세요.
- 원본 답변의 구조와 내용은 최대한 유지하세요.
- 추가된 정보가 실제 정보처럼 보이도록 알아차리기 어려울 정도로 미묘하게 작성하세요.

다음과 같은 방법을 참고할 수 있습니다:
- 근거 없는 상세 정보: 문서에 없는 절차/조건/예외/기간 등 짧은 디테일을 끼워 넣기
- 근거 없는 주관적 의견: 문서에 없는 평가/추천을 사실처럼 보이게 끼워 넣기

다음 형식으로 답변하세요 (먼저 근거를 설명하고, 그 다음 최종 답변을 작성하세요):
근거: [어떤 부분을 왜 수정했는지 설명]
답변: [수정된 답변]"""