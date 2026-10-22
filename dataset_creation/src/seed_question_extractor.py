#!/usr/bin/env python3
"""
Seed Question 추출 스크립트
- doc_id별로 question_id 리스트 수집
- LLM으로 각 doc_id의 적절한 seed question 선택
- JSON 형식으로 출력
"""

import json
import argparse
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from qa_model import QAModel
from tqdm import tqdm

def collect_questions_by_doc(input_data):
    """doc_id별로 question_id와 question 수집"""
    doc_questions = defaultdict(list)
    
    # 새로운 형식: qid_data 안에 question_id를 키로 하는 엔트리들이 있음
    qid_data = input_data.get('qid_data', {})
    
    for question_id, entry in qid_data.items():
        doc_id = entry.get('doc_id')
        if not doc_id:
            continue
        
        question = entry.get('question')
        if question:
            doc_questions[doc_id].append({
                'question_id': question_id,
                'question': question
            })
    
    return dict(doc_questions)

def select_seed_question(provider, model, doc_id, questions):
    """LLM으로 seed question 선택 및 이유 추출"""
    if len(questions) == 1:
        return {
            'doc_id': doc_id,
            'question_id': questions[0]['question_id'],
            'question': questions[0]['question'],
            'reason': '단일 질문이므로 자동 선택'
        }
    
    # 각 스레드마다 LLM 인스턴스 생성
    llm = QAModel(provider=provider, model=model)
    
    questions_text = "\n".join([
        f"{i+1}. [ID: {q['question_id']}] {q['question']}"
        for i, q in enumerate(questions)
    ])
    
    prompt = f"""다음은 동일한 문서(doc_id: {doc_id})에서 나온 질문들입니다.
대화를 시작하는 첫 질문(seed question)으로 가장 적절한 질문을 선택해주세요.

**선택 기준:**
1. 독립성: 이전 대화 맥락 없이도 완전히 이해 가능한가?
2. 핵심성: 문서의 주요 내용이나 핵심 주제를 다루는가? (너무 세부적이지 않은가?)
3. 자연스러움: 실제 사용자가 처음으로 던질 법한 질문인가?
4. 명확성: 질문의 의도가 명확한가?
5. 확장성: 이 질문에서 시작하여 대화를 자연스럽게 이어갈 수 있는가?

**질문 목록:**
{questions_text}

**응답 형식 (중요: 질문 목록 앞의 번호를 사용하세요, 예: 1, 2, 3...):**
선택: [질문 목록의 번호]
이유: [선택한 이유를 2-3문장으로 설명]

**응답 예시:**
선택: 2
이유: 이 질문은 문서의 핵심 주제를 다루면서도 독립적으로 이해 가능하며, 사용자가 처음으로 던질 법한 자연스러운 질문입니다. 또한 이 질문에서 시작하여 문서의 세부 내용에 대한 후속 질문으로 자연스럽게 이어갈 수 있습니다."""

    messages = [
        {
            "role": "system",
            "content": "당신은 대화의 첫 시작 질문(seed question)을 선택하는 전문가입니다. 각 질문을 신중하게 분석하여 가장 적절한 시작 질문을 선택하고, 그 이유를 명확하게 설명합니다."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    try:
        response = llm.call_llm(messages)
        
        # 응답에서 번호와 이유 추출
        choice_num = None
        reason = ""
        
        # question_id로 인덱스 매핑 생성 (fallback용)
        question_id_map = {str(q['question_id']): idx for idx, q in enumerate(questions)}
        
        # 번호 추출
        lines = response.split('\n')
        for line in lines:
            line_lower = line.lower().strip()
            if '선택' in line_lower or 'choice' in line_lower:
                # 콜론 뒤의 숫자 찾기
                if ':' in line:
                    after_colon = line.split(':', 1)[1].strip()
                    for word in after_colon.split():
                        if word.isdigit():
                            num_str = word
                            num = int(word)
                            # 먼저 질문 목록 번호(1, 2, 3, ...)로 확인
                            if 1 <= num <= len(questions):
                                choice_num = num - 1
                                break
                            # question_id로 확인 (fallback)
                            elif num_str in question_id_map:
                                choice_num = question_id_map[num_str]
                                break
                else:
                    # 콜론이 없으면 라인 전체에서 숫자 찾기
                    for word in line.split():
                        if word.isdigit():
                            num_str = word
                            num = int(word)
                            if 1 <= num <= len(questions):
                                choice_num = num - 1
                                break
                            elif num_str in question_id_map:
                                choice_num = question_id_map[num_str]
                                break
                if choice_num is not None:
                    break
        
        # 이유 추출
        reason_started = False
        for line in lines:
            line_lower = line.lower().strip()
            if '이유' in line_lower or 'reason' in line_lower:
                reason_started = True
                # "이유:" 또는 "reason:" 이후의 텍스트 추출
                parts = line.split(':', 1)
                if len(parts) > 1:
                    reason = parts[1].strip()
                continue
            if reason_started and line.strip():
                if reason:
                    reason += " " + line.strip()
                else:
                    reason = line.strip()
        
        if choice_num is not None:
            selected = questions[choice_num].copy()
            selected['doc_id'] = doc_id
            selected['reason'] = reason if reason else '이유 추출 실패'
            return selected
        else:
            return None
    except Exception as e:
        return None

def main():
    parser = argparse.ArgumentParser(description="Seed question 추출")
    parser.add_argument(
        "--input_data",
        type=str,
        default="data/processed_data.json",
        help="processed_data.json 파일 경로"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="seed_questions.json",
        help="출력 JSON 파일 경로"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        help="LLM 모델 이름 (논문: gpt-5-mini)"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai", "google"],
        help="LLM provider"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=8,
        help="병렬 처리 최대 워커 수"
    )
    
    args = parser.parse_args()
    
    print(f"데이터 로딩 중...")
    with open(args.input_data, 'r', encoding='utf-8-sig') as f:
        input_data = json.load(f)
    
    print(f"doc_id별 질문 수집 중...")
    doc_questions = collect_questions_by_doc(input_data)
    print(f"Doc ID 개수: {len(doc_questions):,}")
    
    print(f"LLM으로 seed question 선택 중... (병렬 처리: {args.max_workers} workers)")
    
    # 병렬 처리
    seed_questions = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # 모든 작업 제출
        future_to_doc = {
            executor.submit(select_seed_question, args.provider, args.model, doc_id, questions): doc_id
            for doc_id, questions in doc_questions.items()
        }
        
        # 완료된 작업 처리
        for future in tqdm(as_completed(future_to_doc), total=len(future_to_doc), desc="처리 중"):
            doc_id = future_to_doc[future]
            try:
                selected = future.result()
                if selected is not None:
                    seed_questions.append({
                        'doc_id': selected['doc_id'],
                        'question_id': selected['question_id'],
                        'question': selected['question'],
                        'reason': selected.get('reason', '이유 없음')
                    })
            except Exception as e:
                print(f"Warning: doc_id {doc_id} 처리 중 오류 발생: {str(e)}")
                continue
        
    # JSON 저장
    # 경로와 파일명 분리하여 파일명 앞에만 모델명 추가
    output_dir = os.path.dirname(args.output)
    output_filename = os.path.basename(args.output)
    output_path = os.path.join(output_dir, args.model + "_" + output_filename) if output_dir else args.model + "_" + output_filename
    
    # 디렉토리가 없으면 생성
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(seed_questions, f, ensure_ascii=False, indent=2)
    
    print(f"\n완료! {len(seed_questions):,}개의 seed question을 {output_path}에 저장했습니다.")

if __name__ == "__main__":
    main()