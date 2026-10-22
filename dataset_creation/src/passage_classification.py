#!/usr/bin/env python3
"""
Passage Classification

Step 1 (context_sufficiency): 컨텍스트(C)만으로 질문(Q)의 정답(A)을 도출할 수 있는지 판정
Step 2 (clue_presence): 단서(R)가 컨텍스트(C)에 명시적으로 존재/지지되는지 판정
"""

import json
import re
from typing import Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from qa_model import QAModel


class PassageClassifier:
    """Passage 분류 및 검증 파이프라인"""
    
    def __init__(self, qa_model: QAModel):
        """
        Args:
            qa_model: QAModel 인스턴스
        """
        self.qa_model = qa_model
    
    def validate(self, context: str, question: str, answer: str, clue: str,
                 early_skip_on_step1_yes: bool = False) -> Dict[str, Any]:
        """
        2단계 검증 파이프라인 실행 (step1, step2 병렬)

        Args:
            context: 배경 문맥 또는 지문 (C)
            question: 질문 (Q)
            answer: 정답 (A)
            clue: 정답의 핵심 근거가 되는 단서 (R)
            early_skip_on_step1_yes: True이면 step1=YES일 때 step2를 스킵
                (no-answer passage 검색 시, step1=YES이면 정답이 있으므로 step2 불필요)

        Returns:
            검증 결과 딕셔너리
        """
        if early_skip_on_step1_yes:
            # step1 먼저 실행하고, YES이면 step2 스킵
            step1_result = self._validate_context_sufficiency(context, question, answer)
            if step1_result["result"].upper() == "YES":
                step2_result = {
                    "result": "SKIPPED",
                    "confidence": "N/A",
                    "reason": "step1=YES → step2 skipped (early_skip_on_step1_yes)"
                }
            else:
                step2_result = self._validate_clue_presence(context, clue)
        else:
            # step1, step2 병렬 실행
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_step1 = executor.submit(self._validate_context_sufficiency, context, question, answer)
                future_step2 = executor.submit(self._validate_clue_presence, context, clue)
                step1_result = future_step1.result()
                step2_result = future_step2.result()

        return {
            "validation_steps": {
                "step1_sufficiency": step1_result,
                "step2_presence": step2_result
            }
        }
    
    def _validate_context_sufficiency(self, context: str, question: str, answer: str) -> Dict[str, str]:
        """Step 1: Context Sufficiency Validation (Answerability from Context only)"""
        prompt = f"""주어진 컨텍스트(C)만으로 질문(Q)의 정답(A)을 도출할 수 있습니까?

데이터:
- Context (C): {context}
- Question (Q): {question}
- Answer (A): {answer}

중요 판단 기준:
- YES: 컨텍스트에 질문의 정답(A)을 확정할 수 있는 정보가 명시적으로 존재하는 경우
- NO: 정답을 확정할 정보가 부족/모호하거나, 외부 지식/추론/추가 정보가 필요해 정답을 단정할 수 없는 경우

신뢰도 결정 지침:
- HIGH: 판단(YES/NO)에 어떠한 추론이나 해석의 개입 없이 컨텍스트의 문장 자체로 결론이 명백히 도출되는 경우.
- LOW: 판단을 내리기 위해 논리적 추론이나 의미 해석이 필요하거나, 컨텍스트가 부분적으로만 관련되어 판단에 일말의 불확실성이 존재하는 경우. 

판단을 수행하고, 다음 형식으로 응답하세요:
판단: [YES/NO]
신뢰도: [HIGH/LOW]
근거: [컨텍스트(C)에서 정답(A)을 도출할 수 있는/없는 판단 근거가 되는 부분을 그대로 인용하여 제시하세요]"""
        
        response = self.qa_model.classify(prompt)
        result = self._parse_response_with_confidence(response, ["YES", "NO"])
        return {
            "result": result["result"],
            "confidence": result["confidence"],
            "reason": result["reason"]
        }
    
    
    def _validate_clue_presence(self, context: str, clue: str) -> Dict[str, str]:
        """Step 2: Clue Presence Check"""
        prompt = f"""핵심 단서(R)가 컨텍스트(C) 내에 명시적으로 나타나거나, 단서의 내용이 컨텍스트에서 직접적으로 뒷받침되는지 확인하세요.

데이터:
- Context (C): {context}
- Clue (R): {clue}

중요 판단 기준:
- YES: 단서(R)의 내용이 컨텍스트(C)에서 명확하게 나타나거나, 단서의 핵심 내용을 직접적으로 뒷받침하는 문장이 명시적으로 존재하는 경우
- NO: 단서의 내용이 암시적으로만 존재하거나, 간접적인 관련성만 있고 단서를 직접적으로 뒷받침하는 내용이 부족한 경우

신뢰도 결정 지침:
- HIGH: 판단(YES/NO)에 어떠한 추론이나 해석의 개입 없이 컨텍스트의 문장 자체로 결론이 명백히 도출되는 경우.
- LOW: 판단을 내리기 위해 논리적 추론이나 의미 해석이 필요하거나, 단서의 내용과 컨텍스트의 문장이 부분적으로만 일치하여 판단에 일말의 불확실성이 존재하는 경우. 

판단을 수행하고, 다음 형식으로 응답하세요:
판단: [YES/NO]
신뢰도: [HIGH/LOW]
근거: [단서(R)의 내용이 컨텍스트(C)에서 어떻게 나타나는지, 또는 단서를 직접적으로 뒷받침하는 부분을 그대로 인용하여 제시하세요. 단서가 발견되지 않으면 그 이유를 설명하세요]"""
        
        response = self.qa_model.classify(prompt)
        result = self._parse_response_with_confidence(response, ["YES", "NO"])
        
        return {
            "result": result["result"],
            "confidence": result["confidence"],
            "reason": result["reason"]
        }
    
    def check_answer_in_context(self, context: str, question: str, answer: str, clue: str,
                               early_skip_on_step1_yes: bool = False):
        """
        2단계 검증을 수행 (step1, step2)

        Args:
            context: 배경 문맥 또는 지문 (C)
            question: 질문 (Q)
            answer: 정답 (A)
            clue: 정답의 핵심 근거가 되는 단서 (R)
            early_skip_on_step1_yes: True이면 step1=YES일 때 step2 스킵

        Returns:
            (step1_result, step1_confidence, step2_result, step2_confidence, step1_reason, step2_reason)
        """
        validation_result = self.validate(context, question, answer, clue,
                                          early_skip_on_step1_yes=early_skip_on_step1_yes)
        
        step1 = validation_result["validation_steps"]["step1_sufficiency"]["result"]
        step1_confidence = validation_result["validation_steps"]["step1_sufficiency"]["confidence"]
        step2 = validation_result["validation_steps"]["step2_presence"]["result"]
        step2_confidence = validation_result["validation_steps"]["step2_presence"]["confidence"]
        step1_reason = validation_result["validation_steps"]["step1_sufficiency"]["reason"]
        step2_reason = validation_result["validation_steps"]["step2_presence"]["reason"]
        
        return step1, step1_confidence, step2, step2_confidence, step1_reason, step2_reason
    
    def _parse_response(self, response: str, valid_options: list) -> Dict[str, str]:
        """LLM 응답에서 판단 결과와 근거 추출"""
        result = "UNKNOWN"
        reason = ""
        
        # 판단 추출
        if "판단" in response:
            lines = response.split("\n")
            for line in lines:
                if "판단" in line and ":" in line:
                    judgment_text = line.split(":", 1)[1].strip()
                    
                    # 1. 먼저 대괄호 안의 값 확인 (예: [YES/NO] 또는 [YES])
                    bracket_match = re.search(r'\[([^\]]+)\]', line)
                    if bracket_match:
                        bracket_text = bracket_match.group(1).upper()
                        # 슬래시로 구분된 경우 각 값 확인
                        bracket_options = [opt.strip() for opt in bracket_text.split('/')]
                        for bracket_opt in bracket_options:
                            for option in valid_options:
                                if option.upper() == bracket_opt:
                                    result = option.upper()
                                    break
                            if result != "UNKNOWN":
                                break
                    
                    # 2. 대괄호 매칭 실패 시, 대괄호 제거 후 정확한 단어 매칭 (단어 경계 사용)
                    if result == "UNKNOWN":
                        judgment_cleaned = judgment_text.replace("[", "").replace("]", "").strip()
                        judgment_upper = judgment_cleaned.upper()
                        for option in valid_options:
                            option_upper = option.upper()
                            # 정확한 단어로 매칭 (앞뒤로 단어 경계 체크)
                            pattern = r'\b' + re.escape(option_upper) + r'\b'
                            if re.search(pattern, judgment_upper):
                                result = option.upper()
                                break
                    
                    # 3. 여전히 매칭 안되면 첫 번째 단어만 추출해서 시도
                    if result == "UNKNOWN":
                        judgment_cleaned = judgment_text.replace("[", "").replace("]", "").strip()
                        if judgment_cleaned:
                            words = judgment_cleaned.split()
                            if words:
                                first_word = words[0].upper()
                                for option in valid_options:
                                    if option.upper() == first_word:
                                        result = option.upper()
                                        break
                    
                    break
        
        # 근거 추출 (여러 줄 포함)
        if "근거" in response:
            lines = response.split("\n")
            for i, line in enumerate(lines):
                if "근거" in line and ":" in line:
                    reason = line.split(":", 1)[1].strip()
                    # 다음 줄들도 포함 (빈 줄이나 다른 섹션 시작 전까지)
                    for j in range(i + 1, len(lines)):
                        next_line = lines[j].strip()
                        if not next_line or "판단" in next_line or "신뢰도" in next_line:
                            break
                        reason += " " + next_line
                    break
        
        return {
            "result": result,
            "reason": reason
        }
    
    def _parse_response_with_confidence(self, response: str, valid_options: list) -> Dict[str, str]:
        """LLM 응답에서 판단 결과, 신뢰도, 근거 추출"""
        result = "UNKNOWN"
        confidence = "UNKNOWN"
        reason = ""
        
        lines = response.split("\n")
        
        # 판단 추출
        for line in lines:
            if "판단" in line and ":" in line:
                judgment_text = line.split(":", 1)[1].strip()
                
                # 1. 먼저 대괄호 안의 값 확인 (예: [YES/NO] 또는 [YES])
                bracket_match = re.search(r'\[([^\]]+)\]', line)
                if bracket_match:
                    bracket_text = bracket_match.group(1).upper()
                    bracket_options = [opt.strip() for opt in bracket_text.split('/')]
                    for bracket_opt in bracket_options:
                        for option in valid_options:
                            if option.upper() == bracket_opt:
                                result = option.upper()
                                break
                        if result != "UNKNOWN":
                            break
                
                # 2. 대괄호 매칭 실패 시, 대괄호 제거 후 정확한 단어 매칭
                if result == "UNKNOWN":
                    judgment_cleaned = judgment_text.replace("[", "").replace("]", "").strip()
                    judgment_upper = judgment_cleaned.upper()
                    for option in valid_options:
                        option_upper = option.upper()
                        pattern = r'\b' + re.escape(option_upper) + r'\b'
                        if re.search(pattern, judgment_upper):
                            result = option.upper()
                            break
                
                # 3. 여전히 매칭 안되면 첫 번째 단어만 추출해서 시도
                if result == "UNKNOWN":
                    judgment_cleaned = judgment_text.replace("[", "").replace("]", "").strip()
                    if judgment_cleaned:
                        words = judgment_cleaned.split()
                        if words:
                            first_word = words[0].upper()
                            for option in valid_options:
                                if option.upper() == first_word:
                                    result = option.upper()
                                    break
                break
        
        # 신뢰도 추출
        for line in lines:
            if "신뢰도" in line and ":" in line:
                confidence_text = line.split(":", 1)[1].strip()
                
                # 대괄호 안의 값 확인
                bracket_match = re.search(r'\[([^\]]+)\]', line)
                if bracket_match:
                    bracket_text = bracket_match.group(1).upper()
                    if "HIGH" in bracket_text:
                        confidence = "HIGH"
                    elif "LOW" in bracket_text:
                        confidence = "LOW"
                else:
                    # 대괄호 없이 직접 매칭
                    confidence_upper = confidence_text.upper()
                    if "HIGH" in confidence_upper:
                        confidence = "HIGH"
                    elif "LOW" in confidence_upper:
                        confidence = "LOW"
                break
        
        # 근거 추출 (여러 줄 포함)
        for i, line in enumerate(lines):
            if "근거" in line and ":" in line:
                reason = line.split(":", 1)[1].strip()
                # 다음 줄들도 포함 (빈 줄이나 다른 섹션 시작 전까지)
                for j in range(i + 1, len(lines)):
                    next_line = lines[j].strip()
                    if not next_line or "판단" in next_line or "신뢰도" in next_line:
                        break
                    reason += " " + next_line
                break
        
        return {
            "result": result,
            "confidence": confidence,
            "reason": reason
        }
    
    def classify_passage(
        self,
        qid: str,
        question: str,
        answer: str,
        clue: str,
        similar_context_ids: list,
        context_mapping: Dict[str, Dict[str, Any]],
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        하나의 질문에 대해 similar_context_id들을 분류 (2단계 검증 파이프라인 사용: step1, step2)
        
        Args:
            qid: 질문 ID
            question: 질문 텍스트
            answer: 정답 텍스트
            clue: 정답의 핵심 근거가 되는 단서 (R)
            similar_context_ids: 유사 컨텍스트 ID 리스트
            context_mapping: 컨텍스트 매핑 딕셔너리
            verbose: 상세 출력 여부
            
        Returns:
            {
                "question": "...",
                "answer": "...",
                "similar_contexts": [
                    {
                        "context_id": "...",
                        "context": "...",
                        "step1": "YES/NO",
                        "step1_confidence": "HIGH/LOW",
                        "step2": "YES/NO",
                        "step2_confidence": "HIGH/LOW",
                        "step1_reason": "근거 텍스트",
                        "step2_reason": "근거 텍스트",
                        "classification": "positive" or "similar"
                    }
                ]
            }
        """
        if verbose:
            print(f"Processing QID: {qid}")
            print(f"Question: {question[:80]}...")
            print(f"Similar contexts to check: {len(similar_context_ids)}")
        
        similar_contexts = []
        positive_count = 0
        similar_count = 0
        
        for similar_context_id in similar_context_ids:
                        
            context_info = context_mapping[similar_context_id]
            context = context_info.get('context', '')
            
            try:
                # 2단계 검증 수행 (step1, step2)
                step1, step1_confidence, step2, step2_confidence, step1_reason, step2_reason = self.check_answer_in_context(context, question, answer, clue)
                
                # step1과 step2 모두 통과해야 positive (더 엄격한 기준)
                step1_upper = step1.upper() if step1 else ""
                step2_upper = step2.upper() if step2 else ""
                passes_both_steps = (step1_upper == "YES" and step2_upper == "YES")
                
                classification = "positive" if passes_both_steps else "similar"
                
                if passes_both_steps:
                    positive_count += 1
                    if verbose:
                        print(f"✓ Positive (step1:{step1}[{step1_confidence}], step2:{step2}[{step2_confidence}])")
                else:
                    similar_count += 1
                    if verbose:
                        print(f"Similar (step1:{step1}[{step1_confidence}], step2:{step2}[{step2_confidence}])")
                
                # 결과 저장 (근거 및 confidence 포함)
                similar_contexts.append({
                    "context_id": similar_context_id,
                    "context": context,
                    "step1": step1,
                    "step1_confidence": step1_confidence,
                    "step2": step2,
                    "step2_confidence": step2_confidence,
                    "step1_reason": step1_reason,
                    "step2_reason": step2_reason,
                    "classification": classification
                })
            except Exception as e:
                if verbose:
                    print(f"Error: {e}")
                # 에러 발생 시 similar로 분류
                similar_count += 1
                similar_contexts.append({
                    "context_id": similar_context_id,
                    "context": context,
                    "step1": "UNKNOWN",
                    "step1_confidence": "UNKNOWN",
                    "step2": "UNKNOWN",
                    "step2_confidence": "UNKNOWN",
                    "step1_reason": "",
                    "step2_reason": "",
                    "classification": "similar",
                    "error": str(e)
                })
        
        if verbose:
            print(f"  ✓ Positive: {positive_count}, Similar: {similar_count}")
        
        return {
            "question": question,
            "answer": answer,
            "similar_contexts": similar_contexts
        }
    
    def classify_all_passages(
        self,
        dialogue_path: str,
        processed_data_path: str,
        output_path: str,
        created_data_path: Optional[str] = None,
        max_questions: Optional[int] = None,
        num_workers: int = 4
    ) -> Dict[str, Dict[str, Any]]:
        """
        대화 데이터셋에 사용된 qid에 대해서만 similar_context_id를 분류
        
        Args:
            dialogue_path: dialogue_results_train.json 경로
            processed_data_path: processed_data.json 경로
            output_path: 결과 저장 경로
            created_data_path: created_data.json 경로 (optional, 새로 생성된 질문 포함)
            max_questions: 최대 처리할 질문 수 (테스트용, None이면 모두)
            num_workers: 병렬 처리에 사용할 워커 수
            
        Returns:
            {qid: {positive_context_ids: [], similar_context_ids: []}}
        """
        # 1. 대화에서 사용된 qid 추출
        with open(dialogue_path, 'r', encoding='utf-8') as f:
            dialogue_data = json.load(f)
        
        used_qids = set()
        for conv in dialogue_data.get('conversations', []):
            if conv.get('seed_qid'):
                used_qids.add(conv['seed_qid'])
            for turn in conv.get('dialogue', []):
                if turn.get('qid'):
                    used_qids.add(turn['qid'])
        
        print(f"✅ Found {len(used_qids)} unique qids in dialogue dataset")
        
        # 2. 데이터 로드 및 병합
        with open(processed_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        qid_data = data.get('qid_data', {})
        context_mapping = data.get('context_mapping', {})
        
        # created_data 병합 (필터링 전에 병합해야 함!)
        if created_data_path:
            try:
                with open(created_data_path, 'r', encoding='utf-8') as f:
                    created = json.load(f)
                qid_data.update(created.get('qid_data', {}))
                # context_mapping 병합
                for ctx_id, ctx_info in created.get('context_mapping', {}).items():
                    if ctx_id in context_mapping:
                        context_mapping[ctx_id].update(ctx_info)
                        context_mapping[ctx_id]['q_id'] = list(set(context_mapping[ctx_id].get('q_id', []) + ctx_info.get('q_id', [])))
                    else:
                        context_mapping[ctx_id] = ctx_info
                print(f"✅ Merged created_data")
            except Exception as e:
                print(f"⚠️ Warning: Could not load created_data: {e}")
        
        # 3. 대화에서 사용된 qid만 필터링 (병합 후에 필터링!)
        target_qids = [qid for qid in used_qids if qid in qid_data]
        
        missing = used_qids - set(target_qids)
        if missing:
            print(f"⚠️ Warning: {len(missing)} qids not found in data files")
        
        if max_questions:
            target_qids = target_qids[:max_questions]
        
        print(f"✅ Processing {len(target_qids)} questions with {num_workers} workers...\n")
        
        # 4. 분류 수행 (병렬 처리)
        def process_single_qid(qid: str) -> Tuple[str, Optional[Dict[str, Any]]]:
            """단일 qid 처리 함수"""
            q_data = qid_data[qid]
            if not all([q_data.get('question'), q_data.get('answer'), q_data.get('clue_text'), q_data.get('similar_context_id')]):
                return qid, None
            
            result = self.classify_passage(
                qid=qid,
                question=q_data['question'],
                answer=q_data['answer'],
                clue=q_data['clue_text'],
                similar_context_ids=q_data['similar_context_id'],
                context_mapping=context_mapping,
                verbose=False  # 병렬 처리 시 verbose는 False로
            )
            return qid, result
        
        results = {}
        completed = 0
        skipped = 0
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # 모든 작업 제출
            future_to_qid = {executor.submit(process_single_qid, qid): qid for qid in target_qids}
            
            # 완료된 작업 처리
            for future in as_completed(future_to_qid):
                qid = future_to_qid[future]
                try:
                    result_qid, result = future.result()
                    if result is None:
                        skipped += 1
                        print(f"[{completed + skipped}/{len(target_qids)}] ⚠️ Skipping {qid}: missing required fields")
                    else:
                        results[result_qid] = result
                        completed += 1
                        if completed % 10 == 0 or completed == len(target_qids) - skipped:
                            print(f"[{completed}/{len(target_qids) - skipped}] Processed {result_qid}")
                except Exception as e:
                    skipped += 1
                    print(f"[{completed + skipped}/{len(target_qids)}] ❌ Error processing {qid}: {e}")
        
        # 5. 결과 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 통계
        stats = {'positive': 0, 'similar': 0}
        for result in results.values():
            for ctx in result.get('similar_contexts', []):
                stats[ctx.get('classification', 'similar')] += 1
        
        print(f"\n💾 Saved {len(results)} results")
        print(f"📊 Statistics: {stats['positive']} positive, {stats['similar']} similar")
        
        return results


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Passage Classification Pipeline")
    parser.add_argument("--provider", type=str, default="openai", choices=["openai", "google"])
    parser.add_argument("--model", type=str, default="gpt-5-mini")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--dialogue", type=str, required=True,
                        help="Path to dialogue_results_train.json")
    parser.add_argument("--processed-data", type=str, required=True,
                        help="Path to processed_data.json")
    parser.add_argument("--created-data", type=str, default=None,
                        help="Path to created_data.json (optional, for new questions)")
    parser.add_argument("--output", type=str, default="passage_classification_results.json",
                        help="Output JSON file path")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Maximum number of questions to process (for testing)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of parallel workers for classification")
    
    args = parser.parse_args()
    
    # QA Model 초기화
    qa_model = QAModel(provider=args.provider, model=args.model, api_key=args.api_key)
    
    # Passage Classifier 초기화
    classifier = PassageClassifier(qa_model)
    
    # 분류 실행
    results = classifier.classify_all_passages(
        dialogue_path=args.dialogue,
        processed_data_path=args.processed_data,
        output_path=args.output,
        created_data_path=args.created_data,
        max_questions=args.max_questions,
        num_workers=args.num_workers
    )
    
    print("\n🎉 Classification completed!")


if __name__ == "__main__":
    main()