#!/usr/bin/env python3
"""
Retriever Interface: RAG 기반 대화를 위한 검색 인터페이스
실제 구현은 나중에 추가 예정이므로 input/output만 정의
"""

from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod


class Retriever(ABC):
    """검색기 인터페이스 (input/output만 정의)"""
    
    @abstractmethod
    def retrieve(
        self,
        query: str,
        conversation_history: List[Dict[str, str]],
        gold: Optional[Dict[str, Any]] = None,
        topk: List[Dict[str, Any]] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        질문과 대화 히스토리를 기반으로 관련 passage 검색
        
        Args:
            query: 현재 사용자 질문
            conversation_history: 대화 히스토리 리스트
                각 항목은 {'role': 'user'/'assistant', 'content': str} 형식
            gold: 현재 질문의 정답이 있는 context (Optional)
                {'context_id': str, 'context': str, 'doc_id': str}
            topk: 유사한 context들의 리스트 (Optional)
                [{'context_id': str, 'context': str, 'doc_id': str}, ...]
            top_k: 반환할 passage 개수
            
        Returns:
            List of passage dictionaries:
            [
                {
                    'context_id': str,
                    'context': str,
                    'doc_id': str,
                    'score': float  # optional relevance score
                },
                ...
            ]
        """
        pass


class DummyRetriever(Retriever):
    """테스트용 더미 검색기 - gold와 topk를 반환"""
    
    def retrieve(
        self,
        query: str,
        conversation_history: List[Dict[str, str]],
        gold: Optional[Dict[str, Any]] = None,
        topk: List[Dict[str, Any]] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        gold와 topk를 반환
        
        Args:
            query: 현재 사용자 질문
            conversation_history: 대화 히스토리 리스트
            gold: 현재 질문의 정답이 있는 context
            topk: 유사한 context들의 리스트
            top_k: 반환할 passage 개수 (사용 안 함, gold와 topk만 반환)
            
        Returns:
            [gold, ...topk] 형태의 passage 리스트 (gold가 있으면 첫 번째로, 없으면 topk만)
        """
        passages = []
        
        # Gold를 첫 번째로 추가
        if gold:
            passages.append(gold)
        
        # Topk 추가
        if topk:
            passages.extend(topk)
        
        return passages

