#!/usr/bin/env python3
"""
QA Model: 질문-답변 생성을 위한 LLM 모델 클래스
"""

import os
import re
from typing import Optional
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()


def detect_provider(model: str) -> str:
    """모델 이름으로 provider 자동 감지"""
    model_lower = model.lower()
    if model_lower.startswith("gpt-") or model_lower.startswith("o1-") or model_lower.startswith("o3-"):
        return "openai"
    elif "gemini" in model_lower:
        return "google"
    else:
        return "openai"  # 기본값


class QAModel:
    """질문-답변 생성을 위한 LLM 모델"""
    
    def __init__(self, provider: Optional[str] = None, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        """
        Args:
            provider: "openai" or "google" (None이면 model로 자동 감지)
            model: 모델 이름
            api_key: API 키 (없으면 환경변수에서 가져옴)
        """
        self.model = model
        self.provider = provider if provider else detect_provider(model)
        self.client = None
        self._init_client(api_key)
    
    def _init_client(self, api_key: Optional[str]):
        """API 클라이언트 초기화"""
        if self.provider == "openai":
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
            except ImportError:
                raise ImportError("openai 패키지가 필요합니다: pip install openai")
        elif self.provider == "google":
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key or os.getenv("GEMINI_API_KEY"))
                self.client = genai.GenerativeModel(self.model)
            except ImportError:
                raise ImportError("google-generativeai 패키지가 필요합니다: pip install google-generativeai")
        else:
            raise ValueError(f"지원하지 않는 provider: {self.provider}")
    
    def call_llm(self, messages: list, temperature: float = 0.7) -> str:
        """
        일반적인 LLM 호출 메서드
        
        Args:
            messages: 메시지 리스트 (OpenAI 형식)
            temperature: 온도 파라미터
            
        Returns:
            LLM 응답 텍스트
        """
        if self.provider == "openai":
            # reasoning 모델(gpt-5 계열, o 시리즈)은 temperature를 지원하지 않음
            model_lower = self.model.lower()
            if "gpt-5" in model_lower or re.match(r"o\d", model_lower):
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature
                )
            response_text = response.choices[0].message.content.strip()
        elif self.provider == "google":
            # Google API 형식 변환
            prompt_parts = []
            for msg in messages:
                if msg['role'] == 'system':
                    prompt_parts.append(f"System: {msg['content']}")
                elif msg['role'] == 'user':
                    prompt_parts.append(f"User: {msg['content']}")
                elif msg['role'] == 'assistant':
                    prompt_parts.append(f"Assistant: {msg['content']}")
            
            prompt_text = "\n\n".join(prompt_parts)
            response = self.client.generate_content(prompt_text, generation_config={"temperature": temperature})
            response_text = response.text.strip()
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
        
        return response_text
    
    def generate_answer(self, context: str, question: str) -> str:
        """
        Context와 Question으로부터 답변 생성
        
        Args:
            context: 배경 문맥 또는 지문
            question: 질문
            
        Returns:
            생성된 답변 텍스트
        """
        prompt = f"""다음 문맥을 바탕으로 질문에 답하세요.

문맥:
{context}

질문: {question}

답변:"""
        
        messages = [{"role": "user", "content": prompt}]
        return self.call_llm(messages, temperature=0.7)
    
    def classify(self, prompt: str) -> str:
        """
        분류 작업을 위한 LLM 호출
        
        Args:
            prompt: 분류를 위한 프롬프트
            
        Returns:
            LLM 응답 텍스트
        """
        messages = [{"role": "user", "content": prompt}]
        return self.call_llm(messages, temperature=0.3)

