"""
질문 정규화 및 정제 모듈

기능:
- kiwipiepy를 사용한 질문 정규화, 형태소 분석, 엔티티 추출
- LLM을 사용한 오타 수정 및 띄어쓰기 교정
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from src.utils.logger import get_logger
from src.utils.config import get_config
from src.utils.langchain_utils import create_chat_ollama_from_config, create_simple_message

try:
    from kiwipiepy import Kiwi
    KIWIPIEPY_AVAILABLE = True
except ImportError:
    KIWIPIEPY_AVAILABLE = False
    Kiwi = None


@dataclass
class RefinedQuery:
    \"\"\"정제된 쿼리 결과\"\"\"
    original: str
    normalized: str
    keywords: List[str]
    entities: List[str]
    structured_query: str = ""  # 기본값 추가


class QueryRefiner:
    \"\"\"질문 정제 클래스\"\"\"
    
    def __init__(self):
        self.logger = get_logger()
        self.config = get_config()
        self.kiwi = None
        
        # kiwipiepy 초기화
        if KIWIPIEPY_AVAILABLE:
            try:
                self.kiwi = Kiwi()
                self.logger.info("kiwipiepy 초기화 완료")
            except Exception as e:
                self.logger.warning(f"kiwipiepy 초기화 실패: {str(e)}")
        else:
            self.logger.warning("kiwipiepy가 설치되지 않았습니다. 키워드 추출 기능이 제한됩니다.")
    
    def normalize_basic(self, query: str) -> str:
        \"\"\"기본 정규화 (공백, 특수문자 처리)\"\"\"
        # 공백 정리
        query = re.sub(r'\s+', ' ', query).strip()
        return query
    
    def extract_keywords(self, query: str) -> List[str]:
        \"\"\"키워드 추출 (kiwipiepy 사용)\"\"\"
        if not self.kiwi:
            return []
        
        try:
            # 형태소 분석
            tokens = self.kiwi.analyze(query)
            keywords = []
            
            for token in tokens:
                word = token[0]
                pos = token[1]
                
                # 명사, 동사, 형용사만 추출
                if pos.startswith('N') or pos.startswith('V') or pos.startswith('A'):
                    if len(word) > 1:  # 1글자 제외
                        keywords.append(word)
            
            return list(set(keywords))  # 중복 제거
        except Exception as e:
            self.logger.warning(f"키워드 추출 실패: {str(e)}")
            return []
    
    def extract_entities(self, query: str) -> List[str]:
        \"\"\"엔티티 추출 (간단한 패턴 매칭)\"\"\"
        entities = []
        
        # 전기설비 관련 용어 패턴
        patterns = [
            r'변압기',
            r'전선',
            r'케이블',
            r'절연',
            r'접지',
            r'전압',
            r'전류',
            r'저항',
            r'콘덴서',
            r'인덕터',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, query)
            entities.extend(matches)
        
        return list(set(entities))  # 중복 제거
    
    async def refine_with_llm(self, query: str, use_typo_correction: bool = True) -> str:
        \"\"\"LLM을 사용한 질문 정제 (오타 수정, 띄어쓰기)\"\"\"
        if not use_typo_correction:
            return query
        
        try:
            # gemma3:1b 모델 사용 (빠른 성능)
            llm = create_chat_ollama_from_config(
                model_name='gemma3:1b',
                base_url='http://localhost:11434',
                temperature=0.1,
                max_tokens=200
            )
            
            # 구조화된 프롬프트
            system_prompt = ""
            
            user_prompt = f"다음 질문을 정제하세요:\n{query}"
            
            messages = [
                create_simple_message('system', system_prompt),
                create_simple_message('user', user_prompt)
            ]
            
            response = await llm.ainvoke(messages)
            
            # 응답 파싱
            refined = response.content.strip()
            
            # 응답이 비어있거나 원본과 같으면 원본 반환
            if not refined or refined == query:
                return query
            
            # 응답에서 따옴표 제거
            refined = refined.strip('"\'')
            
            self.logger.info(f"질문 정제: '{query}' -> '{refined}'")
            return refined
            
        except Exception as e:
            self.logger.warning(f"LLM 질문 정제 실패: {str(e)}, 원본 반환")
            return query
    
    async def refine(self, query: str, use_typo_correction: bool = True) -> RefinedQuery:
        \"\"\"질문 정제 메인 메서드\"\"\"
        # 기본 정규화
        normalized = self.normalize_basic(query)
        
        # LLM 정제 (오타 수정, 띄어쓰기)
        if use_typo_correction:
            normalized = await self.refine_with_llm(normalized, use_typo_correction)
        
        # 키워드 추출
        keywords = self.extract_keywords(normalized)
        
        # 엔티티 추출
        entities = self.extract_entities(normalized)
        
        # 구조화된 쿼리 생성
        structured_query = normalized
        if keywords:
            structured_query += f" [키워드: {', '.join(keywords)}]"
        if entities:
            structured_query += f" [엔티티: {', '.join(entities)}]"
        
        return RefinedQuery(
            original=query,
            normalized=normalized,
            keywords=keywords,
            entities=entities,
            structured_query=structured_query
        )


# 싱글톤 인스턴스
_query_refiner = None


def get_query_refiner() -> QueryRefiner:
    \"\"\"QueryRefiner 싱글톤 인스턴스 반환\"\"\"
    global _query_refiner
    if _query_refiner is None:
        _query_refiner = QueryRefiner()
    return _query_refiner
"""
