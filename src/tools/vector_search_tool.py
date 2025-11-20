"""
벡터 검색 도구
RAGSystem의 검색 기능을 LangChain 도구로 래핑
"""

from typing import Any, Dict, List, Optional
from langchain_core.tools import tool

from src.utils.logger import get_logger


@tool
def vector_search_tool(
    query: str,
    max_sources: int = 5,
    score_threshold: float = 0.7
) -> List[Dict[str, Any]]:
    """
    벡터 검색 도구
    
    RAG 시스템을 사용하여 관련 문서를 검색합니다.
    
    Args:
        query: 검색 쿼리
        max_sources: 최대 소스 수 (기본값: 5)
        score_threshold: 점수 임계값 (기본값: 0.7)
        
    Returns:
        검색 결과 리스트 (각 결과는 content, score, source_file 등을 포함)
    """
    logger = get_logger()
    
    try:
        from src.api.main import get_rag_system
        import asyncio
        
        rag = get_rag_system()
        
        # 비동기 메서드를 동기적으로 호출
        response = asyncio.run(
            rag.query_async(
                question=query,
                max_sources=max_sources,
                score_threshold=score_threshold
            )
        )
        
        return response.sources
        
    except Exception as e:
        logger.error(f"벡터 검색 도구 실행 실패: {str(e)}")
        return []


class VectorSearchTool:
    """벡터 검색 도구 클래스"""
    
    def __init__(self):
        self.logger = get_logger()
        self._tool = vector_search_tool
    
    def execute(
        self,
        query: str,
        max_sources: int = 5,
        score_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        벡터 검색 실행
        
        Args:
            query: 검색 쿼리
            max_sources: 최대 소스 수
            score_threshold: 점수 임계값
            
        Returns:
            검색 결과 리스트
        """
        return vector_search_tool.invoke({
            "query": query,
            "max_sources": max_sources,
            "score_threshold": score_threshold
        })
    
    def to_langchain_tool(self):
        """LangChain 도구로 변환"""
        return self._tool

