"""
문서 분석 도구
문서 메타데이터 조회, 통계 정보, 표 데이터 추출
"""

from typing import Any, Dict, List, Optional
from langchain_core.tools import tool

from src.utils.logger import get_logger


@tool
def get_document_metadata(document_id: str) -> Dict[str, Any]:
    """
    문서 메타데이터 조회
    
    Args:
        document_id: 문서 ID 또는 파일명
        
    Returns:
        문서 메타데이터 딕셔너리
    """
    logger = get_logger()
    
    try:
        from src.api.main import get_rag_system
        
        rag = get_rag_system()
        
        # 문서 메타데이터 조회 (구현 필요)
        # 현재는 기본 구조만 제공
        return {
            "document_id": document_id,
            "status": "available",
            "message": "문서 메타데이터 조회 기능은 향후 구현 예정"
        }
        
    except Exception as e:
        logger.error(f"문서 메타데이터 조회 실패: {str(e)}")
        return {
            "document_id": document_id,
            "status": "error",
            "error": str(e)
        }


@tool
def get_document_statistics() -> Dict[str, Any]:
    """
    문서 통계 정보 조회
    
    Returns:
        문서 통계 딕셔너리 (총 문서 수, 총 청크 수 등)
    """
    logger = get_logger()
    
    try:
        from src.api.main import get_rag_system
        
        rag = get_rag_system()
        stats = rag.get_system_stats()
        
        return {
            "total_documents": stats.get("total_documents", 0),
            "total_chunks": stats.get("total_chunks", 0),
            "vector_store_stats": stats.get("vector_store", {})
        }
        
    except Exception as e:
        logger.error(f"문서 통계 조회 실패: {str(e)}")
        return {
            "error": str(e)
        }


@tool
def extract_table_data(document_id: str, table_title: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    표 데이터 추출
    
    Args:
        document_id: 문서 ID 또는 파일명
        table_title: 표 제목 (선택적)
        
    Returns:
        표 데이터 리스트
    """
    logger = get_logger()
    
    try:
        from src.api.main import get_rag_system
        
        rag = get_rag_system()
        
        # 표 데이터 추출 (구현 필요)
        # 현재는 기본 구조만 제공
        return [{
            "document_id": document_id,
            "table_title": table_title,
            "status": "available",
            "message": "표 데이터 추출 기능은 향후 구현 예정"
        }]
        
    except Exception as e:
        logger.error(f"표 데이터 추출 실패: {str(e)}")
        return [{
            "document_id": document_id,
            "status": "error",
            "error": str(e)
        }]


class DocumentAnalysisTool:
    """문서 분석 도구 클래스"""
    
    def __init__(self):
        self.logger = get_logger()
        self._tools = {
            "get_metadata": get_document_metadata,
            "get_statistics": get_document_statistics,
            "extract_table": extract_table_data
        }
    
    def get_metadata(self, document_id: str) -> Dict[str, Any]:
        """문서 메타데이터 조회"""
        return get_document_metadata.invoke({"document_id": document_id})
    
    def get_statistics(self) -> Dict[str, Any]:
        """문서 통계 조회"""
        return get_document_statistics.invoke({})
    
    def extract_table(self, document_id: str, table_title: Optional[str] = None) -> List[Dict[str, Any]]:
        """표 데이터 추출"""
        return extract_table_data.invoke({
            "document_id": document_id,
            "table_title": table_title
        })
    
    def get_all_tools(self) -> List:
        """모든 도구를 LangChain 도구 리스트로 반환"""
        return list(self._tools.values())

