"""
LangChain Embeddings 래퍼
기존 EmbeddingManager를 LangChain Embeddings 인터페이스로 변환
"""

from typing import List

from langchain_core.embeddings import Embeddings

from src.modules.embedding_module import EmbeddingManager


class EmbeddingManagerWrapper(Embeddings):
    """EmbeddingManager를 LangChain Embeddings로 래핑"""
    
    def __init__(self, embedding_manager: EmbeddingManager):
        """
        Args:
            embedding_manager: 기존 EmbeddingManager 인스턴스
        """
        super().__init__()
        self.embedding_manager = embedding_manager
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        여러 문서에 대한 임베딩 생성
        
        Args:
            texts: 문서 텍스트 리스트
            
        Returns:
            임베딩 리스트
        """
        results = self.embedding_manager.get_embeddings_batch(texts)
        return [result.embedding for result in results if result and result.embedding]
    
    def embed_query(self, text: str) -> List[float]:
        """
        단일 쿼리에 대한 임베딩 생성
        
        Args:
            text: 쿼리 텍스트
            
        Returns:
            임베딩 벡터
        """
        result = self.embedding_manager.get_embedding(text)
        if result and result.embedding:
            return result.embedding
        return []
    
    # ========== 비동기 메서드 (Phase 3: 임베딩 생성 비동기화) ==========
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        여러 문서에 대한 임베딩 생성 (비동기)
        
        Args:
            texts: 문서 텍스트 리스트
            
        Returns:
            임베딩 리스트
        """
        results = await self.embedding_manager.get_embeddings_batch_async(texts)
        return [result.embedding for result in results if result and result.embedding]
    
    async def aembed_query(self, text: str) -> List[float]:
        """
        단일 쿼리에 대한 임베딩 생성 (비동기)
        
        Args:
            text: 쿼리 텍스트
            
        Returns:
            임베딩 벡터
        """
        result = await self.embedding_manager.get_embedding_async(text)
        if result and result.embedding:
            return result.embedding
        return []

