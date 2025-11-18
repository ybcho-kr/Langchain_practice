"""
의미 기반 청킹 모듈
문장 단위 의미 유사도 기반 청킹 및 Sliding Window 오버랩
"""

import numpy as np
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from pathlib import Path
from dataclasses import dataclass
import re

from src.utils.logger import get_logger

# 순환 import 방지: 타입 힌트용으로만 import
if TYPE_CHECKING:
    from src.modules.document_processor import DocumentChunk


@dataclass
class SemanticChunk:
    """의미 기반 청크"""
    content: str
    sentences: List[str]
    start_sentence_idx: int
    end_sentence_idx: int
    coherence_score: float  # 의미 일관성 점수
    metadata: Dict[str, Any]


class SemanticChunker:
    """의미 기반 청킹 클래스"""
    
    def __init__(self,
                 embedding_manager,  # EmbeddingManager 인스턴스
                 coherence_threshold: float = 0.7,
                 max_chunk_size: int = 600,
                 min_chunk_size: int = 50,
                 overlap_sentences: int = 3):
        self.logger = get_logger()
        self.embedding_manager = embedding_manager
        self.coherence_threshold = coherence_threshold
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_sentences = overlap_sentences
    
    def chunk_by_semantic_coherence(self,
                                    sentences: List[str],
                                    metadata: Dict[str, Any] = None) -> List[SemanticChunk]:
        """
        의미 기반 청킹:
        1. 문장 단위 분할
        2. 문장 임베딩 생성 (배치)
        3. 연속 문장 간 유사도 계산
        4. 유사도가 낮은 지점에서 청크 분리
        5. 최소/최대 크기 제약 적용
        
        Args:
            sentences: 문장 리스트
            metadata: 메타데이터
            
        Returns:
            의미 기반 청크 리스트
        """
        if len(sentences) == 0:
            return []
        
        if metadata is None:
            metadata = {}
        
        try:
            # 1. 문장 임베딩 생성 (배치)
            self.logger.info(f"문장 임베딩 생성 시작: {len(sentences)}개 문장")
            
            sentence_texts = [s.strip() for s in sentences if s.strip()]
            if len(sentence_texts) == 0:
                return []
            
            # 배치 임베딩 생성
            embedding_results = self.embedding_manager.get_embeddings_batch(
                sentence_texts,
                use_cache=True
            )
            
            # 임베딩 추출
            sentence_embeddings = []
            valid_indices = []
            
            for i, result in enumerate(embedding_results):
                if result and result.embedding:
                    sentence_embeddings.append(np.array(result.embedding))
                    valid_indices.append(i)
                else:
                    self.logger.warning(f"문장 {i}의 임베딩 생성 실패")
            
            if len(sentence_embeddings) == 0:
                self.logger.error("임베딩을 생성할 수 없습니다.")
                return []
            
            sentence_embeddings = np.array(sentence_embeddings)
            
            # 2. 연속 문장 간 유사도 계산
            similarities = self._calculate_semantic_coherence(sentence_embeddings)
            
            # 3. 분할점 찾기
            split_points = self._find_split_points(
                similarities,
                sentence_texts,
                valid_indices
            )
            
            # 4. 청크 생성
            chunks = self._create_semantic_chunks(
                sentence_texts,
                valid_indices,
                split_points,
                sentence_embeddings,
                similarities,
                metadata
            )
            
            self.logger.info(f"의미 기반 청킹 완료: {len(chunks)}개 청크 생성")
            return chunks
            
        except Exception as e:
            self.logger.error(f"의미 기반 청킹 실패: {str(e)}")
            import traceback
            self.logger.error(f"상세 오류: {traceback.format_exc()}")
            return []
    
    def _calculate_semantic_coherence(self, sentence_embeddings: np.ndarray) -> np.ndarray:
        """
        연속 문장 간 코사인 유사도 계산
        
        Args:
            sentence_embeddings: 문장 임베딩 배열 (N, dim)
            
        Returns:
            유사도 배열 (N-1,)
        """
        if len(sentence_embeddings) < 2:
            return np.array([])
        
        similarities = []
        
        for i in range(len(sentence_embeddings) - 1):
            # 코사인 유사도 계산
            vec1 = sentence_embeddings[i]
            vec2 = sentence_embeddings[i + 1]
            
            # L2 정규화된 벡터이므로 내적만으로 코사인 유사도 계산 가능
            dot_product = np.dot(vec1, vec2)
            similarity = float(dot_product)  # 이미 정규화되어 있음
            similarities.append(similarity)
        
        return np.array(similarities)
    
    def _find_split_points(self,
                          similarities: np.ndarray,
                          sentences: List[str],
                          valid_indices: List[int]) -> List[int]:
        """
        의미 일관성이 낮은 지점(분할점) 찾기
        
        Args:
            similarities: 연속 문장 간 유사도 배열
            sentences: 문장 리스트
            valid_indices: 유효한 문장 인덱스
            
        Returns:
            분할점 인덱스 리스트
        """
        split_points = []
        
        if len(similarities) == 0:
            return split_points
        
        # 유사도가 임계값 이하인 지점 찾기
        for i, sim in enumerate(similarities):
            if sim < self.coherence_threshold:
                # 최소 청크 크기 확인 (문자 수 기준)
                start_idx = 0 if len(split_points) == 0 else split_points[-1]
                current_chunk_size = sum(len(sentences[j]) for j in range(start_idx, i + 1))
                
                if current_chunk_size >= self.min_chunk_size:
                    split_points.append(i + 1)
        
        return split_points
    
    def _create_semantic_chunks(self,
                               sentences: List[str],
                               valid_indices: List[int],
                               split_points: List[int],
                               sentence_embeddings: np.ndarray,
                               similarities: np.ndarray,
                               metadata: Dict[str, Any]) -> List[SemanticChunk]:
        """
        의미 기반 청크 생성
        
        Args:
            sentences: 문장 리스트
            valid_indices: 유효한 문장 인덱스
            split_points: 분할점 인덱스
            sentence_embeddings: 문장 임베딩
            similarities: 유사도 배열
            metadata: 메타데이터
            
        Returns:
            의미 기반 청크 리스트
        """
        chunks = []
        
        if len(sentences) == 0:
            return chunks
        
        # 분할점이 없으면 전체를 하나의 청크로
        if len(split_points) == 0:
            content = ' '.join(sentences)
            avg_coherence = float(np.mean(similarities)) if len(similarities) > 0 else 1.0
            
            chunk = SemanticChunk(
                content=content,
                sentences=sentences,
                start_sentence_idx=0,
                end_sentence_idx=len(sentences) - 1,
                coherence_score=avg_coherence,
                metadata=metadata
            )
            chunks.append(chunk)
            return chunks
        
        # 분할점 기반으로 청크 생성
        start_idx = 0
        
        for split_idx in split_points:
            if split_idx > start_idx:
                chunk_sentences = sentences[start_idx:split_idx]
                chunk_content = ' '.join(chunk_sentences)
                
                # 청크 크기 확인
                chunk_size = len(chunk_content)
                
                if chunk_size > self.max_chunk_size:
                    # 최대 크기 초과 시 추가 분할
                    sub_chunks = self._split_large_chunk(
                        chunk_sentences,
                        start_idx,
                        sentence_embeddings,
                        similarities,
                        metadata
                    )
                    chunks.extend(sub_chunks)
                elif chunk_size >= self.min_chunk_size:
                    # 유효한 청크
                    chunk_similarities = similarities[start_idx:split_idx-1] if split_idx > start_idx + 1 else []
                    avg_coherence = float(np.mean(chunk_similarities)) if len(chunk_similarities) > 0 else 1.0
                    
                    chunk = SemanticChunk(
                        content=chunk_content,
                        sentences=chunk_sentences,
                        start_sentence_idx=start_idx,
                        end_sentence_idx=split_idx - 1,
                        coherence_score=avg_coherence,
                        metadata=metadata
                    )
                    chunks.append(chunk)
            
            start_idx = split_idx
        
        # 마지막 청크 처리
        if start_idx < len(sentences):
            chunk_sentences = sentences[start_idx:]
            chunk_content = ' '.join(chunk_sentences)
            
            if len(chunk_content) >= self.min_chunk_size:
                chunk_similarities = similarities[start_idx:] if start_idx < len(similarities) else []
                avg_coherence = float(np.mean(chunk_similarities)) if len(chunk_similarities) > 0 else 1.0
                
                chunk = SemanticChunk(
                    content=chunk_content,
                    sentences=chunk_sentences,
                    start_sentence_idx=start_idx,
                    end_sentence_idx=len(sentences) - 1,
                    coherence_score=avg_coherence,
                    metadata=metadata
                )
                chunks.append(chunk)
        
        return chunks
    
    def _split_large_chunk(self,
                          sentences: List[str],
                          start_idx: int,
                          sentence_embeddings: np.ndarray,
                          similarities: np.ndarray,
                          metadata: Dict[str, Any]) -> List[SemanticChunk]:
        """
        큰 청크를 추가로 분할
        
        Args:
            sentences: 문장 리스트
            start_idx: 시작 인덱스
            sentence_embeddings: 문장 임베딩
            similarities: 유사도 배열
            metadata: 메타데이터
            
        Returns:
            분할된 청크 리스트
        """
        chunks = []
        current_chunk = []
        current_size = 0
        
        for i, sentence in enumerate(sentences):
            sentence_size = len(sentence)
            
            if current_size + sentence_size > self.max_chunk_size and len(current_chunk) > 0:
                # 현재 청크 완성
                chunk_content = ' '.join(current_chunk)
                
                # 유사도 계산 (가능한 범위에서)
                chunk_start_idx = start_idx + i - len(current_chunk)
                chunk_end_idx = start_idx + i - 1
                
                if chunk_start_idx < len(similarities) and chunk_end_idx > chunk_start_idx:
                    chunk_similarities = similarities[chunk_start_idx:chunk_end_idx]
                    avg_coherence = float(np.mean(chunk_similarities)) if len(chunk_similarities) > 0 else 1.0
                else:
                    avg_coherence = 1.0
                
                chunk = SemanticChunk(
                    content=chunk_content,
                    sentences=current_chunk.copy(),
                    start_sentence_idx=chunk_start_idx,
                    end_sentence_idx=chunk_end_idx,
                    coherence_score=avg_coherence,
                    metadata=metadata
                )
                chunks.append(chunk)
                
                # 새 청크 시작
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        # 마지막 청크 처리
        if current_chunk:
            chunk_content = ' '.join(current_chunk)
            chunk = SemanticChunk(
                content=chunk_content,
                sentences=current_chunk,
                start_sentence_idx=start_idx + len(sentences) - len(current_chunk),
                end_sentence_idx=start_idx + len(sentences) - 1,
                coherence_score=1.0,
                metadata=metadata
            )
            chunks.append(chunk)
        
        return chunks
    
    def create_overlapping_chunks(self,
                                 chunks: List[SemanticChunk],
                                 overlap_sentences: Optional[int] = None) -> List[str]:
        """
        Sliding Window 오버랩 생성 (문장 단위)
        
        Args:
            chunks: 의미 기반 청크 리스트
            overlap_sentences: 오버랩 문장 수 (None이면 기본값 사용)
            
        Returns:
            오버랩이 적용된 청크 내용 리스트
        """
        if overlap_sentences is None:
            overlap_sentences = self.overlap_sentences
        
        if len(chunks) == 0:
            return []
        
        overlapping_contents = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                # 첫 번째 청크는 오버랩 없음
                overlapping_contents.append(chunk.content)
            else:
                # 이전 청크의 마지막 N개 문장 추출
                prev_chunk = chunks[i - 1]
                prev_sentences = prev_chunk.sentences
                
                overlap_text = ' '.join(prev_sentences[-overlap_sentences:])
                
                # 현재 청크 시작에 오버랩 추가
                current_content = ' '.join(chunk.sentences)
                new_content = ' '.join([overlap_text, current_content])
                overlapping_contents.append(new_content)
        
        return overlapping_contents
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        한국어 문장부호를 고려한 문장 분할
        
        Args:
            text: 텍스트
            
        Returns:
            문장 리스트
        """
        # 한국어 문장부호 패턴
        sentence_endings = r'[.!?。！？]|\.{2,}|…'
        
        # 문장부호로 분할하되, 문장부호는 유지
        sentences = re.split(f'({sentence_endings})', text)
        
        # 문장부호와 문장을 다시 결합
        result = []
        for i in range(0, len(sentences), 2):
            if i + 1 < len(sentences):
                sentence = sentences[i].strip() + sentences[i + 1]
                if sentence.strip():
                    result.append(sentence.strip())
            elif sentences[i].strip():
                result.append(sentences[i].strip())
        
        return result

