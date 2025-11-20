"""
임베딩 모듈
Ollama 서버 또는 HuggingFace 로컬 모델을 통한 임베딩 생성 및 관리
langchain-ollama를 사용하여 Ollama 임베딩 표준화
"""

import numpy as np
from typing import List, Union, Optional, Dict, Any
from dataclasses import dataclass
import json
import time
import torch
from sentence_transformers import SentenceTransformer
from src.utils.logger import get_logger, log_info, log_error
from src.utils.config import get_embedding_config
from src.utils.langchain_utils import create_ollama_embeddings


@dataclass
class EmbeddingResult:
    """임베딩 결과"""
    text: str
    embedding: List[float]
    model: str
    dimension: int
    processing_time: float


class HuggingFaceEmbeddingClient:
    """HuggingFace 로컬 임베딩 클라이언트"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = get_logger()
        
        if config is None:
            config = get_embedding_config()
        
        self.model_name = config.name
        self.model_path = config.model_path
        self.dimension = config.dimension
        self.max_length = config.max_length
        self.batch_size = config.batch_size
        self.device = config.device if hasattr(config, 'device') else 'cuda'
        
        # GPU 사용 가능 여부 확인
        if self.device == 'cuda' and not torch.cuda.is_available():
            self.logger.warning("CUDA를 사용할 수 없습니다. CPU로 전환합니다.")
            self.device = 'cpu'
        
        # SentenceTransformer 모델 로드
        try:
            self.logger.info(f"HuggingFace 모델 로딩 중: {self.model_path}")
            self.model = SentenceTransformer(self.model_path, device=self.device)
            self.logger.info(f"HuggingFace 임베딩 클라이언트 초기화 완료: {self.model_name} (디바이스: {self.device})")
        except Exception as e:
            self.logger.error(f"HuggingFace 모델 로딩 실패: {str(e)}")
            raise
    
    def _check_server_status(self) -> bool:
        """로컬 모델은 항상 사용 가능"""
        return True
    
    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """단일 텍스트 임베딩 생성"""
        try:
            start_time = time.time()
            
            # 텍스트 길이 제한
            if len(text) > self.max_length:
                text = text[:self.max_length]
            
            # 임베딩 생성
            embedding = self.model.encode(
                text,
                convert_to_tensor=False,
                show_progress_bar=False,
                normalize_embeddings=True  # 정규화
            )
            
            processing_time = time.time() - start_time
            
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            
            self.logger.debug(f"임베딩 생성 완료: {len(embedding)}차원, {processing_time:.2f}초")
            return embedding
            
        except Exception as e:
            self.logger.error(f"임베딩 생성 중 오류: {str(e)}")
            return None
    
    def generate_embedding(self, text: str) -> Optional[EmbeddingResult]:
        """단일 텍스트 임베딩 생성 (결과 객체 반환)"""
        # 텍스트 길이 제한
        if len(text) > self.max_length:
            text = text[:self.max_length]
            # DEBUG 레벨로 변경 (과도한 경고 방지)
            self.logger.debug(f"텍스트가 {self.max_length}자를 초과하여 잘렸습니다")
        
        start_time = time.time()
        embedding = self._generate_embedding(text)
        processing_time = time.time() - start_time
        
        if embedding is None:
            return None
        
        return EmbeddingResult(
            text=text,
            embedding=embedding,
            model=self.model_name,
            dimension=len(embedding),
            processing_time=processing_time
        )
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[Optional[EmbeddingResult]]:
        """배치 임베딩 생성"""
        results = []
        
        # 텍스트 길이 제한
        processed_texts = []
        for text in texts:
            if len(text) > self.max_length:
                processed_texts.append(text[:self.max_length])
            else:
                processed_texts.append(text)
        
        try:
            # 배치 처리
            for i in range(0, len(processed_texts), self.batch_size):
                batch = processed_texts[i:i + self.batch_size]
                self.logger.info(f"배치 처리 중: {i+1}-{min(i+self.batch_size, len(processed_texts))}/{len(processed_texts)}")
                
                start_time = time.time()
                
                # 배치 임베딩 생성
                embeddings = self.model.encode(
                    batch,
                    batch_size=len(batch),
                    convert_to_tensor=False,
                    show_progress_bar=True,
                    normalize_embeddings=True
                )
                
                processing_time = time.time() - start_time
                
                # 결과 변환
                for j, (text, embedding) in enumerate(zip(batch, embeddings)):
                    if isinstance(embedding, np.ndarray):
                        embedding = embedding.tolist()
                    
                    results.append(EmbeddingResult(
                        text=text,
                        embedding=embedding,
                        model=self.model_name,
                        dimension=len(embedding),
                        processing_time=processing_time / len(batch)
                    ))
            
            successful_count = sum(1 for r in results if r is not None)
            self.logger.info(f"배치 임베딩 생성 완료: {successful_count}/{len(texts)} 성공")
            
            return results
            
        except Exception as e:
            self.logger.error(f"배치 임베딩 생성 실패: {str(e)}")
            return [None] * len(texts)


class OllamaEmbeddingClient:
    """Ollama 임베딩 클라이언트 (langchain-ollama 사용)"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = get_logger()
        
        if config is None:
            config = get_embedding_config()
        
        # 딕셔너리와 ModelConfig 객체 모두 처리
        if isinstance(config, dict):
            self.model_name = config.get('name', 'bge-m3-korean')
            self.base_url = config.get('base_url', 'http://localhost:11434')
            self.dimension = config.get('dimension', 1024)
            self.max_length = config.get('max_length', 512)
            self.batch_size = config.get('batch_size', 32)
        else:
            # ModelConfig 객체인 경우
            self.model_name = config.name
            self.base_url = config.base_url
            self.dimension = config.dimension
            self.max_length = config.max_length
            self.batch_size = config.batch_size
        
        # langchain-ollama OllamaEmbeddings 초기화 (유틸리티 함수 사용)
        self.embeddings = create_ollama_embeddings(
            model_name=self.model_name,
            base_url=self.base_url,
        )
        
        if not self.embeddings:
            raise RuntimeError(f"OllamaEmbeddings 인스턴스 생성 실패: model={self.model_name}, base_url={self.base_url}")
        
        self.logger.info(
            f"Ollama 임베딩 클라이언트 초기화 (langchain-ollama): {self.model_name}, "
            f"base_url={self.base_url}, dimension={self.dimension}"
        )
    
    def _check_server_status(self) -> bool:
        """Ollama 서버 상태 확인"""
        # langchain-ollama는 내부적으로 연결 상태를 확인하므로 간단히 체크
        # 실제 사용 시점에 에러가 발생하면 처리됨
        return True
    
    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """단일 텍스트 임베딩 생성 (langchain-ollama 사용)"""
        try:
            # langchain-ollama를 사용하여 임베딩 생성
            embedding = self.embeddings.embed_query(text)
            
            if embedding:
                self.logger.debug(f"임베딩 생성 완료: {len(embedding)}차원")
                return embedding
            else:
                self.logger.error("임베딩 결과가 비어있습니다")
                return None
                
        except Exception as e:
            self.logger.error(f"임베딩 생성 중 오류: {str(e)}")
            return None
    
    def generate_embedding(self, text: str) -> Optional[EmbeddingResult]:
        """단일 텍스트 임베딩 생성 (결과 객체 반환, langchain-ollama 사용)"""
        if not self._check_server_status():
            self.logger.error("Ollama 서버가 실행되지 않았습니다")
            return None
        
        # 텍스트 길이 제한
        original_text = text
        if len(text) > self.max_length:
            text = text[:self.max_length]
            self.logger.debug(f"텍스트가 {self.max_length}자를 초과하여 잘렸습니다")
        
        start_time = time.time()
        embedding = self._generate_embedding(text)
        processing_time = time.time() - start_time
        
        if embedding is None:
            return None
        
        return EmbeddingResult(
            text=original_text,  # 원본 텍스트 반환
            embedding=embedding,
            model=self.model_name,
            dimension=len(embedding),
            processing_time=processing_time
        )
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[Optional[EmbeddingResult]]:
        """배치 임베딩 생성 (langchain-ollama 사용)"""
        if not self._check_server_status():
            self.logger.error("Ollama 서버가 실행되지 않았습니다")
            return [None] * len(texts)
        
        try:
            start_time = time.time()
            
            # langchain-ollama를 사용하여 배치 임베딩 생성
            embeddings_list = self.embeddings.embed_documents(texts)
            processing_time = time.time() - start_time
            
            results = []
            for i, (text, embedding) in enumerate(zip(texts, embeddings_list)):
                if embedding:
                    results.append(EmbeddingResult(
                        text=text,
                        embedding=embedding,
                        model=self.model_name,
                        dimension=len(embedding),
                        processing_time=processing_time / len(texts)  # 평균 처리 시간
                    ))
                else:
                    results.append(None)
            
            success_count = sum(1 for r in results if r is not None)
            self.logger.info(
                f"배치 임베딩 생성 완료 (langchain-ollama): {success_count}/{len(texts)} 성공, "
                f"{processing_time:.2f}초"
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"배치 임베딩 생성 중 오류: {str(e)}")
            return [None] * len(texts)


class EmbeddingManager:
    """임베딩 관리자"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = get_logger()
        
        if config is None:
            config = get_embedding_config()
        
        # provider에 따라 적절한 클라이언트 선택
        if config.provider == "huggingface":
            self.client = HuggingFaceEmbeddingClient(config)
        else:
            self.client = OllamaEmbeddingClient(config)
        
        self.embedding_cache: Dict[str, EmbeddingResult] = {}
    
    def get_embedding(self, text: str, use_cache: bool = True) -> Optional[EmbeddingResult]:
        """임베딩 가져오기 (캐시 사용)"""
        if use_cache and text in self.embedding_cache:
            self.logger.debug(f"캐시에서 임베딩 반환: {text[:50]}...")
            return self.embedding_cache[text]
        
        result = self.client.generate_embedding(text)
        
        if result and use_cache:
            self.embedding_cache[text] = result
        
        return result
    
    def get_embeddings_batch(self, texts: List[str], use_cache: bool = True) -> List[Optional[EmbeddingResult]]:
        """배치 임베딩 가져오기 (캐시 최적화)"""
        if not texts:
            return []
        
        # 캐시에서 먼저 확인
        cache_miss_texts = []
        cache_results = {}
        
        for text in texts:
            if use_cache and text in self.embedding_cache:
                cache_results[text] = self.embedding_cache[text]
            else:
                cache_miss_texts.append(text)
        
        # 캐시에서 찾은 것들 반환
        if not cache_miss_texts:
            self.logger.debug(f"모든 임베딩이 캐시에서 반환됨: {len(texts)}개")
            return [cache_results[text] for text in texts]
        
        # 캐시 미스된 텍스트들은 클라이언트의 배치 처리 사용
        from src.modules.embedding_module import HuggingFaceEmbeddingClient
        if isinstance(self.client, HuggingFaceEmbeddingClient):
            # HuggingFace는 배치 처리가 더 효율적
            batch_results = self.client.generate_embeddings_batch(cache_miss_texts)
            
            # 결과 캐싱
            for i, result in enumerate(batch_results):
                if result and use_cache:
                    self.embedding_cache[cache_miss_texts[i]] = result
        else:
            # Ollama 등은 개별 처리
            batch_results = []
            for text in cache_miss_texts:
                result = self.client.generate_embedding(text)
                if result and use_cache:
                    self.embedding_cache[text] = result
                batch_results.append(result)
        
        # 최종 결과 조합 (순서 유지)
        final_results = []
        cache_idx = 0
        batch_idx = 0
        
        for text in texts:
            if text in cache_results:
                final_results.append(cache_results[text])
            else:
                final_results.append(batch_results[batch_idx])
                batch_idx += 1
        
        self.logger.debug(f"배치 임베딩 완료: 캐시 {len(cache_results)}개, 신규 {len(cache_miss_texts)}개")
        return final_results
    
    def clear_cache(self):
        """캐시 초기화"""
        self.embedding_cache.clear()
        self.logger.info("임베딩 캐시가 초기화되었습니다")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계 반환"""
        return {
            'cache_size': len(self.embedding_cache),
            'model_name': self.client.model_name,
            'dimension': self.client.dimension
        }
    
    # ========== 비동기 메서드 (Phase 3: 임베딩 생성 비동기화) ==========
    
    async def get_embedding_async(self, text: str, use_cache: bool = True) -> Optional[EmbeddingResult]:
        """비동기 임베딩 가져오기 (캐시 사용)"""
        if use_cache and text in self.embedding_cache:
            self.logger.debug(f"캐시에서 임베딩 반환: {text[:50]}...")
            return self.embedding_cache[text]
        
        # CPU/GPU 연산이므로 asyncio.to_thread 사용
        import asyncio
        result = await asyncio.to_thread(self.client.generate_embedding, text)
        
        if result and use_cache:
            self.embedding_cache[text] = result
        
        return result
    
    async def get_embeddings_batch_async(self, texts: List[str], use_cache: bool = True) -> List[Optional[EmbeddingResult]]:
        """비동기 배치 임베딩 가져오기 (캐시 최적화)"""
        if not texts:
            return []
        
        # 캐시에서 먼저 확인
        cache_miss_texts = []
        cache_results = {}
        
        for text in texts:
            if use_cache and text in self.embedding_cache:
                cache_results[text] = self.embedding_cache[text]
            else:
                cache_miss_texts.append(text)
        
        # 캐시에서 찾은 것들 반환
        if not cache_miss_texts:
            self.logger.debug(f"모든 임베딩이 캐시에서 반환됨: {len(texts)}개")
            return [cache_results[text] for text in texts]
        
        # 캐시 미스된 텍스트들은 클라이언트의 배치 처리 사용 (비동기)
        import asyncio
        from src.modules.embedding_module import HuggingFaceEmbeddingClient
        
        if isinstance(self.client, HuggingFaceEmbeddingClient):
            # HuggingFace는 배치 처리가 더 효율적
            batch_results = await asyncio.to_thread(
                self.client.generate_embeddings_batch,
                cache_miss_texts
            )
        else:
            # Ollama 등은 개별 처리 (비동기)
            batch_results = []
            for text in cache_miss_texts:
                result = await asyncio.to_thread(self.client.generate_embedding, text)
                if result and use_cache:
                    self.embedding_cache[text] = result
                batch_results.append(result)
        
        # 결과 캐싱 (HuggingFace의 경우)
        if isinstance(self.client, HuggingFaceEmbeddingClient):
            for i, result in enumerate(batch_results):
                if result and use_cache:
                    self.embedding_cache[cache_miss_texts[i]] = result
        
        # 최종 결과 조합 (순서 유지)
        final_results = []
        batch_idx = 0
        
        for text in texts:
            if text in cache_results:
                final_results.append(cache_results[text])
            else:
                final_results.append(batch_results[batch_idx])
                batch_idx += 1
        
        self.logger.debug(f"비동기 배치 임베딩 완료: 캐시 {len(cache_results)}개, 신규 {len(cache_miss_texts)}개")
        return final_results


def create_embedding_manager(config: Optional[Dict[str, Any]] = None) -> EmbeddingManager:
    """임베딩 관리자 생성"""
    return EmbeddingManager(config)


def generate_embeddings_for_texts(texts: List[str], config: Optional[Dict[str, Any]] = None) -> List[Optional[EmbeddingResult]]:
    """텍스트 리스트에 대한 임베딩 생성"""
    manager = create_embedding_manager(config)
    return manager.get_embeddings_batch(texts)


def generate_embedding_for_text(text: str, config: Optional[Dict[str, Any]] = None) -> Optional[EmbeddingResult]:
    """단일 텍스트에 대한 임베딩 생성"""
    manager = create_embedding_manager(config)
    return manager.get_embedding(text)
