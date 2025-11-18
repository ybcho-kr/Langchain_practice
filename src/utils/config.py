"""
설정 관리 모듈
1단계 개발용 설정 파일 로딩 및 관리
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """모델 설정"""
    provider: str = "ollama"
    name: str
    base_url: str = "http://localhost:4"  # 설정 파일의 기본값과 일치
    model_path: Optional[str] = None
    dimension: Optional[int] = None
    max_length: int = 512  # 설정 파일의 기본값과 일치
    batch_size: int = 256  # 설정 파일의 기본값과 일치
    device: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    timeout: int = 300  # Ollama API 타임아웃 (초)


class QdrantConfig(BaseModel):
    """Qdrant 설정"""
    host: str = "localhost"  # 설정 파일의 기본값과 일치
    port: int = 6334  # 설정 파일의 기본값과 일치
    collection_name: str = "electrical_diagnosis"  # 설정 파일의 기본값과 일치
    vector_size: int = 1024  # 설정 파일의 기본값과 일치
    distance_metric: str = "cosine"  # 설정 파일의 기본값과 일치
    storage_path: str = "data/qdrant_storage"  # 설정 파일의 기본값과 일치
    use_local_storage: bool = True  # 설정 파일의 기본값과 일치
    default_limit: int = 5  # 검색 기본 결과 수
    max_scroll_limit: int = 10000  # 최대 스크롤 제한
    connection_timeout: int = 5  # 연결 타임아웃 (초)
    request_timeout: int = 60  # 요청 타임아웃 (초)
    # BM25 설정
    bm25_storage_path: str = "data/bm25_index"  # BM25 인덱스 저장 경로
    hybrid_search_enabled: bool = True  # 하이브리드 검색 활성화
    hybrid_search_vector_weight: float = 0.7  # 벡터 검색 가중치
    hybrid_search_bm25_weight: float = 0.3  # BM25 검색 가중치
    hybrid_search_rrf_k: int = 60  # RRF 알고리즘 상수
    # FAISS 설정 (LangChain 기반)
    faiss_storage_path: str = "data/faiss_index"  # FAISS 인덱스 저장 경로
    faiss_use_gpu: bool = True  # GPU 사용 여부 (faiss-gpu 설치 필요, CUDA 자동 감지)
    # Sparse 벡터 설정 (Qdrant 네이티브 하이브리드 검색)
    sparse_enabled: bool = True  # Sparse 벡터 활성화 여부
    sparse_vector_name: str = "sparse"  # Sparse 벡터 이름
    hybrid_search_dense_weight: float = 0.7  # Dense 검색 가중치 (Qdrant 하이브리드 검색용)
    hybrid_search_sparse_weight: float = 0.3  # Sparse 검색 가중치 (Qdrant 하이브리드 검색용)
    sparse_vocabulary_path: str = "data/sparse_vocabulary"  # Sparse Vocabulary 저장 경로
    sparse_use_morphological: bool = True  # 형태소 분석 사용 여부 (KoNLPy 필요, 없으면 자동 폴백)
    sparse_include_doc_stats: bool = True  # doc_freqs/doc_len 저장 여부 (기본값: false, 파일 크기 증가)


class APIConfig(BaseModel):
    """API 서버 설정"""
    host: str = "0.0.0.0"  # 설정 파일의 기본값과 일치
    port: int = 8000  # 설정 파일의 기본값과 일치
    workers: int = 1  # 설정 파일의 기본값과 일치
    reload: bool = True  # 설정 파일의 기본값과 일치


class DataConfig(BaseModel):
    """데이터 처리 설정"""
    input_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    embeddings_dir: str = "data/embeddings"
    chunk_size: int = 600 # 설정 파일의 기본값과 일치
    chunk_overlap: int = 60  # 설정 파일의 기본값과 일치
    batch_size: int = 32  # SSD 최적화를 위한 배치 크기 증가


class LoggingConfig(BaseModel):
    """로깅 설정"""
    level: str = "INFO"
    file: str = "logs/stage1.log"
    format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}"


class RAGConfig(BaseModel):
    """RAG 시스템 설정"""
    default_max_sources: int = 5  # 기본 최대 소스 수
    default_max_sources_table: int = 5  # 표 검색 기본 최대 소스 수
    score_threshold: float = 0.7  # 통일된 검색 점수 임계값 (웹 인터페이스와 동일)
    content_preview_length: int = 300  # 소스 내용 미리보기 길이
    low_score_general_threshold: float = 0.3  # 낮은 점수 시 일반 질문으로 처리할 임계값 (일반 질문 판별용)


class RerankerConfig(BaseModel):
    """리랭커 설정 (Cross-Encoder)"""
    enabled: bool = False
    model_path: str = ""
    device: str = "cuda"
    top_k: int = 10
    batch_size: int = 32
    alpha: float = 0.7


class Stage1Config(BaseModel):
    """1단계 전체 설정"""
    stage: int = 1
    debug: bool = True
    model: Dict[str, ModelConfig] = Field(default_factory=lambda: {
        'embedding': ModelConfig(name='bona/bge-m3-korean:latest'),
        'llm': ModelConfig(name='gemma3:4b')
    })
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    feature_flags: Optional[Dict[str, Any]] = None
    agentic: Optional[Dict[str, Any]] = None    


class ConfigManager:
    """설정 관리자"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self._config: Optional[Stage1Config] = None
    
    def load_config(self) -> Stage1Config:
        """설정 파일 로드"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        # 모델 설정을 Pydantic 모델로 변환
        if 'model' in config_data:
            for model_type, model_data in config_data['model'].items():
                config_data['model'][model_type] = ModelConfig(**model_data)
        
        # 다른 설정들도 Pydantic 모델로 변환
        if 'qdrant' in config_data:
            config_data['qdrant'] = QdrantConfig(**config_data['qdrant'])
        
        if 'api' in config_data:
            config_data['api'] = APIConfig(**config_data['api'])
        
        if 'data' in config_data:
            config_data['data'] = DataConfig(**config_data['data'])
        
        if 'logging' in config_data:
            config_data['logging'] = LoggingConfig(**config_data['logging'])
        
        if 'rag' in config_data:
            config_data['rag'] = RAGConfig(**config_data['rag'])
        
        if 'reranker' in config_data:
            config_data['reranker'] = RerankerConfig(**config_data['reranker'])
        
        self._config = Stage1Config(**config_data)
        return self._config
    
    def get_config(self) -> Stage1Config:
        """설정 반환"""
        if self._config is None:
            self.load_config()
        return self._config
    
    def get_embedding_config(self) -> ModelConfig:
        """임베딩 모델 설정 반환"""
        config = self.get_config()
        return config.model['embedding']
    
    def get_llm_config(self) -> ModelConfig:
        """LLM 모델 설정 반환"""
        config = self.get_config()
        return config.model['llm']
    
    def get_qdrant_config(self) -> QdrantConfig:
        """Qdrant 설정 반환"""
        config = self.get_config()
        return config.qdrant
    
    def get_api_config(self) -> APIConfig:
        """API 설정 반환"""
        config = self.get_config()
        return config.api
    
    def get_data_config(self) -> DataConfig:
        """데이터 설정 반환"""
        config = self.get_config()
        return config.data
    
    def get_logging_config(self) -> LoggingConfig:
        """로깅 설정 반환"""
        config = self.get_config()
        return config.logging

    def get_reranker_config(self) -> RerankerConfig:
        """리랭커 설정 반환"""
        config = self.get_config()
        return config.reranker
    
# 전역 설정 관리자 인스턴스
config_manager = ConfigManager()


def get_config() -> Stage1Config:
    """전역 설정 반환"""
    return config_manager.get_config()


def get_embedding_config() -> ModelConfig:
    """임베딩 모델 설정 반환"""
    return config_manager.get_embedding_config()


def get_llm_config() -> ModelConfig:
    """LLM 모델 설정 반환"""
    return config_manager.get_llm_config()


def get_qdrant_config() -> QdrantConfig:
    """Qdrant 설정 반환"""
    return config_manager.get_qdrant_config()

def get_reranker_config() -> RerankerConfig:
    """리랭커 설정 반환"""
    return config_manager.get_reranker_config()

def get_api_config() -> APIConfig:
    """API 설정 반환"""
    return config_manager.get_api_config()


def get_data_config() -> DataConfig:
    """데이터 설정 반환"""
    return config_manager.get_data_config()


def get_logging_config() -> LoggingConfig:
    """로깅 설정 반환"""
    return config_manager.get_logging_config()


def get_rag_config() -> RAGConfig:
    """RAG 설정 반환"""
    config = config_manager.get_config()
    return config.rag
