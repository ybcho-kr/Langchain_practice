"""
FastAPI 기반 REST API 서버
"""

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Query
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, model_validator
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import time
import uvicorn
import shutil
from pathlib import Path

from src.utils.logger import setup_logging, get_logger
from src.utils.config import get_config, get_api_config
from src.modules.rag_system import RAGSystem, RAGResponse
from src.utils.langchain_utils import create_chat_ollama_from_config, create_simple_message

if TYPE_CHECKING:
    from src.session.session_manager import SessionManager

# 메트릭 API 라우터 포함
try:
    from src.api.metrics import router as metrics_router
except ImportError:
    metrics_router = None


# RebuildIndexesResponse 클래스 제거됨 (FAISS/BM25 제거)


# 요청/응답 모델
# 세션 관리자 전역 인스턴스
_session_manager: Optional["SessionManager"] = None


def get_session_manager() -> "SessionManager":
    """세션 관리자 싱글톤"""
    global _session_manager
    if _session_manager is None:
        from src.utils.config import get_session_config
        from src.utils.logger import get_logger
        logger = get_logger()
        
        session_config = get_session_config()
        logger.info(
            f"세션 설정 로드: storage_path={session_config.storage_path}, "
            f"use_sqlite={session_config.use_sqlite}, "
            f"session_ttl={session_config.session_ttl}, "
            f"cleanup_interval={session_config.cleanup_interval_seconds}"
        )
        
        from src.session.session_manager import SessionManager
        _session_manager = SessionManager(session_config=session_config)
    
    return _session_manager


class SessionCreateResponse(BaseModel):
    """세션 생성 응답"""
    session_id: str = Field(..., description="생성된 세션 ID")


class SessionInfo(BaseModel):
    """세션 정보"""
    session_id: str = Field(..., description="세션 ID", json_schema_extra={"example": "session-12345"})
    title: str = Field(..., description="세션 제목 (첫 메시지 또는 '새 대화')", json_schema_extra={"example": "변압기 진단 기준은?"})
    last_message: str = Field(..., description="마지막 메시지 미리보기", json_schema_extra={"example": "변압기 진단 기준은 다음과 같습니다..."})
    created_at: float = Field(..., description="생성 시간 (Unix timestamp)", json_schema_extra={"example": 1704067200.0})
    last_accessed: float = Field(..., description="마지막 접근 시간 (Unix timestamp)", json_schema_extra={"example": 1704067260.0})
    message_count: int = Field(..., description="메시지 수", json_schema_extra={"example": 5})


class SessionListResponse(BaseModel):
    """세션 목록 응답"""
    sessions: List[SessionInfo] = Field(..., description="세션 목록", json_schema_extra={"example": []})
    total_count: int = Field(..., description="총 세션 수", json_schema_extra={"example": 10})


class SessionHistoryResponse(BaseModel):
    """세션 히스토리 응답"""
    session_id: str = Field(..., description="세션 ID")
    history: List[Dict[str, Any]] = Field(..., description="메시지 히스토리")


class SessionStatsResponse(BaseModel):
    """세션 통계 응답"""
    total_sessions: int = Field(..., description="총 세션 수")
    total_memory_mb: float = Field(..., description="전체 메모리 사용량 (MB)")
    oldest_session_age: float = Field(..., description="가장 오래된 세션 연령 (초)")
    newest_session_age: float = Field(..., description="가장 최근 세션 연령 (초)")
    active_sessions: int = Field(..., description="활성 세션 수")


class QueryRequest(BaseModel):
    """
    RAG 질의응답 요청 모델
    
    사용자 질문과 검색 옵션을 포함합니다.
    """
    question: str = Field(
        ...,
        description="사용자 질문 (1-1000자)",
        min_length=1,
        max_length=1000,
        json_schema_extra={"example": "변압기 진단 기준은 무엇인가요?"}
    )
    max_sources: int = Field(
        default=5,
        description="검색 결과로 반환할 최대 소스 문서 수 (1-20)",
        ge=1,
        le=20,
        json_schema_extra={"example": 5}
    )
    score_threshold: float = Field(
        default=0.7,
        description="유사도 점수 임계값 (0.0-1.0). 이 값보다 낮은 점수의 문서는 제외됩니다.",
        ge=0.0,
        le=1.0,
        json_schema_extra={"example": 0.7}
    )
    temperature: Optional[float] = Field(
        default=None,
        description="LLM 생성 온도 (0.0-2.0). 높을수록 창의적인 답변, 낮을수록 일관된 답변. None이면 설정 파일 값 사용",
        ge=0.0,
        le=2.0,
        json_schema_extra={"example": 0.1}
    )
    model: Optional[str] = Field(
        default=None,
        description="사용할 LLM 모델명 (예: 'gemma3:4b', 'llama3.1:8b'). None이면 설정 파일의 기본 모델 사용",
        json_schema_extra={"example": "gemma3:4b"}
    )
    # Qdrant만 사용 (FAISS/BM25 제거됨)
    use_reranker: bool = Field(
        default=True,
        description="리랭커 사용 여부. CrossEncoder를 사용하여 검색 결과를 재정렬합니다.",
        json_schema_extra={"example": True}
    )
    reranker_alpha: float = Field(
        default=0.7,
        description="리랭커 점수 가중치 (0.0-1.0). 1.0에 가까울수록 리랭커 점수를 더 많이 반영합니다.",
        ge=0.0,
        le=1.0,
        json_schema_extra={"example": 0.7}
    )
    # reranker_top_k와 weights 필드 제거됨 (Qdrant만 사용)
    dense_weight: Optional[float] = Field(
        default=None,
        description="Qdrant 하이브리드 검색의 Dense 벡터 가중치 (0.0-1.0). None이면 config.yaml의 기본값 사용. Sparse 벡터가 활성화되어 있을 때만 의미가 있습니다.",
        ge=0.0,
        le=1.0,
        json_schema_extra={"example": 0.7}
    )
    sparse_weight: Optional[float] = Field(
        default=None,
        description="Qdrant 하이브리드 검색의 Sparse 벡터 가중치 (0.0-1.0). None이면 config.yaml의 기본값 사용. Sparse 벡터가 활성화되어 있을 때만 의미가 있습니다.",
        ge=0.0,
        le=1.0,
        json_schema_extra={"example": 0.3}
    )
    session_id: Optional[str] = Field(
        default=None,
        description="세션 ID (선택적). 멀티턴 대화를 위해 사용됩니다.",
        json_schema_extra={"example": "session-12345"}
    )
    
    # validate_retrievers_and_weights 메서드 제거됨 (Qdrant만 사용)


class QueryResponse(BaseModel):
    """
    RAG 질의응답 결과 모델
    
    생성된 답변과 출처 정보를 포함합니다.
    """
    answer: str = Field(..., description="생성된 답변 텍스트", json_schema_extra={"example": "변압기 진단 기준은 다음과 같습니다..."})
    sources: List[Dict[str, Any]] = Field(
        ...,
        description="답변 생성에 사용된 출처 문서 목록. 각 문서는 content, score, metadata 등을 포함합니다.",
        json_schema_extra={
            "example": [
            {
                "content": "변압기 진단 기준...",
                "score": 0.95,
                "source_file": "data/raw/transformer_guide.md",
                "chunk_index": 0
            }
        ]
        }
    )
    confidence: float = Field(
        ...,
        description="답변의 신뢰도 점수 (0.0-1.0). 검색 결과의 평균 점수를 기반으로 계산됩니다.",
        ge=0.0,
        le=1.0,
        json_schema_extra={"example": 0.92}
    )
    processing_time: float = Field(
        ...,
        description="전체 처리 시간 (초). 검색 + 리랭킹 + 답변 생성 시간을 포함합니다.",
        json_schema_extra={"example": 3.45}
    )
    query: str = Field(..., description="원본 질문", json_schema_extra={"example": "변압기 진단 기준은 무엇인가요?"})
    model_used: str = Field(..., description="사용된 LLM 모델명", json_schema_extra={"example": "gemma3:4b"})
    warnings: Optional[List[str]] = Field(
        default_factory=list,
        description="경고 메시지 목록. 예: 리랭커 요청했지만 사용 불가능한 경우",
        json_schema_extra={"example": ["리랭커 사용이 요청되었지만 리랭커가 초기화되지 않았습니다."]}
    )
    is_general_answer: bool = Field(
        default=False,
        description="일반 질문 답변 여부 (벡터 검색 없이 LLM 직접 답변). True이면 sources가 비어있을 수 있습니다.",
        json_schema_extra={"example": False}
    )
    is_rag_answer: bool = Field(
        default=True,
        description="RAG 답변 여부 (벡터 검색 + LLM 답변). False이면 일반 대화 답변입니다.",
        json_schema_extra={"example": True}
    )


class HealthResponse(BaseModel):
    """헬스 체크 응답"""
    status: str = Field(..., description="서버 상태", json_schema_extra={"example": "healthy"})
    timestamp: float = Field(..., description="응답 생성 시각 (Unix timestamp)", json_schema_extra={"example": 1234567890.123})
    version: str = Field(..., description="API 버전", json_schema_extra={"example": "1.0.0"})


class StatsResponse(BaseModel):
    """
    시스템 통계 응답 모델
    
    시스템의 현재 상태 및 통계 정보를 포함합니다.
    """
    embedding_cache_stats: Dict[str, Any] = Field(
        ...,
        description="임베딩 캐시 통계 (캐시 크기, 히트/미스율, 모델 정보 등)",
        json_schema_extra={"example": {"cache_size": 1000, "hits": 500, "misses": 200, "model_name": "BGE-m3-ko", "dimension": 1024}}
    )
    vector_store_stats: Dict[str, Any] = Field(
        ...,
        description="벡터 저장소 통계 (총 문서 수, 청크 수, 컬렉션 정보 등)",
        json_schema_extra={"example": {"total_documents": 50, "total_chunks": 500, "collection_name": "electrical_diagnosis"}}
    )
    llm_model: str = Field(..., description="현재 사용 중인 LLM 모델명", json_schema_extra={"example": "gemma3:4b"})


class ModelsResponse(BaseModel):
    """
    사용 가능한 모델 목록 응답 모델
    
    Ollama 서버에서 사용 가능한 모델 목록을 포함합니다.
    """
    available_models: List[Dict[str, str]] = Field(
        ...,
        description="사용 가능한 모델 목록. 각 모델은 name, size, modified_at, family 정보를 포함합니다.",
        json_schema_extra={
            "example": [
            {"name": "gemma3:4b", "size": "2.5GB", "modified_at": "2024-01-01T00:00:00Z", "family": "gemma"},
            {"name": "llama3.1:8b", "size": "4.7GB", "modified_at": "2024-01-01T00:00:00Z", "family": "llama"}
        ]
        }
    )
    current_model: str = Field(..., description="현재 사용 중인 모델명", json_schema_extra={"example": "gemma3:4b"})


class ConfigResponse(BaseModel):
    """
    시스템 설정 응답 모델
    
    현재 시스템 설정 정보를 포함합니다.
    """
    llm_model: str = Field(..., description="현재 사용 중인 LLM 모델명", json_schema_extra={"example": "gemma3:4b"})
    embedding_model: str = Field(..., description="현재 사용 중인 임베딩 모델명", json_schema_extra={"example": "BGE-m3-ko"})
    max_sources: int = Field(..., description="기본 최대 소스 수", json_schema_extra={"example": 5})
    score_threshold: float = Field(..., description="기본 점수 임계값", json_schema_extra={"example": 0.7})
    default_limit: int = Field(..., description="기본 검색 제한값", json_schema_extra={"example": 5})
    reranker_model: Optional[str] = Field(
        default=None,
        description="리랭커 모델 경로 또는 이름",
        json_schema_extra={"example": "ms-marco-MiniLM-L-6-v2"}
    )
    reranker_enabled: bool = Field(
        default=False,
        description="리랭커 활성화 여부",
        json_schema_extra={"example": True}
    )
    reranker_alpha: Optional[float] = Field(
        default=None,
        description="리랭커 alpha 값 (0.0-1.0)",
        json_schema_extra={"example": 0.7}
    )


class DocumentUploadResponse(BaseModel):
    """
    문서 업로드 응답 모델
    
    문서 업로드 및 처리 결과를 포함합니다.
    """
    success: bool = Field(..., description="업로드 성공 여부", json_schema_extra={"example": True})
    message: str = Field(..., description="응답 메시지", json_schema_extra={"example": "3개 파일 처리 완료"})
    processed_files: int = Field(..., description="처리된 파일 수", json_schema_extra={"example": 3})
    total_chunks: int = Field(..., description="생성된 청크 수", json_schema_extra={"example": 150})
    processing_time: float = Field(..., description="처리 시간(초)", json_schema_extra={"example": 12.34})


# Lifespan 이벤트 핸들러 (Pydantic V2 호환)
@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 생명주기 관리 (startup/shutdown)"""
    # Startup
    global rag_system
    
    # 로깅 설정
    setup_logging()
    logger = get_logger()
    
    logger.info("API 서버 시작 중...")
    
    try:
        # 설정 로드
        config = get_config()
        embedding_name = config.model.get('embedding').name if config.model.get('embedding') else 'unknown'
        llm_name = config.model.get('llm').name if config.model.get('llm') else 'unknown'
        logger.info(f"설정 로드 완료: 임베딩={embedding_name}, LLM={llm_name}")
        
        # RAG 시스템 초기화 (설정 명시적 전달)
        rag_system = RAGSystem(config)
        
        # 벡터 저장소 설정
        if not rag_system.vector_store.create_collection(force_recreate=False):
            logger.error("벡터 저장소 설정 실패")
            raise Exception("벡터 저장소 설정 실패")
        
        # Kiwipiepy 전처리 상태 로깅
        try:
            # config에서 직접 확인
            from src.utils.config import get_qdrant_config
            qdrant_config = get_qdrant_config()
            config_kiwipiepy_value = qdrant_config.sparse_use_kiwipiepy if hasattr(qdrant_config, 'sparse_use_kiwipiepy') else True
            logger.info(f"[Kiwipiepy 설정 확인] config.yaml의 sparse_use_kiwipiepy 값: {config_kiwipiepy_value}")
            
            kiwipiepy_enabled = rag_system.vector_store.use_kiwipiepy_preprocessing
            kiwipiepy_dict = rag_system.vector_store.kiwipiepy_dictionary_path
            kiwipiepy_preprocessor = rag_system.vector_store.kiwipiepy_preprocessor
            logger.info(f"[Kiwipiepy 상태] vector_store.use_kiwipiepy_preprocessing: {kiwipiepy_enabled}")
            logger.info(f"[Kiwipiepy 상태] kiwipiepy_preprocessor 존재: {kiwipiepy_preprocessor is not None}")
            if kiwipiepy_preprocessor:
                logger.info(f"[Kiwipiepy 상태] kiwipiepy_preprocessor.use_kiwipiepy: {kiwipiepy_preprocessor.use_kiwipiepy}")
                logger.info(f"[Kiwipiepy 상태] kiwipiepy_preprocessor.kiwi 존재: {kiwipiepy_preprocessor.kiwi is not None}")
            
            if kiwipiepy_enabled and kiwipiepy_preprocessor and kiwipiepy_preprocessor.use_kiwipiepy:
                logger.info(
                    f"✅ Kiwipiepy 형태소 전처리 활성화 (사전 경로: {kiwipiepy_dict if kiwipiepy_dict else '기본 사전'})"
                )
            else:
                if not kiwipiepy_enabled:
                    logger.warning(
                        f"❌ Kiwipiepy 형태소 전처리 비활성화 "
                        f"(config.yaml의 sparse_use_kiwipiepy={config_kiwipiepy_value}, "
                        f"vector_store.use_kiwipiepy_preprocessing={kiwipiepy_enabled})"
                    )
                elif not kiwipiepy_preprocessor:
                    logger.warning("❌ Kiwipiepy 형태소 전처리 비활성화 (KiwipiepyPreprocessor 초기화 실패 - Kiwipiepy 설치 확인 필요)")
                elif not kiwipiepy_preprocessor.use_kiwipiepy:
                    logger.warning("❌ Kiwipiepy 형태소 전처리 비활성화 (KiwipiepyPreprocessor.use_kiwipiepy = False - Kiwipiepy 설치 또는 초기화 실패)")
        except Exception as kiwipiepy_log_error:
            logger.warning(f"Kiwipiepy 전처리 상태 로깅 실패: {kiwipiepy_log_error}")
            import traceback
            logger.error(f"상세 오류: {traceback.format_exc()}")
        
        # 세션 관리자 초기화 (서버 시작 시점에 초기화하여 SQLite 설정 확인)
        try:
            session_manager = get_session_manager()
            logger.info("세션 관리자 초기화 완료")
        except Exception as e:
            logger.warning(f"세션 관리자 초기화 중 오류 발생 (계속 진행): {str(e)}")
        
        logger.info("API 서버 시작 완료")
        
    except Exception as e:
        logger.error(f"서버 시작 실패: {str(e)}")
        raise
    
    # Yield로 앱 실행 중 상태 유지
    yield
    
    # Shutdown (필요시 정리 작업)
    logger.info("API 서버 종료 중...")


# FastAPI 앱 생성
app = FastAPI(
    title="전기설비 진단 RAG 시스템",
    description="""
    전기설비 진단 로컬 LLM RAG 시스템 API
    
    ## 주요 기능
    - **질의응답**: RAG 기반 지식 검색 및 답변 생성
    - **문서 관리**: 문서 업로드, 처리, 조회, 삭제
    - **인덱스 관리**: FAISS, BM25 인덱스 재구축 및 캐시 관리
    - **모델 관리**: 임베딩/리랭커 모델 동적 재로드 및 GPU 메모리 관리
    - **시스템 모니터링**: 통계 조회, 설정 확인, 헬스 체크
    
    ## 검색 전략
    - **Qdrant**: 벡터 검색 (기본)
    - **FAISS**: 고속 벡터 검색 (GPU 지원)
    - **BM25**: 키워드 검색
    - **하이브리드**: EnsembleRetriever (FAISS + BM25 + RRF)
    - **리랭킹**: CrossEncoder 기반 결과 재정렬
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 메트릭 라우터 포함
if metrics_router:
    app.include_router(metrics_router)

# 전역 RAG 시스템 인스턴스
rag_system: Optional[RAGSystem] = None


def get_rag_system() -> RAGSystem:
    """RAG 시스템 의존성"""
    global rag_system
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG 시스템이 초기화되지 않았습니다")
    return rag_system


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Favicon 핸들러 (404 오류 방지)"""
    return Response(status_code=204)  # No Content


@app.get(
    "/",
    response_model=Dict[str, str],
    tags=["시스템 관리"],
    summary="API 루트 엔드포인트",
    description="API 서버의 기본 정보를 반환합니다. API 버전 및 문서 링크를 제공합니다.",
    response_description="API 기본 정보 (메시지, 버전, 문서 URL)"
)
async def root():
    """루트 엔드포인트"""
    return {
        "message": "전기설비 진단 RAG 시스템 API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["시스템 관리"],
    summary="헬스 체크",
    description="API 서버의 상태를 확인합니다. 서버가 정상적으로 동작 중인지 확인할 때 사용합니다.",
    response_description="서버 상태 정보 (상태, 타임스탬프, 버전)"
)
async def health_check():
    """헬스 체크"""
    return HealthResponse(
        status="healthy",
        timestamp=time.time(),
        version="1.0.0"
    )


@app.post(
    "/query",
    response_model=QueryResponse,
    tags=["질의응답"],
    summary="RAG 질의응답",
    description="""
    RAG(Retrieval-Augmented Generation) 기반 질의응답을 수행합니다.
    
    ## 검색 전략
    - **Qdrant**: 벡터 유사도 검색 (기본값, 필수)
    - **FAISS**: 고속 벡터 검색 (현재 숨김 처리됨)
    - **BM25**: 키워드 기반 검색 (현재 숨김 처리됨)
    
    ## 리랭킹
    - CrossEncoder 기반 결과 재정렬 지원
    - 리랭커 활성화 시 검색 정확도 향상
    
    ## 주의사항
    - 현재는 Qdrant만 사용 가능합니다
    - 검색기 가중치는 자동으로 정규화됩니다
    """,
    response_description="생성된 답변, 출처 문서, 신뢰도, 처리 시간 등"
)
async def query(request: QueryRequest, rag: RAGSystem = Depends(get_rag_system)):
    """질의응답 (비동기)"""
    try:
        # Qdrant만 사용 (FAISS, BM25는 제거됨)
        # use_qdrant 필드는 더 이상 필요 없음 (항상 Qdrant 사용)
        
        # 입력 파라미터 검증
        if not isinstance(request.question, str) or not request.question.strip():
            raise HTTPException(status_code=400, detail="question은 비어있지 않은 문자열이어야 합니다.")
        
        if request.max_sources is not None and (not isinstance(request.max_sources, int) or request.max_sources < 1):
            raise HTTPException(status_code=400, detail="max_sources는 1 이상의 정수여야 합니다.")
        
        if request.score_threshold is not None and (not isinstance(request.score_threshold, (int, float)) or request.score_threshold < 0.0 or request.score_threshold > 1.0):
            raise HTTPException(status_code=400, detail="score_threshold는 0.0과 1.0 사이의 값이어야 합니다.")
        
        if request.session_id is not None and (not isinstance(request.session_id, str) or not request.session_id.strip()):
            raise HTTPException(status_code=400, detail="session_id는 비어있지 않은 문자열이어야 합니다.")
        
        # 검색기 선택 정보 구성 (Qdrant만 사용)
        retrievers = {
            'use_qdrant': True,  # Qdrant만 사용
            'use_reranker': bool(request.use_reranker),
            'reranker_alpha': float(request.reranker_alpha) if request.reranker_alpha is not None else None,
            'dense_weight': float(request.dense_weight) if request.dense_weight is not None else None,
            'sparse_weight': float(request.sparse_weight) if request.sparse_weight is not None else None
        }
        
        # 세션에 사용자 메시지 추가
        if request.session_id:
            try:
                session_manager = get_session_manager()
                if session_manager is None:
                    logger = get_logger()
                    logger.warning("세션 관리자를 가져올 수 없습니다. 메시지 저장을 건너뜁니다.")
                else:
                    success = session_manager.add_message(
                        session_id=request.session_id,
                        role='user',
                        content=request.question
                    )
                    if not success:
                        logger = get_logger()
                        logger.warning(f"세션 메시지 추가 실패: {request.session_id}")
            except Exception as e:
                logger = get_logger()
                logger.warning(f"세션 메시지 추가 중 오류 발생 (무시): {str(e)}")
        
        # RAG 시스템을 통한 질의 처리 (비동기 메서드 사용)
        response = await rag.query_async(
            question=request.question,
            max_sources=request.max_sources,
            score_threshold=request.score_threshold,
            model_name=request.model,
            retrievers=retrievers,
            session_id=request.session_id,
            dense_weight=request.dense_weight,
            sparse_weight=request.sparse_weight
        )
        
        # RAGResponse 검증
        if response is None:
            raise HTTPException(status_code=500, detail="RAG 시스템이 응답을 생성하지 못했습니다.")
        
        # 필수 필드 검증
        if not hasattr(response, 'answer') or response.answer is None:
            raise HTTPException(status_code=500, detail="RAG 응답에 answer 필드가 없습니다.")
        if not hasattr(response, 'sources') or not isinstance(response.sources, list):
            raise HTTPException(status_code=500, detail="RAG 응답에 sources 필드가 없거나 리스트가 아닙니다.")
        if not hasattr(response, 'confidence') or response.confidence is None:
            raise HTTPException(status_code=500, detail="RAG 응답에 confidence 필드가 없습니다.")
        if not hasattr(response, 'processing_time') or response.processing_time is None:
            raise HTTPException(status_code=500, detail="RAG 응답에 processing_time 필드가 없습니다.")
        if not hasattr(response, 'query') or response.query is None:
            raise HTTPException(status_code=500, detail="RAG 응답에 query 필드가 없습니다.")
        if not hasattr(response, 'model_used') or response.model_used is None:
            raise HTTPException(status_code=500, detail="RAG 응답에 model_used 필드가 없습니다.")
        
        # 리랭커 요청했지만 사용 불가능한 경우 경고 추가
        reranker_requested = request.use_reranker
        reranker_available = rag.reranker is not None
        warnings = []
        if reranker_requested and not reranker_available:
            warnings.append("리랭커 사용이 요청되었지만 리랭커가 초기화되지 않았습니다. config.yaml에서 reranker.enabled: true로 설정하세요.")
        
        # 세션에 어시스턴트 메시지 추가
        if request.session_id:
            try:
                session_manager = get_session_manager()
                if session_manager is None:
                    logger = get_logger()
                    logger.warning("세션 관리자를 가져올 수 없습니다. 메시지 저장을 건너뜁니다.")
                else:
                    success = session_manager.add_message(
                        session_id=request.session_id,
                        role='assistant',
                        content=response.answer,
                        search_results=response.sources,
                        model_used=response.model_used,
                        confidence=response.confidence,
                        processing_time=response.processing_time
                    )
                    if not success:
                        logger = get_logger()
                        logger.warning(f"세션 메시지 추가 실패: {request.session_id}")
            except Exception as e:
                logger = get_logger()
                logger.warning(f"세션 메시지 추가 중 오류 발생 (무시): {str(e)}")
        
        return QueryResponse(
            answer=response.answer,
            sources=response.sources,
            confidence=response.confidence,
            processing_time=response.processing_time,
            query=response.query,
            model_used=response.model_used,
            warnings=warnings,
            is_general_answer=bool(response.is_general_answer) if hasattr(response, 'is_general_answer') else False,
            is_rag_answer=bool(response.is_rag_answer) if hasattr(response, 'is_rag_answer') else True
        )
        
    except Exception as e:
        logger = get_logger()
        logger.error(f"질의 처리 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"질의 처리 실패: {str(e)}")


@app.post(
    "/process-documents",
    tags=["문서 관리"],
    summary="문서 처리 (레거시)",
    description="""
    지정된 디렉토리의 문서를 처리합니다. (레거시 엔드포인트)
    
    ## 참고
    - `/process-directory` 엔드포인트 사용을 권장합니다
    - 이 엔드포인트는 기본 디렉토리(`data/raw`)를 사용합니다
    
    ## 파라미터
    - **input_dir**: 처리할 디렉토리 경로 (기본값: `data/raw`)
    """,
    response_description="처리 성공 여부 및 상태 메시지"
)
async def process_documents(
    input_dir: str = "data/raw",
    rag: RAGSystem = Depends(get_rag_system)
):
    """문서 처리"""
    try:
        # 입력 파라미터 검증
        if not isinstance(input_dir, str) or not input_dir.strip():
            raise HTTPException(status_code=400, detail="input_dir는 비어있지 않은 문자열이어야 합니다.")
        
        success = rag.process_and_store_documents(input_dir)
        
        # 반환값 검증
        if success is None:
            raise HTTPException(status_code=500, detail="문서 처리 결과를 확인할 수 없습니다.")
        
        if not isinstance(success, bool):
            logger = get_logger()
            logger.warning(f"process_and_store_documents 반환값이 bool이 아닙니다: {type(success)}")
            success = bool(success)
        
        if success:
            return {"message": "문서 처리 완료", "status": "success"}
        else:
            raise HTTPException(status_code=500, detail="문서 처리 실패")
            
    except HTTPException:
        raise
    except Exception as e:
        logger = get_logger()
        logger.error(f"문서 처리 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"문서 처리 실패: {str(e)}")


@app.get(
    "/stats",
    response_model=StatsResponse,
    tags=["시스템 관리"],
    summary="시스템 통계 조회",
    description="""
    시스템의 현재 상태 및 통계 정보를 조회합니다.
    
    ## 포함 정보
    - **임베딩 캐시 통계**: 캐시 크기, 히트/미스율, 모델 정보
    - **벡터 저장소 통계**: 총 문서 수, 청크 수, 컬렉션 정보
    - **LLM 모델**: 현재 사용 중인 LLM 모델명
    
    ## 활용
    - 시스템 모니터링
    - 성능 최적화 참고
    - 리소스 사용량 확인
    """,
    response_description="임베딩 캐시, 벡터 저장소, LLM 모델 통계"
)
async def get_stats(rag: RAGSystem = Depends(get_rag_system)):
    """시스템 통계"""
    try:
        stats = rag.get_system_stats()
        
        # 기본값 제공
        if not stats:
            stats = {
                'vector_store_stats': {},
                'embedding_cache_stats': {'cache_size': 0, 'model_name': 'unknown', 'dimension': 1024},
                'llm_model': 'unknown'
            }
        
        return StatsResponse(**stats)
        
    except Exception as e:
        logger = get_logger()
        logger.error(f"통계 조회 실패: {str(e)}")
        # 오류 시에도 기본값 반환
        return StatsResponse(
            vector_store_stats={},
            embedding_cache_stats={'cache_size': 0, 'model_name': 'unknown', 'dimension': 1024},
            llm_model='unknown'
        )


@app.get(
    "/models",
    response_model=ModelsResponse,
    tags=["시스템 관리"],
    summary="사용 가능한 모델 목록 조회",
    description="""
    Ollama 서버에서 사용 가능한 LLM 모델 목록을 조회합니다.
    
    ## 정보 포함
    - 모델 이름
    - 모델 크기
    - 수정 일시
    - 모델 패밀리 (llama, gemma, qwen 등)
    - 현재 사용 중인 모델
    
    ## 주의사항
    - Ollama 서버가 실행 중이어야 합니다
    - Ollama 서버에 연결할 수 없는 경우 기본 모델 목록을 반환합니다
    """,
    response_description="사용 가능한 모델 목록 및 현재 사용 중인 모델"
)
async def get_available_models(rag: RAGSystem = Depends(get_rag_system)):
    """사용 가능한 모델 목록 조회"""
    try:
        # Ollama에서 사용 가능한 모델 목록 조회
        import requests
        
        ollama_url = "http://localhost:11434/api/tags"
        response = requests.get(ollama_url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            available_models = []
            
            for model in data.get('models', []):
                model_info = {
                    'name': model['name'],
                    'size': f"{model.get('size', 0) / (1024**3):.1f}GB",
                    'modified_at': model.get('modified_at', ''),
                    'family': model.get('details', {}).get('family', 'unknown')
                }
                available_models.append(model_info)
            
            # 현재 사용 중인 모델 확인
            current_model = rag.llm_client.model_name if hasattr(rag, 'llm_client') else 'llama3.1:8b'
            
            return ModelsResponse(
                available_models=available_models,
                current_model=current_model
            )
        else:
            # Ollama 서버에 연결할 수 없는 경우 기본 모델 목록 반환
            return ModelsResponse(
                available_models=[
                    {'name': 'llama3.1:8b', 'size': '4.7GB', 'modified_at': '', 'family': 'llama'},
                    {'name': 'llama3.1:70b', 'size': '40.0GB', 'modified_at': '', 'family': 'llama'},
                    {'name': 'llama3.2:3b', 'size': '2.0GB', 'modified_at': '', 'family': 'llama'},
                    {'name': 'qwen2.5:7b', 'size': '4.4GB', 'modified_at': '', 'family': 'qwen'},
                    {'name': 'gemma2:9b', 'size': '5.4GB', 'modified_at': '', 'family': 'gemma'}
                ],
                current_model='llama3.1:8b'
            )
        
    except Exception as e:
        logger = get_logger()
        logger.error(f"모델 목록 조회 실패: {str(e)}")
        # 오류 시 기본 모델 목록 반환
        return ModelsResponse(
            available_models=[
                {'name': 'llama3.1:8b', 'size': '4.7GB', 'modified_at': '', 'family': 'llama'},
                {'name': 'llama3.1:70b', 'size': '40.0GB', 'modified_at': '', 'family': 'llama'},
                {'name': 'llama3.2:3b', 'size': '2.0GB', 'modified_at': '', 'family': 'llama'},
                {'name': 'qwen2.5:7b', 'size': '4.4GB', 'modified_at': '', 'family': 'qwen'},
                {'name': 'gemma2:9b', 'size': '5.4GB', 'modified_at': '', 'family': 'gemma'}
            ],
            current_model='llama3.1:8b'
        )


@app.get(
    "/config",
    response_model=ConfigResponse,
    tags=["시스템 관리"],
    summary="시스템 설정 조회",
    description="""
    현재 시스템 설정 정보를 조회합니다.
    
    ## 포함 정보
    - **LLM 모델**: 현재 사용 중인 LLM 모델명
    - **임베딩 모델**: 현재 사용 중인 임베딩 모델명
    - **검색 설정**: 최대 소스 수, 점수 임계값, 기본 제한값
    - **리랭커 설정**: 리랭커 모델, 활성화 여부, alpha 값
    
    ## 활용
    - 현재 설정 확인
    - API 요청 시 기본값 참고
    - 설정 변경 전 현재 상태 확인
    """,
    response_description="LLM, 임베딩, 검색, 리랭커 설정 정보"
)
async def get_system_config():
    """시스템 설정 조회 (설정 파일에서 가져옴)"""
    try:
        from src.utils.config import get_llm_config, get_embedding_config, get_rag_config, get_qdrant_config, get_reranker_config
        
        llm_config = get_llm_config()
        embedding_config = get_embedding_config()
        rag_config = get_rag_config()
        qdrant_config = get_qdrant_config()
        reranker_config = get_reranker_config()
        
        # 리랭커 모델 이름 생성 (경로에서 마지막 디렉토리명 또는 전체 경로)
        reranker_model_name = None
        if reranker_config.enabled and reranker_config.model_path:
            # 경로에서 모델 이름 추출 (마지막 디렉토리명)
            import os
            reranker_model_name = os.path.basename(reranker_config.model_path.rstrip('\\/')) or reranker_config.model_path
        
        return ConfigResponse(
            llm_model=llm_config.name,
            embedding_model=embedding_config.name,
            max_sources=rag_config.default_max_sources,
            score_threshold=rag_config.score_threshold,
            default_limit=qdrant_config.default_limit,
            reranker_model=reranker_model_name,
            reranker_enabled=reranker_config.enabled,
            reranker_alpha=reranker_config.alpha if reranker_config.enabled else None
        )
        
    except Exception as e:
        logger = get_logger()
        logger.error(f"설정 조회 실패: {str(e)}")
        # 오류 시 기본값 반환
        return ConfigResponse(
            llm_model='gemma3:4b',
            embedding_model='BGE-m3-ko',
            max_sources=5,
            score_threshold=0.7,
            default_limit=5,
            reranker_model=None,
            reranker_enabled=False,
            reranker_alpha=None
        )


class DocumentInfoResponse(BaseModel):
    """
    문서 정보 응답 모델
    
    벡터 저장소에 저장된 문서 목록 정보를 포함합니다.
    """
    total_documents: int = Field(..., description="총 문서 수", json_schema_extra={"example": 50})
    documents: List[Dict[str, Any]] = Field(
        ...,
        description="문서 목록. 각 문서는 파일명, 경로, 청크 수, 메타데이터 등을 포함합니다.",
        json_schema_extra={
            "example": [
            {
                "file_name": "transformer_guide.md",
                "source_file": "data/raw/transformer_guide.md",
                "chunk_count": 10,
                "metadata": {}
            }
        ]
        }
    )
    total_chunks: int = Field(..., description="총 청크 수", json_schema_extra={"example": 500})
    collection_info: Dict[str, Any] = Field(
        ...,
        description="컬렉션 정보 (컬렉션명, 포인트 수 등)",
        json_schema_extra={"example": {"collection_name": "electrical_diagnosis", "points_count": 500}}
    )


class DocumentDeleteRequest(BaseModel):
    """문서 삭제 요청"""
    source_file: str = Field(..., description="삭제할 문서의 source_file 경로")


class DocumentDeleteResponse(BaseModel):
    """
    문서 삭제 응답 모델
    
    문서 삭제 결과 및 각 저장소별 삭제 상태를 포함합니다.
    """
    success: bool = Field(..., description="삭제 성공 여부", json_schema_extra={"example": True})
    message: str = Field(..., description="응답 메시지", json_schema_extra={"example": "문서 삭제 완료: 10개 청크 삭제됨"})
    deleted_chunks_count: int = Field(default=0, description="삭제된 청크 수", json_schema_extra={"example": 10})
    qdrant_deleted: bool = Field(default=False, description="Qdrant 삭제 성공 여부", json_schema_extra={"example": True})
    # FAISS/BM25 관련 필드 제거됨 (Qdrant만 사용)
    warnings: List[str] = Field(
        default_factory=list,
        description="경고 메시지 목록",
        json_schema_extra={"example": []}
    )


@app.get(
    "/documents",
    response_model=DocumentInfoResponse,
    tags=["문서 관리"],
    summary="문서 목록 조회",
    description="""
    벡터 저장소에 저장된 모든 문서의 목록을 조회합니다.
    
    ## 포함 정보
    - 총 문서 수
    - 각 문서의 상세 정보 (파일명, 경로, 청크 수, 메타데이터 등)
    - 총 청크 수
    - 컬렉션 정보
    
    ## 활용
    - 저장된 문서 확인
    - 문서 관리 및 모니터링
    - 특정 문서 조회 전 목록 확인
    """,
    response_description="문서 목록, 총 문서 수, 총 청크 수, 컬렉션 정보"
)
async def get_documents(rag: RAGSystem = Depends(get_rag_system)):
    """벡터 DB에 저장된 문서 목록 조회"""
    try:
        # 벡터 저장소에서 문서 정보 조회
        collection_info = rag.vector_store.get_collection_info()
        
        # 저장된 문서들의 메타데이터 조회
        documents_info = rag.vector_store.get_documents_info()
        
        # 응답 검증: documents_info가 리스트인지 확인
        if not isinstance(documents_info, list):
            logger = get_logger()
            logger.error(f"get_documents_info 반환값이 리스트가 아닙니다: {type(documents_info)}")
            documents_info = []
        
        # 각 문서 정보 검증
        validated_documents = []
        for doc in documents_info:
            if isinstance(doc, dict):
                # source_file이 없거나 비어있으면 건너뛰기
                source_file = doc.get('source_file', '')
                if not source_file or source_file.strip() == '' or source_file == 'unknown':
                    continue
                validated_documents.append(doc)
            else:
                # dict가 아닌 경우도 건너뛰기
                continue
        
        return DocumentInfoResponse(
            total_documents=len(validated_documents),
            documents=validated_documents,
            total_chunks=collection_info.get('points_count', 0) if isinstance(collection_info, dict) else 0,
            collection_info=collection_info if isinstance(collection_info, dict) else {}
        )
        
    except Exception as e:
        logger = get_logger()
        logger.error(f"문서 목록 조회 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"문서 목록 조회 실패: {str(e)}")


@app.get(
    "/documents/{document_id}/chunks",
    tags=["문서 관리"],
    summary="문서 청크 조회",
    description="""
    특정 문서의 모든 청크 정보를 조회합니다.
    
    ## 파라미터
    - **document_id**: 조회할 문서의 source_file 경로
    
    ## 포함 정보
    - 문서 ID
    - 총 청크 수
    - 각 청크의 상세 정보 (내용, 메타데이터, 인덱스 등)
    
    ## 활용
    - 문서 내용 확인
    - 청크 분할 상태 확인
    - 문서 검색 결과 분석
    """,
    response_description="문서 ID, 총 청크 수, 청크 목록"
)
async def get_document_chunks(document_id: str, rag: RAGSystem = Depends(get_rag_system)):
    """특정 문서의 청크 정보 조회"""
    try:
        # 입력 검증: document_id가 유효한지 확인
        if not document_id or document_id.strip() == '':
            raise HTTPException(status_code=400, detail="document_id는 비어있을 수 없습니다.")
        
        if document_id == 'N/A' or document_id.strip() == 'N/A':
            raise HTTPException(
                status_code=400, 
                detail="'N/A'는 유효한 문서 ID가 아닙니다. 검색 결과에서 source_file이 없는 경우입니다."
            )
        
        chunks = rag.vector_store.get_document_chunks(document_id)
        
        # 결과 검증: 문서가 존재하는지 확인
        if not chunks or len(chunks) == 0:
            raise HTTPException(
                status_code=404, 
                detail=f"문서를 찾을 수 없습니다: {document_id}"
            )
        
        return {
            "document_id": document_id,
            "total_chunks": len(chunks),
            "chunks": chunks
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger = get_logger()
        logger.error(f"문서 청크 조회 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"문서 청크 조회 실패: {str(e)}")


@app.get(
    "/sparse-vocabulary",
    tags=["시스템 관리"],
    summary="Sparse 벡터 Vocabulary 조회",
    description="""
    Sparse 벡터에 저장된 Vocabulary 정보를 조회합니다.
    
    ## 포함 정보
    - **Vocabulary 크기**: 전체 토큰 수
    - **Corpus 크기**: 학습에 사용된 문서 수
    - **Vocabulary 목록**: 토큰과 인덱스 매핑
    - **IDF 값**: 각 토큰의 IDF (Inverse Document Frequency) 값
    - **통계 정보**: IDF 최소/최대/평균값, 상위 토큰 목록
    
    ## 쿼리 파라미터
    - **limit**: 반환할 vocabulary 항목 수 (기본값: 최대 1000개)
    - **search_token**: 특정 토큰 검색 (토큰이 포함된 항목만 반환)
    
    ## 활용
    - Vocabulary 구성 확인
    - 특정 토큰의 IDF 값 확인
    - Sparse 벡터 모델 학습 상태 확인
    
    ## 주의사항
    - Sparse 벡터가 비활성화되어 있거나 모델이 학습되지 않은 경우 메시지가 반환됩니다
    - Vocabulary가 매우 큰 경우 기본적으로 상위 1000개만 반환됩니다
    """,
    response_description="Vocabulary 정보 (토큰, 인덱스, IDF 값, 통계)"
)
async def get_sparse_vocabulary(
    limit: Optional[int] = None,
    search_token: Optional[str] = None,
    rag: RAGSystem = Depends(get_rag_system)
):
    """Sparse 벡터 Vocabulary 조회"""
    try:
        vocabulary_info = rag.vector_store.get_sparse_vocabulary(limit, search_token)
        return vocabulary_info
        
    except Exception as e:
        logger = get_logger()
        logger.error(f"Vocabulary 조회 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Vocabulary 조회 실패: {str(e)}")


@app.get(
    "/analyze-sparse-quality",
    tags=["시스템 관리"],
    summary="Sparse 벡터 DB 품질 분석",
    description="""
    Qdrant에 저장된 Sparse 벡터의 품질을 분석합니다.
    
    ## 분석 항목
    - **기본 통계**: 총 포인트 수, Sparse 벡터 포함/미포함 비율
    - **토큰 통계**: 포인트당 평균 토큰 수, 분포
    - **가중치 통계**: 평균, 중앙값, 분포
    - **Vocabulary 통계**: 고유 토큰 수, 빈도 상위 토큰
    - **품질 평가**: 발견된 문제점 및 개선 제안
    
    ## 활용
    - Sparse 벡터 DB 품질 확인
    - 데이터 부족 여부 판단
    - 개선 방향 제시
    """,
    response_description="Sparse 벡터 품질 분석 결과"
)
async def analyze_sparse_quality(rag: RAGSystem = Depends(get_rag_system)):
    """Sparse 벡터 DB 품질 분석"""
    try:
        quality_report = rag.vector_store.analyze_sparse_quality()
        return quality_report
        
    except Exception as e:
        logger = get_logger()
        logger.error(f"Sparse 벡터 품질 분석 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"품질 분석 실패: {str(e)}")


@app.get(
    "/inspect-vectors",
    tags=["시스템 관리"],
    summary="벡터 확인 (Dense + Sparse)",
    description="""
    저장된 포인트의 Dense와 Sparse 벡터를 확인합니다.
    
    ## 사용 방법
    - **sample_size**: 랜덤 샘플 포인트 수 (기본값: 3)
    - **point_ids**: 특정 포인트 ID 리스트 (지정 시 sample_size 무시)
    
    ## 확인 항목
    - Dense 벡터: 크기, 미리보기 (처음 5개 값)
    - Sparse 벡터: 토큰 수, 가중치 수, 미리보기 (처음 10개)
    - Payload: 메타데이터 정보
    
    ## 활용
    - 업로드한 문서의 벡터 생성 여부 확인
    - Dense/Sparse 벡터 품질 검증
    - 특정 포인트의 벡터 구조 확인
    """,
    response_description="벡터 확인 결과 (컬렉션 정보, 샘플 포인트의 Dense/Sparse 벡터)"
)
async def inspect_vectors(
    sample_size: int = Query(3, ge=1, le=100, description="확인할 샘플 포인트 수"),
    point_ids: Optional[str] = Query(None, description="특정 포인트 ID 리스트 (쉼표로 구분)"),
    rag: RAGSystem = Depends(get_rag_system)
):
    """벡터 확인 (Dense + Sparse)"""
    try:
        # point_ids 파싱
        parsed_point_ids = None
        if point_ids:
            try:
                parsed_point_ids = [pid.strip() for pid in point_ids.split(',')]
            except Exception as e:
                logger = get_logger()
                logger.warning(f"point_ids 파싱 실패: {str(e)}, 무시하고 진행")
        
        vector_info = rag.vector_store.inspect_vectors(
            sample_size=sample_size,
            point_ids=parsed_point_ids
        )
        return vector_info
        
    except Exception as e:
        logger = get_logger()
        logger.error(f"벡터 확인 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"벡터 확인 실패: {str(e)}")


@app.post(
    "/test-vector-creation",
    tags=["시스템 관리"],
    summary="테스트 문장 벡터 생성 확인",
    description="""
    테스트용 문장을 업로드하여 Dense와 Sparse 벡터가 제대로 생성되는지 확인합니다.
    
    ## 사용 방법
    1. 간단한 문장을 입력
    2. 자동으로 처리되어 벡터 저장소에 저장
    3. 생성된 벡터 정보 반환
    
    ## 확인 항목
    - Dense 벡터 생성 여부 및 크기
    - Sparse 벡터 생성 여부 및 토큰 수
    - 저장된 포인트 ID
    """,
    response_description="벡터 생성 결과 및 확인 정보"
)
async def test_vector_creation(
    text: str = Form(..., description="테스트할 문장", min_length=1, max_length=1000),
    rag: RAGSystem = Depends(get_rag_system)
):
    """테스트 문장 벡터 생성 확인"""
    try:
        logger = get_logger()
        logger.info(f"테스트 벡터 생성 시작: {text[:50]}...")
        
        # 임시 파일 생성
        import tempfile
        import uuid
        from pathlib import Path
        
        test_id = str(uuid.uuid4())[:8]
        temp_dir = Path("data/temp_test")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        temp_file = temp_dir / f"test_{test_id}.txt"
        temp_file.write_text(text, encoding='utf-8')
        
        try:
            # 문서 처리
            from src.modules.document_processor import DocumentProcessor
            processor = DocumentProcessor()
            chunks = processor.process_file(str(temp_file))
            
            if not chunks:
                raise HTTPException(status_code=400, detail="문서 처리 결과가 없습니다.")
            
            # 벡터 저장소에 추가 (동기 메서드)
            import uuid
            success = rag.vector_store.add_documents(chunks)
            
            if not success:
                raise HTTPException(status_code=500, detail="벡터 저장소에 저장 실패")
            
            # chunk_id를 기반으로 point_id 생성 (add_documents와 동일한 로직)
            point_ids = []
            for chunk in chunks:
                if chunk.chunk_id:
                    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))
                else:
                    point_id = str(uuid.uuid4())
                point_ids.append(point_id)
            
            # 저장된 포인트의 벡터 확인
            vector_info = rag.vector_store.inspect_vectors(point_ids=point_ids)
            
            result = {
                'success': True,
                'test_text': text,
                'chunks_created': len(chunks),
                'chunk_ids': [chunk.chunk_id for chunk in chunks],
                'point_ids': point_ids,
                'vector_info': vector_info,
                'message': f'{len(chunks)}개 청크가 생성되어 벡터 저장소에 저장되었습니다.'
            }
            
            return result
            
        finally:
            # 임시 파일 삭제
            try:
                temp_file.unlink()
            except Exception:
                pass
        
    except HTTPException:
        raise
    except Exception as e:
        logger = get_logger()
        logger.error(f"테스트 벡터 생성 실패: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"테스트 벡터 생성 실패: {str(e)}")


@app.delete(
    "/documents/{document_id}",
    response_model=DocumentDeleteResponse,
    tags=["문서 관리"],
    summary="문서 삭제",
    description=    """
    벡터 저장소에서 특정 문서를 삭제합니다.
    
    ## 삭제 범위
    - **Qdrant**: 벡터 데이터 삭제 (Dense + Sparse 벡터)
    
    ## 파라미터
    - **document_id**: 삭제할 문서의 source_file 경로
    
    ## 주의사항
    - 문서가 존재하지 않으면 실패 응답을 반환합니다
    - 삭제된 청크 수가 응답에 포함됩니다
    - 삭제 후 문서 목록이 자동으로 업데이트됩니다
    """,
    response_description="삭제 성공 여부, 메시지, 삭제된 청크 수, Qdrant 삭제 상태"
)
async def delete_document(document_id: str, rag: RAGSystem = Depends(get_rag_system)):
    """특정 문서 삭제 (Qdrant만 사용, FAISS/BM25 제거됨)"""
    try:
        logger = get_logger()
        logger.info(f"문서 삭제 요청: {document_id}")
        
        # 입력 파라미터 검증
        if not isinstance(document_id, str) or not document_id.strip():
            raise HTTPException(status_code=400, detail="document_id는 비어있지 않은 문자열이어야 합니다.")
        
        if document_id == 'N/A' or document_id.strip() == 'N/A':
            raise HTTPException(
                status_code=400, 
                detail="'N/A'는 유효한 문서 ID가 아닙니다. 검색 결과에서 source_file이 없는 경우입니다."
            )
        
        # 문서 존재 여부 확인
        documents_info = rag.vector_store.get_documents_info()
        document_exists = any(doc.get('source_file') == document_id for doc in documents_info)
        
        if not document_exists:
            logger.warning(f"삭제할 문서가 없습니다: {document_id}")
            return DocumentDeleteResponse(
                success=False,
                message=f"문서를 찾을 수 없습니다: {document_id}",
                deleted_chunks_count=0
            )
        
        # 문서 삭제 실행
        result = rag.delete_document(document_id)
        
        # 반환값 검증
        if result is None:
            raise HTTPException(status_code=500, detail="문서 삭제 결과를 확인할 수 없습니다.")
        
        if not isinstance(result, dict):
            raise HTTPException(status_code=500, detail=f"delete_document 반환값이 딕셔너리가 아닙니다: {type(result)}")
        
        # 응답 생성 (Qdrant만 사용, FAISS/BM25 제거됨)
        qdrant_success = result.get('qdrant_success', result.get('qdrant_deleted', False))
        deleted_chunks_count = result.get('deleted_chunks_count', 0)
        
        if qdrant_success:
            message = f"문서 삭제 완료: {deleted_chunks_count}개 청크 삭제됨"
            if result.get('warnings'):
                message += f". 경고: {', '.join(result['warnings'])}"
            
            return DocumentDeleteResponse(
                success=True,
                message=message,
                deleted_chunks_count=deleted_chunks_count,
                qdrant_deleted=qdrant_success,
                warnings=result.get('warnings', [])
            )
        else:
            error_message = f"문서 삭제 실패: {', '.join(result.get('warnings', [])) if result.get('warnings') else '알 수 없는 오류'}"
            logger.error(error_message)
            return DocumentDeleteResponse(
                success=False,
                message=error_message,
                deleted_chunks_count=deleted_chunks_count,
                qdrant_deleted=False,
                warnings=result.get('warnings', [])
            )
            
    except Exception as e:
        logger = get_logger()
        logger.error(f"문서 삭제 중 오류 발생: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"문서 삭제 실패: {str(e)}")

# /rebuild-indexes ?붾뱶?ъ씤???쒓굅??(FAISS/BM25 ?쒓굅)


@app.post(
    "/clear-cache",
    tags=["인덱스 관리"],
    summary="캐시 초기화",
    description="""
    임베딩 캐시를 초기화합니다.
    
    ## 초기화 대상
    - 임베딩 캐시 (LRU 캐시)
    
    ## 사용 시기
    - 메모리 사용량 감소 필요 시
    - 임베딩 모델 변경 후
    - 캐시 관련 문제 발생 시
    
    ## 주의사항
    - 캐시 초기화 후 첫 요청은 캐시 미스로 인해 느릴 수 있습니다
    - 자주 사용되는 임베딩은 다시 캐시에 저장됩니다
    """,
    response_description="캐시 초기화 성공 여부 및 상태 메시지"
)
async def clear_cache(rag: RAGSystem = Depends(get_rag_system)):
    """캐시 초기화"""
    try:
        rag.clear_cache()
        return {"message": "캐시 초기화 완료", "status": "success"}
        
    except Exception as e:
        logger = get_logger()
        logger.error(f"캐시 초기화 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"캐시 초기화 실패: {str(e)}")


# ==================== 세션 관리 API ====================

@app.get(
    "/sessions",
    response_model=SessionListResponse,
    tags=["세션 관리"],
    summary="세션 목록 조회",
    description="""
    모든 세션 목록을 조회합니다.
    
    ## 포함 정보
    - **session_id**: 세션 ID
    - **title**: 세션 제목 (첫 사용자 메시지 또는 "새 대화")
    - **last_message**: 마지막 메시지 미리보기
    - **created_at**: 생성 시간
    - **last_accessed**: 마지막 접근 시간
    - **message_count**: 메시지 수
    
    ## 정렬
    - 최신 순으로 정렬 (last_accessed 기준 내림차순)
    
    ## 활용
    - 사이드바에 세션 목록 표시
    - 세션 선택 및 전환
    - 대화 히스토리 관리
    """,
    response_description="세션 목록 및 총 개수"
)
async def list_sessions():
    """세션 목록 조회"""
    try:
        session_manager = get_session_manager()
        if session_manager is None:
            raise HTTPException(status_code=503, detail="세션 관리자를 초기화할 수 없습니다.")
        
        sessions_data = session_manager.list_sessions()
        if sessions_data is None:
            sessions_data = []
        
        if not isinstance(sessions_data, list):
            logger = get_logger()
            logger.warning(f"세션 목록이 리스트가 아닙니다: {type(sessions_data)}, 빈 리스트로 변환")
            sessions_data = []
        
        # SessionInfo 리스트로 변환
        sessions = []
        for session_data in sessions_data:
            if not isinstance(session_data, dict):
                continue
            
            try:
                session_info = SessionInfo(
                    session_id=session_data.get("session_id", ""),
                    title=session_data.get("title", "새 대화"),
                    last_message=session_data.get("last_message", ""),
                    created_at=float(session_data.get("created_at", 0.0)),
                    last_accessed=float(session_data.get("last_accessed", 0.0)),
                    message_count=int(session_data.get("message_count", 0))
                )
                sessions.append(session_info)
            except Exception as e:
                logger = get_logger()
                logger.warning(f"세션 정보 변환 실패: {str(e)}, 건너뜀")
                continue
        
        return SessionListResponse(
            sessions=sessions,
            total_count=len(sessions)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger = get_logger()
        logger.error(f"세션 목록 조회 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"세션 목록 조회 실패: {str(e)}")


@app.post(
    "/sessions",
    response_model=SessionCreateResponse,
    tags=["세션 관리"],
    summary="세션 생성",
    description="""
    새로운 대화 세션을 생성합니다.
    
    ## 기능
    - 고유한 세션 ID 생성
    - 세션 메모리 캐시 초기화
    - 세션 TTL 설정
    
    ## 응답
    - **session_id**: 생성된 세션 ID (향후 요청에 사용)
    
    ## 활용
    - 멀티턴 대화 시작
    - 대화 히스토리 관리
    - 컨텍스트 유지
    """,
    response_description="생성된 세션 ID"
)
async def create_session(session_id: Optional[str] = Query(None, description="세션 ID (선택적, 없으면 자동 생성)")):
    """세션 생성"""
    try:
        session_manager = get_session_manager()
        if session_manager is None:
            raise HTTPException(status_code=503, detail="세션 관리자를 초기화할 수 없습니다.")
        
        session = session_manager.create_session(session_id)
        if session is None:
            raise HTTPException(status_code=500, detail="세션 생성에 실패했습니다.")
        
        if not hasattr(session, 'session_id') or session.session_id is None:
            raise HTTPException(status_code=500, detail="생성된 세션에 session_id가 없습니다.")
        
        return SessionCreateResponse(session_id=session.session_id)
    except HTTPException:
        raise
    except Exception as e:
        logger = get_logger()
        logger.error(f"세션 생성 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"세션 생성 실패: {str(e)}")


@app.get(
    "/sessions/{session_id}/history",
    response_model=SessionHistoryResponse,
    tags=["세션 관리"],
    summary="세션 히스토리 조회",
    description="""
    특정 세션의 대화 히스토리를 조회합니다.
    
    ## 파라미터
    - **session_id**: 조회할 세션 ID
    - **limit**: 반환할 최대 메시지 수 (선택적, 기본값: 전체)
    
    ## 포함 정보
    - 세션 ID
    - 메시지 히스토리 (역순, 최신 메시지가 마지막)
    - 각 메시지의 역할, 내용, 타임스탬프
    
    ## 활용
    - 대화 내용 확인
    - 컨텍스트 복원
    - 대화 분석
    """,
    response_description="세션 ID 및 메시지 히스토리"
)
async def get_session_history(session_id: str, limit: Optional[int] = None):
    """세션 히스토리 조회"""
    try:
        session_manager = get_session_manager()
        if session_manager is None:
            raise HTTPException(status_code=503, detail="세션 관리자를 초기화할 수 없습니다.")
        
        history = session_manager.get_history(session_id, limit=limit)
        if history is None:
            logger = get_logger()
            logger.warning(f"세션 히스토리가 None입니다: {session_id}, 빈 리스트로 반환")
            history = []
        
        if not isinstance(history, list):
            logger = get_logger()
            logger.warning(f"세션 히스토리가 리스트가 아닙니다: {type(history)}, 빈 리스트로 변환")
            history = []
        
        # Pydantic 검증을 위해 명시적으로 리스트로 변환
        history_list = list(history) if history else []
        
        return SessionHistoryResponse(session_id=session_id, history=history_list)
    except HTTPException:
        raise
    except Exception as e:
        logger = get_logger()
        logger.error(f"세션 히스토리 조회 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"세션 히스토리 조회 실패: {str(e)}")


@app.delete(
    "/sessions/{session_id}",
    tags=["세션 관리"],
    summary="세션 삭제",
    description="""
    특정 세션을 삭제합니다.
    
    ## 파라미터
    - **session_id**: 삭제할 세션 ID
    
    ## 삭제 범위
    - 세션 메모리 캐시
    - 세션 히스토리
    - 세션 관련 모든 데이터
    
    ## 활용
    - 세션 정리
    - 메모리 관리
    - 개인정보 보호
    """,
    response_description="삭제 성공 여부 및 메시지"
)
async def delete_session(session_id: str):
    """세션 삭제"""
    try:
        session_manager = get_session_manager()
        if session_manager is None:
            raise HTTPException(status_code=503, detail="세션 관리자를 초기화할 수 없습니다.")
        
        # 세션 존재 여부 확인
        session = session_manager.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"세션을 찾을 수 없습니다: {session_id}")
        
        result = session_manager.delete_session(session_id)
        if result:
            return {"message": f"세션 {session_id}가 삭제되었습니다.", "status": "success"}
        else:
            raise HTTPException(status_code=500, detail=f"세션 삭제에 실패했습니다: {session_id}")
    except HTTPException:
        raise
    except Exception as e:
        logger = get_logger()
        logger.error(f"세션 삭제 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"세션 삭제 실패: {str(e)}")


@app.get(
    "/sessions/stats",
    response_model=SessionStatsResponse,
    tags=["세션 관리"],
    summary="세션 통계 조회",
    description="""
    전체 세션 통계 정보를 조회합니다.
    
    ## 포함 정보
    - **total_sessions**: 총 세션 수
    - **total_memory_mb**: 전체 메모리 사용량 (MB)
    - **oldest_session_age**: 가장 오래된 세션 연령 (초)
    - **newest_session_age**: 가장 최근 세션 연령 (초)
    - **active_sessions**: 활성 세션 수
    
    ## 활용
    - 시스템 모니터링
    - 메모리 사용량 확인
    - 세션 관리 최적화
    """,
    response_description="세션 통계 정보"
)
async def get_session_stats():
    """세션 통계 조회"""
    try:
        session_manager = get_session_manager()
        if session_manager is None:
            raise HTTPException(status_code=503, detail="세션 관리자를 초기화할 수 없습니다.")
        
        stats = session_manager.get_stats()
        if stats is None:
            raise HTTPException(status_code=500, detail="세션 통계를 가져올 수 없습니다.")
        
        # SessionStats 필드 검증
        if not hasattr(stats, 'total_sessions'):
            raise HTTPException(status_code=500, detail="세션 통계에 total_sessions 필드가 없습니다.")
        if not hasattr(stats, 'total_memory_mb'):
            raise HTTPException(status_code=500, detail="세션 통계에 total_memory_mb 필드가 없습니다.")
        if not hasattr(stats, 'oldest_session_age'):
            raise HTTPException(status_code=500, detail="세션 통계에 oldest_session_age 필드가 없습니다.")
        if not hasattr(stats, 'newest_session_age'):
            raise HTTPException(status_code=500, detail="세션 통계에 newest_session_age 필드가 없습니다.")
        if not hasattr(stats, 'active_sessions'):
            raise HTTPException(status_code=500, detail="세션 통계에 active_sessions 필드가 없습니다.")
        
        return SessionStatsResponse(
            total_sessions=int(stats.total_sessions) if stats.total_sessions is not None else 0,
            total_memory_mb=float(stats.total_memory_mb) if stats.total_memory_mb is not None else 0.0,
            oldest_session_age=float(stats.oldest_session_age) if stats.oldest_session_age is not None else 0.0,
            newest_session_age=float(stats.newest_session_age) if stats.newest_session_age is not None else 0.0,
            active_sessions=int(stats.active_sessions) if stats.active_sessions is not None else 0
        )
    except HTTPException:
        raise
    except Exception as e:
        logger = get_logger()
        logger.error(f"세션 통계 조회 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"세션 통계 조회 실패: {str(e)}")


class ReloadModelRequest(BaseModel):
    """
    모델 재로드 요청 모델
    
    임베딩 또는 리랭커 모델을 동적으로 재로드하기 위한 요청입니다.
    """
    model_type: str = Field(
        ...,
        description="모델 타입: 'embedding' 또는 'reranker'",
        json_schema_extra={"example": "embedding"}
    )
    config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="새로운 모델 설정 (선택적). None이면 설정 파일의 기본 설정 사용",
        json_schema_extra={"example": {"model_path": "./models/new_embedding_model", "device": "cuda"}}
    )


class ReloadModelResponse(BaseModel):
    """
    모델 재로드 응답 모델
    
    모델 재로드 결과 및 이전/새 모델 정보를 포함합니다.
    """
    success: bool = Field(..., description="재로드 성공 여부", json_schema_extra={"example": True})
    message: str = Field(..., description="응답 메시지", json_schema_extra={"example": "임베딩 모델 재로드 완료"})
    model_type: str = Field(..., description="재로드된 모델 타입", json_schema_extra={"example": "embedding"})
    old_model: Optional[str] = Field(
        default=None,
        description="이전 모델 이름 또는 경로",
        json_schema_extra={"example": "BGE-m3-ko"}
    )
    new_model: Optional[str] = Field(
        default=None,
        description="새 모델 이름 또는 경로",
        json_schema_extra={"example": "new-embedding-model"}
    )


@app.post(
    "/reload-model",
    response_model=ReloadModelResponse,
    tags=["모델 관리"],
    summary="모델 동적 재로드",
    description="""
    임베딩 모델 또는 리랭커 모델을 동적으로 재로드합니다.
    
    ## 지원 모델 타입
    - **embedding**: 임베딩 모델 재로드
    - **reranker**: 리랭커 모델 재로드
    
    ## 재로드 옵션
    - 설정 파일의 기본 설정 사용 (config=None)
    - 커스텀 설정 사용 (config 제공)
    
    ## 활용
    - 모델 변경 시 서버 재시작 없이 모델 교체
    - 모델 성능 비교 및 테스트
    - A/B 테스트
    
    ## 주의사항
    - 모델 재로드 중에는 해당 모델을 사용하는 요청이 지연될 수 있습니다
    - GPU 메모리가 부족한 경우 실패할 수 있습니다
    - 이전 모델은 메모리에서 해제됩니다
    """,
    response_description="재로드 성공 여부, 메시지, 모델 타입, 이전/새 모델 정보"
)
async def reload_model(request: ReloadModelRequest, rag: RAGSystem = Depends(get_rag_system)):
    """모델 동적 재로드 (임베딩 또는 리랭커)"""
    try:
        logger = get_logger()
        model_type = request.model_type.lower()
        
        if model_type == "embedding":
            # 기존 모델 이름 저장
            old_model = None
            if hasattr(rag, 'embedding_manager') and rag.embedding_manager:
                old_model = getattr(rag.embedding_manager.client, 'model_name', None) if hasattr(rag.embedding_manager, 'client') else None
            
            # 모델 재로드
            success = rag.reload_embedding_model(request.config)
            
            if success:
                # 새 모델 이름 가져오기
                new_model = None
                if hasattr(rag, 'embedding_manager') and rag.embedding_manager:
                    new_model = getattr(rag.embedding_manager.client, 'model_name', None) if hasattr(rag.embedding_manager, 'client') else None
                
                return ReloadModelResponse(
                    success=True,
                    message="임베딩 모델 재로드 완료",
                    model_type="embedding",
                    old_model=old_model,
                    new_model=new_model
                )
            else:
                raise HTTPException(status_code=500, detail="임베딩 모델 재로드 실패")
        
        elif model_type == "reranker":
            # 기존 모델 경로 저장
            old_model = None
            if rag.reranker and hasattr(rag.reranker, 'model_path'):
                old_model = rag.reranker.model_path
            
            # 모델 재로드
            success = rag.reload_reranker(request.config)
            
            if success:
                # 새 모델 경로 가져오기
                new_model = None
                if rag.reranker and hasattr(rag.reranker, 'model_path'):
                    new_model = rag.reranker.model_path
                
                return ReloadModelResponse(
                    success=True,
                    message="리랭커 모델 재로드 완료",
                    model_type="reranker",
                    old_model=old_model,
                    new_model=new_model
                )
            else:
                raise HTTPException(status_code=500, detail="리랭커 모델 재로드 실패")
        
        else:
            raise HTTPException(
                status_code=400,
                detail=f"지원하지 않는 모델 타입: {model_type}. 'embedding' 또는 'reranker'만 사용 가능합니다."
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger = get_logger()
        logger.error(f"모델 재로드 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"모델 재로드 실패: {str(e)}")


@app.post(
    "/release-gpu-memory",
    tags=["모델 관리"],
    summary="GPU 메모리 해제",
    description="""
    GPU에 로드된 모델을 언로드하여 메모리를 해제합니다.
    
    ## 해제 대상
    - 임베딩 모델 (GPU에 로드된 경우)
    - 리랭커 모델 (GPU에 로드된 경우)
    
    ## 사용 시기
    - GPU 메모리 부족 시
    - 다른 작업을 위해 GPU 메모리 확보 필요 시
    - 모델 사용을 일시 중단할 때
    
    ## 주의사항
    - 모델 언로드 후 해당 모델을 사용하는 요청은 실패합니다
    - 모델을 다시 사용하려면 서버를 재시작하거나 `/reload-model` API를 사용해야 합니다
    - CPU 모드로 동작하는 모델은 영향을 받지 않습니다
    """,
    response_description="GPU 메모리 해제 성공 여부 및 상태 메시지"
)
async def release_gpu_memory(rag: RAGSystem = Depends(get_rag_system)):
    """GPU 메모리 해제 (모델 언로드)"""
    try:
        logger = get_logger()
        logger.info("GPU 메모리 해제 요청 받음")
        
        rag._release_gpu_memory()
        
        return {
            "message": "GPU 메모리 해제 완료",
            "status": "success",
            "note": "모델을 다시 사용하려면 서버를 재시작하거나 /reload-model API를 사용하세요"
        }
        
    except Exception as e:
        logger = get_logger()
        logger.error(f"GPU 메모리 해제 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"GPU 메모리 해제 실패: {str(e)}")


@app.post(
    "/upload-documents",
    response_model=DocumentUploadResponse,
    tags=["문서 관리"],
    summary="문서 업로드 및 처리",
    description="""
    파일을 업로드하고 자동으로 처리하여 벡터 저장소에 저장합니다.
    
    ## 지원 형식
    - Markdown (.md)
    - Word 문서 (.docx)
    - PDF (.pdf)
    - 텍스트 파일 (.txt)
    
    ## 처리 모드
    - **증분 업데이트 모드** (기본): 기존 문서가 있으면 업데이트, 없으면 추가
    - **교체 모드**: 같은 이름의 파일이 있으면 완전히 교체
    
    ## 자동 처리
    - 문서 파싱 및 청킹
    - 임베딩 생성
    - 벡터 저장소 저장 (Qdrant)
    - FAISS/BM25 인덱스 자동 업데이트 또는 재구축
    - EnsembleRetriever 재초기화
    
    ## 파라미터
    - **files**: 업로드할 파일 목록 (다중 파일 지원)
    - **force_update**: 강제 업데이트 여부
    - **replace_mode**: 교체 모드 활성화 여부
    """,
    response_description="처리 성공 여부, 메시지, 처리된 파일 수, 생성된 청크 수, 처리 시간"
)
async def upload_documents(
    files: List[UploadFile] = File(...),
    force_update: bool = Form(False),
    replace_mode: bool = Form(False),
    rag: RAGSystem = Depends(get_rag_system)
):
    """문서 업로드 및 처리"""
    start_time = time.time()
    
    try:
        logger = get_logger()
        logger.info(f"문서 업로드 시작: {len(files)}개 파일")
        
        # 업로드 디렉토리 생성
        upload_dir = Path("data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        processed_files = 0
        total_chunks = 0
        
        for file in files:
            if not file.filename:
                continue
                
            # 파일 저장
            file_path = upload_dir / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # 문서 처리
            chunks = rag.document_processor.process_file(file_path, force_process=True)
            if chunks:
                # 같은 이름의 파일이 이미 존재하는지 확인
                existing_documents = rag.vector_store.get_documents_info()
                
                # 상세 로그: 기존 문서 목록 출력
                logger.info(f"기존 문서 목록 ({len(existing_documents)}개):")
                for i, doc in enumerate(existing_documents):
                    logger.info(f"  [{i+1}] 파일명: '{doc.get('file_name', 'N/A')}', 경로: '{doc.get('source_file', 'N/A')}'")
                
                # 파일명 비교 (정확한 매칭)
                file_exists = any(doc.get('file_name') == file.filename for doc in existing_documents)
                
                # 상세 로그: 파일명 비교 결과
                logger.info(f"업로드 파일명: '{file.filename}'")
                logger.info(f"파일명 중복 검사 결과: {file_exists}")
                logger.info(f"교체 모드 체크박스: {replace_mode}")
                
                # 같은 이름의 파일이 있거나 교체 모드가 활성화된 경우 교체 모드 사용
                should_replace = replace_mode or file_exists
                logger.info(f"최종 교체 모드 결정: {should_replace} (교체모드체크박스: {replace_mode} OR 파일존재: {file_exists})")
                
                if should_replace:
                    success = rag.vector_store.replace_document_vectors(str(file_path), chunks)
                    if file_exists:
                        logger.info(f"같은 이름의 파일 발견, 교체 모드로 처리: {file.filename}")
                    else:
                        logger.info(f"교체 모드로 처리: {file.filename}")
                else:
                    success = rag.vector_store.add_documents(chunks, force_update)
                    logger.info(f"증분 업데이트 모드로 처리: {file.filename}")
                
                if success:
                    processed_files += 1
                    total_chunks += len(chunks)
                    logger.info(f"파일 처리 완료: {file.filename}, 청크 수: {len(chunks)}")
                else:
                    logger.error(f"벡터 저장소 저장 실패: {file.filename}")
            else:
                logger.warning(f"처리할 청크가 없음: {file.filename}")
        
        # 모든 파일 처리 완료 (Qdrant Dense+Sparse만 사용)
        if processed_files > 0:
            logger.info(f"문서 처리 완료: {processed_files}개 파일, {total_chunks}개 청크")
        
        processing_time = time.time() - start_time
        
        return DocumentUploadResponse(
            success=True,
            message=f"{processed_files}개 파일 처리 완료",
            processed_files=processed_files,
            total_chunks=total_chunks,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger = get_logger()
        logger.error(f"문서 업로드 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"문서 업로드 실패: {str(e)}")


@app.post(
    "/process-directory",
    tags=["문서 관리"],
    summary="디렉토리 일괄 처리",
    description="""
    지정된 디렉토리 내의 모든 문서를 일괄 처리합니다.
    
    ## 기능
    - 디렉토리 내 모든 지원 형식 파일 자동 탐지
    - 일괄 문서 처리 및 벡터 저장소 저장
    - 증분 업데이트 또는 교체 모드 지원
    
    ## 처리 모드
    - **증분 업데이트 모드** (기본): 기존 문서 업데이트 또는 추가
    - **교체 모드**: 기존 문서 완전 교체
    
    ## 파라미터
    - **directory_path**: 처리할 디렉토리 경로
    - **force_update**: 강제 업데이트 여부
    - **replace_mode**: 교체 모드 활성화 여부
    
    ## 활용
    - 대량 문서 일괄 처리
    - 정기적인 문서 업데이트
    - 초기 문서 로딩
    """,
    response_description="처리 성공 여부, 메시지, 처리 시간, 총 문서 수"
)
async def process_directory(
    directory_path: str = Form(...),
    force_update: bool = Form(False),
    replace_mode: bool = Form(False),
    rag: RAGSystem = Depends(get_rag_system)
):
    """디렉토리 내 문서 일괄 처리"""
    start_time = time.time()
    
    try:
        logger = get_logger()
        logger.info(f"디렉토리 처리 시작: {directory_path}")
        
        # 문서 처리 (비동기)
        # 입력 파라미터 검증
        if not isinstance(directory_path, str) or not directory_path.strip():
            raise HTTPException(status_code=400, detail="directory_path는 비어있지 않은 문자열이어야 합니다.")
        
        if not isinstance(force_update, bool):
            force_update = bool(force_update)
        
        if not isinstance(replace_mode, bool):
            replace_mode = bool(replace_mode)
        
        success = await rag.process_and_store_documents_async(directory_path, force_update, replace_mode)
        
        # 반환값 검증
        if success is None:
            raise HTTPException(status_code=500, detail="문서 처리 결과를 확인할 수 없습니다.")
        
        if not isinstance(success, bool):
            logger = get_logger()
            logger.warning(f"process_and_store_documents_async 반환값이 bool이 아닙니다: {type(success)}")
            success = bool(success)
        
        processing_time = time.time() - start_time
        
        if success:
            stats = rag.vector_store.get_collection_info()
            return {
                "success": True,
                "message": "디렉토리 처리 완료",
                "processing_time": processing_time,
                "total_documents": stats.get("points_count", 0)
            }
        else:
            return {
                "success": False,
                "message": "디렉토리 처리 실패",
                "processing_time": processing_time
            }
        
    except Exception as e:
        logger = get_logger()
        logger.error(f"디렉토리 처리 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"디렉토리 처리 실패: {str(e)}")


def get_local_ip():
    """로컬 IP 주소 자동 감지"""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            return local_ip
    except Exception:
        try:
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            return local_ip
        except Exception:
            return "127.0.0.1"

def run_server():
    """서버 실행"""
    import os
    
    config = get_api_config()
    
    # reload와 workers 동시 사용 방지
    # reload=True일 때는 workers를 1로 강제 설정 (uvicorn 제약)
    workers = config.workers
    if config.reload:
        if config.workers > 1:
            logger = get_logger()
            logger.warning(
                f"reload 모드에서는 workers를 1로 고정합니다. "
                f"설정된 workers={config.workers}는 무시됩니다. "
                f"멀티 워커를 사용하려면 reload=false로 설정하세요."
            )
        workers = 1
    else:
        # 프로덕션 모드: CPU 코어 수 기반으로 워커 수 최적화
        cpu_count = os.cpu_count() or 1
        if config.workers == 1:
            # 설정이 1이면 CPU 코어 수 기반으로 자동 설정 (최소 2, 최대 8)
            workers = max(2, min(cpu_count, 8))
            logger = get_logger()
            logger.info(
                f"워커 수가 1로 설정되어 있습니다. "
                f"CPU 코어 수({cpu_count})를 기반으로 {workers}개 워커로 자동 설정합니다."
            )
        else:
            workers = config.workers
    
    # 네트워크 정보 표시
    if config.host == "0.0.0.0":
        local_ip = get_local_ip()
        print(f"🌐 API 서버 시작 중...")
        print(f"📍 로컬 주소: http://localhost:{config.port}")
        print(f"🌍 네트워크 주소: http://{local_ip}:{config.port}")
        print(f"📚 API 문서: http://localhost:{config.port}/docs")
        print(f"🔗 웹 인터페이스: http://localhost:{config.port}/web_interface.html")
        print(f"⚙️  워커 수: {workers} ({'개발 모드 (reload)' if config.reload else '프로덕션 모드'})")
        print(f"⏹️  종료하려면 Ctrl+C를 누르세요")
        print(f"\n💡 다른 컴퓨터에서 접근:")
        print(f"   http://{local_ip}:{config.port}/web_interface.html")
    else:
        print(f"🌐 API 서버 시작 중...")
        print(f"📍 주소: http://{config.host}:{config.port}")
        print(f"⚙️  워커 수: {workers} ({'개발 모드 (reload)' if config.reload else '프로덕션 모드'})")
    
    # uvicorn 실행 파라미터 구성
    run_kwargs = {
        "app": "src.api.main:app",
        "host": config.host,
        "port": config.port,
        "reload": config.reload,
    }
    
    # reload=False일 때만 workers 파라미터 추가
    if not config.reload:
        run_kwargs["workers"] = workers
    
    uvicorn.run(**run_kwargs)


if __name__ == "__main__":
    run_server()
