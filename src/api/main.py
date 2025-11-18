"""
FastAPI 기반 REST API 서버
"""

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
import time
import uvicorn
import shutil
from pathlib import Path

from src.utils.logger import setup_logging, get_logger
from src.utils.config import get_config, get_api_config
from src.modules.rag_system import RAGSystem, RAGResponse


class RebuildIndexesResponse(BaseModel):
    """인덱스 재구축 응답"""
    success: bool
    message: str
    chunks_count: Optional[int] = None


# 요청/응답 모델
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
        example="변압기 진단 기준은 무엇인가요?"
    )
    max_sources: int = Field(
        default=5,
        description="검색 결과로 반환할 최대 소스 문서 수 (1-20)",
        ge=1,
        le=20,
        example=5
    )
    score_threshold: float = Field(
        default=0.7,
        description="유사도 점수 임계값 (0.0-1.0). 이 값보다 낮은 점수의 문서는 제외됩니다.",
        ge=0.0,
        le=1.0,
        example=0.7
    )
    temperature: Optional[float] = Field(
        default=None,
        description="LLM 생성 온도 (0.0-2.0). 높을수록 창의적인 답변, 낮을수록 일관된 답변. None이면 설정 파일 값 사용",
        ge=0.0,
        le=2.0,
        example=0.1
    )
    model: Optional[str] = Field(
        default=None,
        description="사용할 LLM 모델명 (예: 'gemma3:4b', 'llama3.1:8b'). None이면 설정 파일의 기본 모델 사용",
        example="gemma3:4b"
    )
    use_qdrant: bool = Field(
        default=True,
        description="Qdrant 벡터 검색 사용 여부. Qdrant와 FAISS는 동시에 선택할 수 없습니다.",
        example=True
    )
    use_faiss: bool = Field(
        default=False,
        description="FAISS 벡터 검색 사용 여부 (GPU 가속 지원). Qdrant와 FAISS는 동시에 선택할 수 없습니다.",
        example=False
    )
    use_bm25: bool = Field(
        default=False,
        description="BM25 키워드 검색 사용 여부 (숨김 처리됨 - Qdrant만 사용).",
        example=False
    )
    use_reranker: bool = Field(
        default=True,
        description="리랭커 사용 여부. CrossEncoder를 사용하여 검색 결과를 재정렬합니다.",
        example=True
    )
    reranker_alpha: float = Field(
        default=0.7,
        description="리랭커 점수 가중치 (0.0-1.0). 1.0에 가까울수록 리랭커 점수를 더 많이 반영합니다.",
        ge=0.0,
        le=1.0,
        example=0.7
    )
    reranker_top_k: Optional[int] = Field(
        default=None,
        description="리랭커에 전달할 상위 K 후보 수 (1-50). None이면 max_sources 값 사용",
        ge=1,
        le=50,
        example=10
    )
    weights: Optional[Dict[str, float]] = Field(
        default=None,
        description="검색기 가중치 딕셔너리. 예: {'qdrant': 0.7, 'faiss': 0.0, 'bm25': 0.3}. None이면 자동 계산",
        example={"qdrant": 0.7, "faiss": 0.0, "bm25": 0.3}
    )
    dense_weight: Optional[float] = Field(
        default=None,
        description="Qdrant 하이브리드 검색의 Dense 벡터 가중치 (0.0-1.0). None이면 config.yaml의 기본값 사용. Sparse 벡터가 활성화되어 있을 때만 의미가 있습니다.",
        ge=0.0,
        le=1.0,
        example=0.7
    )
    sparse_weight: Optional[float] = Field(
        default=None,
        description="Qdrant 하이브리드 검색의 Sparse 벡터 가중치 (0.0-1.0). None이면 config.yaml의 기본값 사용. Sparse 벡터가 활성화되어 있을 때만 의미가 있습니다.",
        ge=0.0,
        le=1.0,
        example=0.3
    )
    
    @validator('use_bm25', always=True)
    def validate_at_least_one_retriever(cls, v, values):
        """Qdrant만 사용 (FAISS, BM25는 숨김 처리됨)"""
        use_qdrant = values.get('use_qdrant', True)
        use_faiss = values.get('use_faiss', False)
        use_bm25 = v
        
        # Qdrant만 사용하도록 강제
        if not use_qdrant:
            raise ValueError('Qdrant 검색기는 필수입니다. (FAISS, BM25는 현재 사용할 수 없습니다)')
        
        # Qdrant와 FAISS는 배타적 선택 (FAISS는 숨김 처리되었지만 검증은 유지)
        if use_qdrant and use_faiss:
            raise ValueError('Qdrant와 FAISS는 동시에 선택할 수 없습니다. Qdrant만 사용하세요.')
        
        return v

    @validator('weights', always=True)
    def normalize_weights(cls, v, values):
        """가중치 정규화: Qdrant/FAISS 배타, BM25 독립. BM25 선택 시 (vector+bm25)=1."""
        if v is None:
            # 기본값: BM25 미선택이면 선택된 벡터=1, BM25 선택이면 vector/bm25 균등
            use_q = values.get('use_qdrant', True)
            use_f = values.get('use_faiss', False)
            use_b = values.get('use_bm25', True)
            if use_q and not use_f:
                if use_b:
                    return {'qdrant': 0.7, 'faiss': 0.0, 'bm25': 0.3}
                return {'qdrant': 1.0, 'faiss': 0.0, 'bm25': 0.0}
            if use_f and not use_q:
                if use_b:
                    return {'qdrant': 0.0, 'faiss': 0.5, 'bm25': 0.5}
                return {'qdrant': 0.0, 'faiss': 1.0, 'bm25': 0.0}
            if use_b:
                return {'qdrant': 0.0, 'faiss': 0.0, 'bm25': 1.0}
            return {'qdrant': 1.0, 'faiss': 0.0, 'bm25': 0.0}
        # 입력 가중치 정리
        wq = float(v.get('qdrant', 0.0))
        wf = float(v.get('faiss', 0.0))
        wb = float(v.get('bm25', 0.0))
        # 선택되지 않은 검색기는 0 강제, 배타성 반영
        use_q = values.get('use_qdrant', True)
        use_f = values.get('use_faiss', False)
        use_b = values.get('use_bm25', False)
        if not use_q:
            wq = 0.0
        if not use_f:
            wf = 0.0
        if not use_b:
            wb = 0.0
        # 배타 규칙: 둘 다 true인 상태는 이전 validator에서 차단
        if use_b:
            # BM25 포함: (vector + bm25) = 1
            vector = wq if use_q else (wf if use_f else 0.0)
            total = vector + wb
            if total <= 0:
                if use_q:
                    return {'qdrant': 0.5, 'faiss': 0.0, 'bm25': 0.5}
                if use_f:
                    return {'qdrant': 0.0, 'faiss': 0.5, 'bm25': 0.5}
                return {'qdrant': 0.0, 'faiss': 0.0, 'bm25': 1.0}
            q = (vector / total) if use_q else 0.0
            f = (vector / total) if use_f else 0.0
            b = wb / total
            return {'qdrant': q, 'faiss': f, 'bm25': b}
        # BM25 미포함: 선택된 벡터=1
        if use_q:
            return {'qdrant': 1.0, 'faiss': 0.0, 'bm25': 0.0}
        if use_f:
            return {'qdrant': 0.0, 'faiss': 1.0, 'bm25': 0.0}
        return {'qdrant': 0.0, 'faiss': 0.0, 'bm25': 1.0}


class QueryResponse(BaseModel):
    """
    RAG 질의응답 결과 모델
    
    생성된 답변과 출처 정보를 포함합니다.
    """
    answer: str = Field(..., description="생성된 답변 텍스트", example="변압기 진단 기준은 다음과 같습니다...")
    sources: List[Dict[str, Any]] = Field(
        ...,
        description="답변 생성에 사용된 출처 문서 목록. 각 문서는 content, score, metadata 등을 포함합니다.",
        example=[
            {
                "content": "변압기 진단 기준...",
                "score": 0.95,
                "source_file": "data/raw/transformer_guide.md",
                "chunk_index": 0
            }
        ]
    )
    confidence: float = Field(
        ...,
        description="답변의 신뢰도 점수 (0.0-1.0). 검색 결과의 평균 점수를 기반으로 계산됩니다.",
        ge=0.0,
        le=1.0,
        example=0.92
    )
    processing_time: float = Field(
        ...,
        description="전체 처리 시간 (초). 검색 + 리랭킹 + 답변 생성 시간을 포함합니다.",
        example=3.45
    )
    query: str = Field(..., description="원본 질문", example="변압기 진단 기준은 무엇인가요?")
    model_used: str = Field(..., description="사용된 LLM 모델명", example="gemma3:4b")
    warnings: Optional[List[str]] = Field(
        default_factory=list,
        description="경고 메시지 목록. 예: 리랭커 요청했지만 사용 불가능한 경우",
        example=["리랭커 사용이 요청되었지만 리랭커가 초기화되지 않았습니다."]
    )


class AgenticExecuteRequest(BaseModel):
    """
    Agentic RAG 실행 요청 모델
    
    LangGraph 기반 워크플로우 실행을 위한 요청입니다.
    """
    question: str = Field(..., description="사용자 질문", example="변압기 진단 기준은 무엇인가요?")
    session_id: Optional[str] = Field(
        default=None,
        description="세션 ID (선택적). 세션 관리를 위해 사용됩니다.",
        example="session-12345"
    )
    graph: str = Field(
        default="basic",
        description="실행할 그래프 타입. 현재는 'basic'만 지원됩니다.",
        example="basic"
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="추가 파라미터. 검색 옵션 등을 포함할 수 있습니다.",
        example={"max_sources": 5, "use_reranker": True}
    )


class AgenticExecuteResponse(BaseModel):
    """
    Agentic RAG 실행 결과 모델
    
    LangGraph 워크플로우 실행 결과를 포함합니다.
    """
    answer: str = Field(..., description="생성된 답변 텍스트", example="변압기 진단 기준은 다음과 같습니다...")
    sources: List[Dict[str, Any]] = Field(
        ...,
        description="검색 결과 문서 목록",
        example=[{"content": "...", "score": 0.95, "source_file": "..."}]
    )
    confidence: float = Field(
        ...,
        description="답변의 신뢰도 점수 (0.0-1.0)",
        ge=0.0,
        le=1.0,
        example=0.92
    )
    processing_time: float = Field(..., description="전체 처리 시간 (초)", example=4.23)
    graph_run_id: str = Field(..., description="그래프 실행 ID (고유 식별자)", example="basic-1234567890")


class HealthResponse(BaseModel):
    """헬스 체크 응답"""
    status: str = Field(..., description="서버 상태", example="healthy")
    timestamp: float = Field(..., description="응답 생성 시각 (Unix timestamp)", example=1234567890.123)
    version: str = Field(..., description="API 버전", example="1.0.0")


class StatsResponse(BaseModel):
    """
    시스템 통계 응답 모델
    
    시스템의 현재 상태 및 통계 정보를 포함합니다.
    """
    embedding_cache_stats: Dict[str, Any] = Field(
        ...,
        description="임베딩 캐시 통계 (캐시 크기, 히트/미스율, 모델 정보 등)",
        example={"cache_size": 1000, "hits": 500, "misses": 200, "model_name": "BGE-m3-ko", "dimension": 1024}
    )
    vector_store_stats: Dict[str, Any] = Field(
        ...,
        description="벡터 저장소 통계 (총 문서 수, 청크 수, 컬렉션 정보 등)",
        example={"total_documents": 50, "total_chunks": 500, "collection_name": "electrical_diagnosis"}
    )
    llm_model: str = Field(..., description="현재 사용 중인 LLM 모델명", example="gemma3:4b")


class ModelsResponse(BaseModel):
    """
    사용 가능한 모델 목록 응답 모델
    
    Ollama 서버에서 사용 가능한 모델 목록을 포함합니다.
    """
    available_models: List[Dict[str, str]] = Field(
        ...,
        description="사용 가능한 모델 목록. 각 모델은 name, size, modified_at, family 정보를 포함합니다.",
        example=[
            {"name": "gemma3:4b", "size": "2.5GB", "modified_at": "2024-01-01T00:00:00Z", "family": "gemma"},
            {"name": "llama3.1:8b", "size": "4.7GB", "modified_at": "2024-01-01T00:00:00Z", "family": "llama"}
        ]
    )
    current_model: str = Field(..., description="현재 사용 중인 모델명", example="gemma3:4b")


class ConfigResponse(BaseModel):
    """
    시스템 설정 응답 모델
    
    현재 시스템 설정 정보를 포함합니다.
    """
    llm_model: str = Field(..., description="현재 사용 중인 LLM 모델명", example="gemma3:4b")
    embedding_model: str = Field(..., description="현재 사용 중인 임베딩 모델명", example="BGE-m3-ko")
    max_sources: int = Field(..., description="기본 최대 소스 수", example=5)
    score_threshold: float = Field(..., description="기본 점수 임계값", example=0.7)
    default_limit: int = Field(..., description="기본 검색 제한값", example=5)
    reranker_model: Optional[str] = Field(
        default=None,
        description="리랭커 모델 경로 또는 이름",
        example="ms-marco-MiniLM-L-6-v2"
    )
    reranker_enabled: bool = Field(
        default=False,
        description="리랭커 활성화 여부",
        example=True
    )
    reranker_alpha: Optional[float] = Field(
        default=None,
        description="리랭커 alpha 값 (0.0-1.0)",
        example=0.7
    )


class DocumentUploadResponse(BaseModel):
    """
    문서 업로드 응답 모델
    
    문서 업로드 및 처리 결과를 포함합니다.
    """
    success: bool = Field(..., description="업로드 성공 여부", example=True)
    message: str = Field(..., description="응답 메시지", example="3개 파일 처리 완료")
    processed_files: int = Field(..., description="처리된 파일 수", example=3)
    total_chunks: int = Field(..., description="생성된 청크 수", example=150)
    processing_time: float = Field(..., description="처리 시간(초)", example=12.34)


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
    redoc_url="/redoc"
)

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 RAG 시스템 인스턴스
rag_system: Optional[RAGSystem] = None


def get_rag_system() -> RAGSystem:
    """RAG 시스템 의존성"""
    global rag_system
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG 시스템이 초기화되지 않았습니다")
    return rag_system


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 실행"""
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
        if not rag_system.vector_store_manager.setup_collection():
            logger.error("벡터 저장소 설정 실패")
            raise Exception("벡터 저장소 설정 실패")
        
        logger.info("API 서버 시작 완료")
        
    except Exception as e:
        logger.error(f"서버 시작 실패: {str(e)}")
        raise


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
        # Qdrant만 사용 (FAISS, BM25는 숨김 처리됨)
        if not request.use_qdrant:
            raise HTTPException(
                status_code=400, 
                detail="Qdrant 검색기는 필수입니다. (FAISS, BM25는 현재 사용할 수 없습니다)"
            )
        
        # 검색기 선택 정보 구성
        retrievers = {
            'use_qdrant': request.use_qdrant,
            'use_faiss': request.use_faiss,
            'use_bm25': request.use_bm25,
            'use_reranker': request.use_reranker,
            'reranker_alpha': request.reranker_alpha,
            'reranker_top_k': request.reranker_top_k,
            'weights': request.weights or {'qdrant': 1.0, 'faiss': 0.0, 'bm25': 0.0},
            'dense_weight': request.dense_weight,
            'sparse_weight': request.sparse_weight
        }
        
        # RAG 시스템을 통한 질의 처리 (비동기 메서드 사용)
        response = await rag.query_async(
            question=request.question,
            max_sources=request.max_sources,
            score_threshold=request.score_threshold,
            model_name=request.model,
            retrievers=retrievers
        )
        
        # 리랭커 요청했지만 사용 불가능한 경우 경고 추가
        reranker_requested = request.use_reranker
        reranker_available = rag.reranker is not None
        warnings = []
        if reranker_requested and not reranker_available:
            warnings.append("리랭커 사용이 요청되었지만 리랭커가 초기화되지 않았습니다. config.yaml에서 reranker.enabled: true로 설정하세요.")
        
        return QueryResponse(
            answer=response.answer,
            sources=response.sources,
            confidence=response.confidence,
            processing_time=response.processing_time,
            query=response.query,
            model_used=response.model_used,
            warnings=warnings
        )
        
    except Exception as e:
        logger = get_logger()
        logger.error(f"질의 처리 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"질의 처리 실패: {str(e)}")


@app.post(
    "/agentic/execute",
    response_model=AgenticExecuteResponse,
    tags=["질의응답"],
    summary="Agentic RAG 실행 (LangGraph)",
    description="""
    LangGraph 기반 Agentic RAG 워크플로우를 실행합니다.
    
    ## 기능
    - 질문 분석 → 검색 실행 → 답변 생성의 단계별 처리
    - 세션 지원 (선택적)
    - 동적 파라미터 설정
    
    ## 활성화 조건
    - `config.yaml`의 `feature_flags.langgraph_enabled`가 `true`여야 합니다
    - LangGraph 모듈이 설치되어 있어야 합니다
    
    ## 워크플로우
    1. **질문 분석**: 사용자 질문의 의도 파악
    2. **검색 실행**: 다중 검색 전략을 통한 관련 문서 검색
    3. **답변 생성**: 검색된 문서를 기반으로 최종 답변 생성
    """,
    response_description="생성된 답변, 검색 결과, 신뢰도, 처리 시간, 그래프 실행 ID"
)
async def agentic_execute(request: AgenticExecuteRequest):
    """Execute basic LangGraph workflow behind feature flag."""
    try:
        config = get_config()
        feature_flags = getattr(config, 'feature_flags', None) or {}
        if not feature_flags or not feature_flags.get('langgraph_enabled', False):
            raise HTTPException(status_code=404, detail="Agentic API is disabled (feature flag: langgraph_enabled=false)")

        try:
            from src.agentic_rag.workflows.basic_graph import run_basic_graph  # type: ignore
        except Exception:
            raise HTTPException(status_code=501, detail="LangGraph is not available")

        start = time.time()
        out = run_basic_graph(
            question=request.question,
            session_id=request.session_id,
            parameters=request.parameters or {},
        )
        elapsed = time.time() - start

        graph_run_id = out.get('graph_run_id', f"basic-{int(time.time()*1000)}")
        
        return AgenticExecuteResponse(
            answer=str(out.get('answer', '')),
            sources=list(out.get('search_results', []) or []),
            confidence=float(out.get('confidence', 0.0) or 0.0),
            processing_time=float(out.get('processing_time', elapsed) or elapsed),
            graph_run_id=str(graph_run_id),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger = get_logger()
        logger.error(f"/agentic/execute 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"agentic execute failed: {str(e)}")


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
        success = rag.process_and_store_documents(input_dir)
        
        if success:
            return {"message": "문서 처리 완료", "status": "success"}
        else:
            raise HTTPException(status_code=500, detail="문서 처리 실패")
            
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
    total_documents: int = Field(..., description="총 문서 수", example=50)
    documents: List[Dict[str, Any]] = Field(
        ...,
        description="문서 목록. 각 문서는 파일명, 경로, 청크 수, 메타데이터 등을 포함합니다.",
        example=[
            {
                "file_name": "transformer_guide.md",
                "source_file": "data/raw/transformer_guide.md",
                "chunk_count": 10,
                "metadata": {}
            }
        ]
    )
    total_chunks: int = Field(..., description="총 청크 수", example=500)
    collection_info: Dict[str, Any] = Field(
        ...,
        description="컬렉션 정보 (컬렉션명, 포인트 수 등)",
        example={"collection_name": "electrical_diagnosis", "points_count": 500}
    )


class DocumentDeleteRequest(BaseModel):
    """문서 삭제 요청"""
    source_file: str = Field(..., description="삭제할 문서의 source_file 경로")


class DocumentDeleteResponse(BaseModel):
    """
    문서 삭제 응답 모델
    
    문서 삭제 결과 및 각 저장소별 삭제 상태를 포함합니다.
    """
    success: bool = Field(..., description="삭제 성공 여부", example=True)
    message: str = Field(..., description="응답 메시지", example="문서 삭제 완료: 10개 청크 삭제됨")
    deleted_chunks_count: int = Field(default=0, description="삭제된 청크 수", example=10)
    qdrant_deleted: bool = Field(default=False, description="Qdrant 삭제 성공 여부", example=True)
    faiss_deleted: bool = Field(default=False, description="FAISS 삭제 성공 여부 (레거시)", example=False)
    faiss_handled: bool = Field(
        default=False,
        description="FAISS 처리 여부. True이면 재구축이 필요할 수 있습니다.",
        example=True
    )
    bm25_deleted: bool = Field(default=False, description="BM25 삭제 성공 여부", example=True)
    warnings: List[str] = Field(
        default_factory=list,
        description="경고 메시지 목록",
        example=["FAISS 인덱스 재구축이 필요합니다."]
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
        collection_info = rag.vector_store_manager.get_collection_info()
        
        # 저장된 문서들의 메타데이터 조회
        documents_info = rag.vector_store_manager.get_documents_info()
        
        return DocumentInfoResponse(
            total_documents=len(documents_info),
            documents=documents_info,
            total_chunks=collection_info.get('points_count', 0),
            collection_info=collection_info
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
        chunks = rag.vector_store_manager.get_document_chunks(document_id)
        return {
            "document_id": document_id,
            "total_chunks": len(chunks),
            "chunks": chunks
        }
        
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
        vocabulary_info = rag.vector_store_manager.get_sparse_vocabulary(limit, search_token)
        return vocabulary_info
        
    except Exception as e:
        logger = get_logger()
        logger.error(f"Vocabulary 조회 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Vocabulary 조회 실패: {str(e)}")


@app.delete(
    "/documents/{document_id}",
    response_model=DocumentDeleteResponse,
    tags=["문서 관리"],
    summary="문서 삭제",
    description="""
    벡터 저장소에서 특정 문서를 삭제합니다.
    
    ## 삭제 범위
    - **Qdrant**: 벡터 데이터 삭제
    - **BM25**: BM25 인덱스에서 문서 제거
    - **FAISS**: FAISS 인덱스 재구축 필요 (자동 처리)
    
    ## 파라미터
    - **document_id**: 삭제할 문서의 source_file 경로
    
    ## 주의사항
    - 문서가 존재하지 않으면 실패 응답을 반환합니다
    - FAISS 인덱스는 삭제 후 재구축이 필요할 수 있습니다
    - 삭제된 청크 수가 응답에 포함됩니다
    """,
    response_description="삭제 성공 여부, 메시지, 삭제된 청크 수, 각 저장소별 삭제 상태"
)
async def delete_document(document_id: str, rag: RAGSystem = Depends(get_rag_system)):
    """특정 문서 삭제 (Qdrant + BM25 + FAISS)"""
    try:
        logger = get_logger()
        logger.info(f"문서 삭제 요청: {document_id}")
        
        # 문서 존재 여부 확인
        documents_info = rag.vector_store_manager.get_documents_info()
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
        
        # 응답 생성
        qdrant_success = result.get('qdrant_success', result.get('qdrant_deleted', False))
        bm25_success = result.get('bm25_success', result.get('bm25_deleted', False))
        faiss_handled = result.get('faiss_handled', False)
        
        if qdrant_success:
            message = f"문서 삭제 완료: {result.get('deleted_chunks_count', 0)}개 청크 삭제됨"
            if result.get('warnings'):
                message += f". 경고: {', '.join(result['warnings'])}"
            
            return DocumentDeleteResponse(
                success=True,
                message=message,
                deleted_chunks_count=result.get('deleted_chunks_count', 0),
                qdrant_deleted=qdrant_success,
                bm25_deleted=bm25_success,
                faiss_handled=faiss_handled,
                warnings=result.get('warnings', [])
            )
        else:
            error_message = f"문서 삭제 실패: {', '.join(result.get('warnings', [])) if result.get('warnings') else '알 수 없는 오류'}"
            logger.error(error_message)
            return DocumentDeleteResponse(
                success=False,
                message=error_message,
                deleted_chunks_count=result.get('deleted_chunks_count', 0),
                qdrant_deleted=False,
                bm25_deleted=bm25_success,
                faiss_handled=faiss_handled,
                warnings=result.get('warnings', [])
            )
            
    except Exception as e:
        logger = get_logger()
        logger.error(f"문서 삭제 중 오류 발생: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"문서 삭제 실패: {str(e)}")


@app.post(
    "/rebuild-indexes",
    response_model=RebuildIndexesResponse,
    tags=["인덱스 관리"],
    summary="인덱스 재구축",
    description="""
    FAISS 및 BM25 인덱스를 Qdrant의 기존 문서를 기반으로 재구축합니다.
    
    ## 재구축 대상
    - **FAISS 인덱스**: 벡터 검색 인덱스 재구축
    - **BM25 인덱스**: 키워드 검색 인덱스 재구축
    
    ## 사용 시기
    - 문서 삭제 후 인덱스 동기화
    - 인덱스 손상 또는 불일치 발생 시
    - 대량 문서 업데이트 후 일관성 확보
    - 하이브리드 검색 정확도 향상을 위해
    
    ## 주의사항
    - 재구축에는 시간이 소요될 수 있습니다 (문서 수에 비례)
    - 재구축 중에는 검색 성능이 일시적으로 저하될 수 있습니다
    - Qdrant에 저장된 모든 문서를 기반으로 재구축됩니다
    """,
    response_description="재구축 성공 여부, 메시지, 재구축된 청크 수"
)
async def rebuild_indexes(rag: RAGSystem = Depends(get_rag_system)):
    """FAISS 및 BM25 인덱스 재구축 (Qdrant의 기존 문서 사용)"""
    try:
        logger = get_logger()
        logger.info("인덱스 재구축 요청 받음")
        
        success = await rag.rebuild_faiss_and_bm25_indexes_async()
        
        if success:
            # 재구축된 청크 수 가져오기
            stats = rag.get_system_stats()
            vector_stats = stats.get('vector_store', {})
            chunks_count = vector_stats.get('total_chunks', None)
            
            return RebuildIndexesResponse(
                success=True,
                message="인덱스 재구축이 완료되었습니다.",
                chunks_count=chunks_count
            )
        else:
            return RebuildIndexesResponse(
                success=False,
                message="인덱스 재구축에 실패했습니다. 로그를 확인해주세요."
            )
            
    except Exception as e:
        logger = get_logger()
        logger.error(f"인덱스 재구축 실패: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"인덱스 재구축 실패: {str(e)}"
        )


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


class ReloadModelRequest(BaseModel):
    """
    모델 재로드 요청 모델
    
    임베딩 또는 리랭커 모델을 동적으로 재로드하기 위한 요청입니다.
    """
    model_type: str = Field(
        ...,
        description="모델 타입: 'embedding' 또는 'reranker'",
        example="embedding"
    )
    config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="새로운 모델 설정 (선택적). None이면 설정 파일의 기본 설정 사용",
        example={"model_path": "./models/new_embedding_model", "device": "cuda"}
    )


class ReloadModelResponse(BaseModel):
    """
    모델 재로드 응답 모델
    
    모델 재로드 결과 및 이전/새 모델 정보를 포함합니다.
    """
    success: bool = Field(..., description="재로드 성공 여부", example=True)
    message: str = Field(..., description="응답 메시지", example="임베딩 모델 재로드 완료")
    model_type: str = Field(..., description="재로드된 모델 타입", example="embedding")
    old_model: Optional[str] = Field(
        default=None,
        description="이전 모델 이름 또는 경로",
        example="BGE-m3-ko"
    )
    new_model: Optional[str] = Field(
        default=None,
        description="새 모델 이름 또는 경로",
        example="new-embedding-model"
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
                existing_documents = rag.vector_store_manager.get_documents_info()
                
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
                    success = rag.vector_store_manager.replace_document_vectors(str(file_path), chunks)
                    if file_exists:
                        logger.info(f"같은 이름의 파일 발견, 교체 모드로 처리: {file.filename}")
                    else:
                        logger.info(f"교체 모드로 처리: {file.filename}")
                else:
                    success = rag.vector_store_manager.add_chunks(chunks, force_update)
                    logger.info(f"증분 업데이트 모드로 처리: {file.filename}")
                
                if success:
                    processed_files += 1
                    total_chunks += len(chunks)
                    logger.info(f"파일 처리 완료: {file.filename}, 청크 수: {len(chunks)}")
                else:
                    logger.error(f"벡터 저장소 저장 실패: {file.filename}")
            else:
                logger.warning(f"처리할 청크가 없음: {file.filename}")
        
        # 모든 파일 처리 후 FAISS 및 BM25 인덱스 확인, 검증 및 EnsembleRetriever 재초기화
        if processed_files > 0:
            logger.info("FAISS 및 BM25 인덱스 상태 확인 및 검증 중...")
            
            # 현재 FAISS 인덱스 상태 확인
            langchain_manager = rag.vector_store_manager.langchain_retrieval_manager
            if langchain_manager:
                # FAISS 인덱스가 없으면 전체 문서에서 재구축
                if langchain_manager.faiss_store is None:
                    logger.info("FAISS 인덱스가 없습니다. 재구축을 시작합니다...")
                    rebuild_success = await rag.rebuild_faiss_and_bm25_indexes_async()
                    if rebuild_success:
                        logger.info("FAISS 및 BM25 인덱스 재구축 완료")
                    else:
                        logger.warning("FAISS 및 BM25 인덱스 재구축 실패")
                # BM25Retriever가 없으면 구축
                elif langchain_manager.bm25_retriever is None:
                    logger.info("BM25 인덱스가 없습니다. 전체 문서에서 구축합니다...")
                    # Qdrant에서 모든 청크 가져오기
                    all_chunks_data = []
                    documents_info = rag.vector_store_manager.get_documents_info()
                    for doc_info in documents_info:
                        source_file = doc_info.get('source_file', '')
                        if source_file:
                            chunks_data = rag.vector_store_manager.get_document_chunks(source_file)
                            all_chunks_data.extend(chunks_data)
                    
                    if all_chunks_data:
                        # DocumentChunk로 변환
                        from src.modules.document_processor import DocumentChunk
                        chunks = []
                        for chunk_data in all_chunks_data:
                            content = chunk_data.get('content_full', '')
                            if not content:
                                content = chunk_data.get('content_preview', '')
                            
                            metadata = chunk_data.get('metadata', {})
                            source_file = (
                                metadata.get('source_file') or 
                                metadata.get('file_path') or
                                chunk_data.get('source_file', '')
                            )
                            
                            chunk = DocumentChunk(
                                content=content,
                                metadata=metadata,
                                chunk_id=chunk_data.get('chunk_id', ''),
                                source_file=source_file,
                                chunk_index=chunk_data.get('chunk_index', 0)
                            )
                            chunks.append(chunk)
                        
                        # BM25 구축
                        bm25_success = langchain_manager.initialize_bm25_from_chunks(chunks)
                        if bm25_success:
                            logger.info("BM25 인덱스 구축 완료")
                        else:
                            logger.warning("BM25 인덱스 구축 실패")
                else:
                    logger.info("FAISS 및 BM25 인덱스가 이미 존재합니다.")
                
                # 인덱스 일관성 검증
                qdrant_stats = rag.vector_store_manager.get_stats()
                qdrant_doc_count = qdrant_stats.get('points_count', None)
                
                validation_result = langchain_manager.validate_indexes(qdrant_doc_count)
                
                if validation_result['warnings']:
                    for warning in validation_result['warnings']:
                        logger.warning(warning)
                    logger.info("인덱스 재구축을 권장합니다. /rebuild-indexes API를 사용하세요.")
                else:
                    logger.info("인덱스 일관성 검증 완료")
                    if validation_result['faiss_available']:
                        logger.info(f"  - FAISS: {validation_result['faiss_document_count']}개 문서")
                    if validation_result['bm25_available']:
                        logger.info(f"  - BM25: {validation_result['bm25_document_count']}개 문서")
                    if qdrant_doc_count is not None:
                        logger.info(f"  - Qdrant: {qdrant_doc_count}개 문서")
                
                # EnsembleRetriever 재초기화 (인덱스 업데이트 반영)
                if validation_result['faiss_available'] and validation_result['bm25_available']:
                    try:
                        from src.utils.config import get_qdrant_config
                        qdrant_config = get_qdrant_config()
                        faiss_weight = qdrant_config.hybrid_search_vector_weight if hasattr(qdrant_config, 'hybrid_search_vector_weight') else 0.7
                        bm25_weight = qdrant_config.hybrid_search_bm25_weight if hasattr(qdrant_config, 'hybrid_search_bm25_weight') else 0.3
                        rrf_c = qdrant_config.hybrid_search_rrf_k if hasattr(qdrant_config, 'hybrid_search_rrf_k') else 60
                        
                        ensemble_created = langchain_manager.create_ensemble_retriever(
                            faiss_weight=faiss_weight,
                            bm25_weight=bm25_weight,
                            c=rrf_c,
                            k=rag.rag_config.default_max_sources
                        )
                        if ensemble_created:
                            logger.info("EnsembleRetriever 재초기화 완료 (업로드된 문서 반영)")
                        else:
                            logger.warning("EnsembleRetriever 재초기화 실패")
                    except Exception as e:
                        logger.warning(f"EnsembleRetriever 재초기화 실패: {str(e)}")
        
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
        success = await rag.process_and_store_documents_async(directory_path, force_update, replace_mode)
        
        processing_time = time.time() - start_time
        
        if success:
            stats = rag.vector_store_manager.get_stats()
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
