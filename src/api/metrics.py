"""
메트릭 API 엔드포인트
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from pydantic import BaseModel, Field
from src.utils.metrics import get_metrics_collector

router = APIRouter(prefix="/metrics", tags=["메트릭"])


class SearchMetricsResponse(BaseModel):
    """검색 메트릭 응답"""
    total_queries: int
    successful_searches: int
    failed_searches: int
    success_rate: float
    avg_score: float
    avg_results_count: float
    avg_diversity: float
    score_distribution: Dict[str, int]


class AnswerMetricsResponse(BaseModel):
    """답변 메트릭 응답"""
    total_answers: int
    avg_confidence: float
    avg_hallucination_score: float
    avg_processing_time: float
    hallucination_rate: float
    hallucination_cases: int
    confidence_distribution: Dict[str, int]


class AgentMetricsResponse(BaseModel):
    """에이전트 메트릭 응답"""
    agents: Dict[str, Dict[str, Any]]


class MetricsSummaryResponse(BaseModel):
    """종합 메트릭 응답"""
    search: Dict[str, Any]
    answer: Dict[str, Any]
    agents: Dict[str, Any]


@router.get(
    "/search",
    response_model=SearchMetricsResponse,
    summary="검색 품질 메트릭",
    description="검색 품질 관련 메트릭을 조회합니다."
)
async def get_search_metrics():
    """검색 품질 메트릭 조회"""
    collector = get_metrics_collector()
    metrics = collector.get_search_metrics()
    return SearchMetricsResponse(**metrics)


@router.get(
    "/answer",
    response_model=AnswerMetricsResponse,
    summary="답변 품질 메트릭",
    description="답변 품질 관련 메트릭을 조회합니다."
)
async def get_answer_metrics():
    """답변 품질 메트릭 조회"""
    collector = get_metrics_collector()
    metrics = collector.get_answer_metrics()
    return AnswerMetricsResponse(**metrics)


@router.get(
    "/agents",
    response_model=AgentMetricsResponse,
    summary="에이전트 메트릭",
    description="에이전트 실행 관련 메트릭을 조회합니다."
)
async def get_agent_metrics():
    """에이전트 메트릭 조회"""
    collector = get_metrics_collector()
    metrics = collector.get_agent_metrics()
    return AgentMetricsResponse(agents=metrics)


@router.get(
    "/summary",
    response_model=MetricsSummaryResponse,
    summary="종합 메트릭",
    description="모든 메트릭을 종합하여 조회합니다."
)
async def get_metrics_summary():
    """종합 메트릭 조회"""
    collector = get_metrics_collector()
    summary = collector.get_summary()
    return MetricsSummaryResponse(**summary)

