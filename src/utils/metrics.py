"""
메트릭 수집 모듈
검색 품질, 답변 품질, 에이전트 메트릭 수집 및 집계
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque
import time
from src.utils.logger import get_logger


@dataclass
class SearchMetrics:
    """검색 품질 메트릭"""
    total_queries: int = 0
    successful_searches: int = 0
    failed_searches: int = 0
    avg_score: float = 0.0
    avg_results_count: float = 0.0
    avg_diversity: float = 0.0
    score_distribution: Dict[str, int] = field(default_factory=lambda: defaultdict(int))


@dataclass
class AnswerMetrics:
    """답변 품질 메트릭"""
    total_answers: int = 0
    avg_confidence: float = 0.0
    avg_hallucination_score: float = 0.0
    avg_processing_time: float = 0.0
    confidence_distribution: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    hallucination_cases: int = 0  # 할루시네이션 점수 > 0.5인 경우


@dataclass
class AgentMetrics:
    """에이전트 메트릭"""
    agent_name: str
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    avg_execution_time: float = 0.0
    execution_times: deque = field(default_factory=lambda: deque(maxlen=100))


class MetricsCollector:
    """메트릭 수집기"""
    
    def __init__(self):
        self.logger = get_logger()
        self.search_metrics = SearchMetrics()
        self.answer_metrics = AnswerMetrics()
        self.agent_metrics: Dict[str, AgentMetrics] = {}
    
    def record_search(
        self,
        success: bool,
        results_count: int = 0,
        avg_score: float = 0.0,
        diversity: float = 0.0
    ):
        """
        검색 메트릭 기록
        
        Args:
            success: 검색 성공 여부
            results_count: 검색 결과 수
            avg_score: 평균 점수
            diversity: 다양성 점수
        """
        self.search_metrics.total_queries += 1
        
        if success:
            self.search_metrics.successful_searches += 1
            self.search_metrics.avg_results_count = (
                (self.search_metrics.avg_results_count * (self.search_metrics.successful_searches - 1) + results_count) /
                self.search_metrics.successful_searches
            )
            self.search_metrics.avg_score = (
                (self.search_metrics.avg_score * (self.search_metrics.successful_searches - 1) + avg_score) /
                self.search_metrics.successful_searches
            )
            self.search_metrics.avg_diversity = (
                (self.search_metrics.avg_diversity * (self.search_metrics.successful_searches - 1) + diversity) /
                self.search_metrics.successful_searches
            )
            
            # 점수 분포
            if avg_score >= 0.9:
                self.search_metrics.score_distribution['high'] += 1
            elif avg_score >= 0.7:
                self.search_metrics.score_distribution['medium'] += 1
            elif avg_score >= 0.5:
                self.search_metrics.score_distribution['low'] += 1
            else:
                self.search_metrics.score_distribution['very_low'] += 1
        else:
            self.search_metrics.failed_searches += 1
    
    def record_answer(
        self,
        confidence: float,
        hallucination_score: float = 0.0,
        processing_time: float = 0.0
    ):
        """
        답변 메트릭 기록
        
        Args:
            confidence: 신뢰도
            hallucination_score: 할루시네이션 점수
            processing_time: 처리 시간
        """
        self.answer_metrics.total_answers += 1
        
        # 평균 계산
        self.answer_metrics.avg_confidence = (
            (self.answer_metrics.avg_confidence * (self.answer_metrics.total_answers - 1) + confidence) /
            self.answer_metrics.total_answers
        )
        self.answer_metrics.avg_hallucination_score = (
            (self.answer_metrics.avg_hallucination_score * (self.answer_metrics.total_answers - 1) + hallucination_score) /
            self.answer_metrics.total_answers
        )
        self.answer_metrics.avg_processing_time = (
            (self.answer_metrics.avg_processing_time * (self.answer_metrics.total_answers - 1) + processing_time) /
            self.answer_metrics.total_answers
        )
        
        # 신뢰도 분포
        if confidence >= 0.9:
            self.answer_metrics.confidence_distribution['high'] += 1
        elif confidence >= 0.7:
            self.answer_metrics.confidence_distribution['medium'] += 1
        elif confidence >= 0.5:
            self.answer_metrics.confidence_distribution['low'] += 1
        else:
            self.answer_metrics.confidence_distribution['very_low'] += 1
        
        # 할루시네이션 케이스
        if hallucination_score > 0.5:
            self.answer_metrics.hallucination_cases += 1
    
    def record_agent_execution(
        self,
        agent_name: str,
        success: bool,
        execution_time: float
    ):
        """
        에이전트 실행 메트릭 기록
        
        Args:
            agent_name: 에이전트 이름
            success: 실행 성공 여부
            execution_time: 실행 시간 (초)
        """
        if agent_name not in self.agent_metrics:
            self.agent_metrics[agent_name] = AgentMetrics(agent_name=agent_name)
        
        metrics = self.agent_metrics[agent_name]
        metrics.total_executions += 1
        
        if success:
            metrics.successful_executions += 1
        else:
            metrics.failed_executions += 1
        
        # 실행 시간 기록
        metrics.execution_times.append(execution_time)
        
        # 평균 실행 시간 계산
        if metrics.execution_times:
            metrics.avg_execution_time = sum(metrics.execution_times) / len(metrics.execution_times)
    
    def get_search_metrics(self) -> Dict[str, Any]:
        """검색 메트릭 조회"""
        success_rate = (
            self.search_metrics.successful_searches / self.search_metrics.total_queries
            if self.search_metrics.total_queries > 0 else 0.0
        )
        
        return {
            "total_queries": self.search_metrics.total_queries,
            "successful_searches": self.search_metrics.successful_searches,
            "failed_searches": self.search_metrics.failed_searches,
            "success_rate": success_rate,
            "avg_score": self.search_metrics.avg_score,
            "avg_results_count": self.search_metrics.avg_results_count,
            "avg_diversity": self.search_metrics.avg_diversity,
            "score_distribution": dict(self.search_metrics.score_distribution)
        }
    
    def get_answer_metrics(self) -> Dict[str, Any]:
        """답변 메트릭 조회"""
        hallucination_rate = (
            self.answer_metrics.hallucination_cases / self.answer_metrics.total_answers
            if self.answer_metrics.total_answers > 0 else 0.0
        )
        
        return {
            "total_answers": self.answer_metrics.total_answers,
            "avg_confidence": self.answer_metrics.avg_confidence,
            "avg_hallucination_score": self.answer_metrics.avg_hallucination_score,
            "avg_processing_time": self.answer_metrics.avg_processing_time,
            "hallucination_rate": hallucination_rate,
            "hallucination_cases": self.answer_metrics.hallucination_cases,
            "confidence_distribution": dict(self.answer_metrics.confidence_distribution)
        }
    
    def get_agent_metrics(self) -> Dict[str, Any]:
        """에이전트 메트릭 조회"""
        result = {}
        
        for agent_name, metrics in self.agent_metrics.items():
            success_rate = (
                metrics.successful_executions / metrics.total_executions
                if metrics.total_executions > 0 else 0.0
            )
            
            result[agent_name] = {
                "total_executions": metrics.total_executions,
                "successful_executions": metrics.successful_executions,
                "failed_executions": metrics.failed_executions,
                "success_rate": success_rate,
                "avg_execution_time": metrics.avg_execution_time
            }
        
        return result
    
    def get_summary(self) -> Dict[str, Any]:
        """종합 메트릭 조회"""
        return {
            "search": self.get_search_metrics(),
            "answer": self.get_answer_metrics(),
            "agents": self.get_agent_metrics()
        }


# 전역 메트릭 수집기 인스턴스
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """메트릭 수집기 싱글톤"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector

