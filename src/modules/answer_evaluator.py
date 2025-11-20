"""
답변 품질 평가 모듈
할루시네이션 감지, 답변-소스 일관성 검증, 신뢰도 계산 개선
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re
from src.utils.logger import get_logger
from src.modules.embedding_module import EmbeddingManager


@dataclass
class AnswerQualityMetrics:
    """답변 품질 메트릭"""
    confidence: float  # 전체 신뢰도 (0.0-1.0)
    hallucination_score: float  # 할루시네이션 점수 (0.0=할루시네이션 없음, 1.0=심각한 할루시네이션)
    source_consistency: float  # 소스 일관성 (0.0-1.0)
    answer_completeness: float  # 답변 완전성 (0.0-1.0)
    source_quality: float  # 소스 품질 (0.0-1.0)
    factors: Dict[str, float]  # 각 팩터별 점수


class AnswerEvaluator:
    """답변 평가자"""
    
    def __init__(self, embedding_manager: Optional[EmbeddingManager] = None):
        """
        Args:
            embedding_manager: 임베딩 관리자 (일관성 검증용)
        """
        self.logger = get_logger()
        self.embedding_manager = embedding_manager
    
    def evaluate_answer(
        self,
        answer: str,
        sources: List[Dict[str, Any]],
        question: Optional[str] = None
    ) -> AnswerQualityMetrics:
        """
        답변 품질 종합 평가
        
        Args:
            answer: 생성된 답변
            sources: 사용된 소스 문서 리스트
            question: 원본 질문 (선택적)
            
        Returns:
            답변 품질 메트릭
        """
        # 1. 할루시네이션 감지
        hallucination_score = self._detect_hallucination(answer, sources)
        
        # 2. 소스 일관성 검증
        source_consistency = self._check_source_consistency(answer, sources)
        
        # 3. 답변 완전성 평가
        answer_completeness = self._evaluate_completeness(answer, sources, question)
        
        # 4. 소스 품질 평가
        source_quality = self._evaluate_source_quality(sources)
        
        # 5. 종합 신뢰도 계산
        confidence = self._calculate_confidence(
            hallucination_score=hallucination_score,
            source_consistency=source_consistency,
            answer_completeness=answer_completeness,
            source_quality=source_quality
        )
        
        factors = {
            'hallucination': 1.0 - hallucination_score,  # 할루시네이션 점수를 신뢰도로 변환
            'consistency': source_consistency,
            'completeness': answer_completeness,
            'source_quality': source_quality
        }
        
        return AnswerQualityMetrics(
            confidence=confidence,
            hallucination_score=hallucination_score,
            source_consistency=source_consistency,
            answer_completeness=answer_completeness,
            source_quality=source_quality,
            factors=factors
        )
    
    def _detect_hallucination(
        self,
        answer: str,
        sources: List[Dict[str, Any]]
    ) -> float:
        """
        할루시네이션 감지
        
        답변 내용이 소스 문서에 없는 정보를 포함하는지 확인합니다.
        
        Args:
            answer: 생성된 답변
            sources: 사용된 소스 문서 리스트
            
        Returns:
            할루시네이션 점수 (0.0=할루시네이션 없음, 1.0=심각한 할루시네이션)
        """
        if not sources:
            # 소스가 없으면 할루시네이션 가능성 높음
            return 0.8
        
        # 소스 내용 통합
        source_texts = []
        for source in sources:
            content = source.get('content', '') or source.get('page_content', '')
            if content:
                source_texts.append(content)
        
        if not source_texts:
            return 0.8
        
        combined_source = ' '.join(source_texts)
        
        # 1. 키워드 기반 검증
        # 답변에서 중요한 키워드/구문 추출
        answer_keywords = self._extract_key_phrases(answer)
        source_keywords = self._extract_key_phrases(combined_source)
        
        # 답변의 키워드가 소스에 있는지 확인
        matched_keywords = sum(1 for kw in answer_keywords if any(kw in src_kw for src_kw in source_keywords))
        keyword_match_ratio = matched_keywords / len(answer_keywords) if answer_keywords else 0.0
        
        # 2. 숫자/수치 검증
        # 답변의 숫자가 소스에 있는지 확인
        answer_numbers = self._extract_numbers(answer)
        source_numbers = self._extract_numbers(combined_source)
        number_match_ratio = self._match_numbers(answer_numbers, source_numbers)
        
        # 3. 임베딩 기반 유사도 검증 (임베딩 매니저가 있는 경우)
        embedding_similarity = 1.0
        if self.embedding_manager and len(answer) > 10:
            try:
                # EmbeddingManager의 get_embedding 메서드 사용
                answer_result = self.embedding_manager.get_embedding(answer)
                source_result = self.embedding_manager.get_embedding(combined_source[:2000])  # 소스는 일부만 사용
                
                if answer_result and source_result and answer_result.embedding and source_result.embedding:
                    # 코사인 유사도 계산
                    import numpy as np
                    answer_vec = np.array(answer_result.embedding)
                    source_vec = np.array(source_result.embedding)
                    
                    similarity = np.dot(answer_vec, source_vec) / (
                        np.linalg.norm(answer_vec) * np.linalg.norm(source_vec)
                    )
                    embedding_similarity = float(similarity)
                else:
                    self.logger.warning("임베딩 결과가 None이거나 embedding 속성이 없습니다.")
            except Exception as e:
                self.logger.warning(f"임베딩 기반 유사도 계산 실패: {str(e)}")
        
        # 할루시네이션 점수 계산 (낮을수록 좋음)
        # 각 팩터의 가중 평균
        hallucination_score = (
            0.3 * (1.0 - keyword_match_ratio) +
            0.2 * (1.0 - number_match_ratio) +
            0.5 * (1.0 - embedding_similarity)
        )
        
        # 점수 정규화 (0.0-1.0)
        hallucination_score = max(0.0, min(1.0, hallucination_score))
        
        self.logger.debug(
            f"할루시네이션 감지: 점수={hallucination_score:.3f}, "
            f"키워드매칭={keyword_match_ratio:.3f}, 숫자매칭={number_match_ratio:.3f}, "
            f"임베딩유사도={embedding_similarity:.3f}"
        )
        
        return hallucination_score
    
    def _check_source_consistency(
        self,
        answer: str,
        sources: List[Dict[str, Any]]
    ) -> float:
        """
        소스 일관성 검증
        
        여러 소스 간 일관성과 답변-소스 간 일관성을 확인합니다.
        
        Args:
            answer: 생성된 답변
            sources: 사용된 소스 문서 리스트
            
        Returns:
            일관성 점수 (0.0-1.0)
        """
        if not sources or len(sources) == 0:
            return 0.0
        
        if len(sources) == 1:
            # 소스가 하나면 일관성은 높음
            return 0.9
        
        # 소스 간 일관성 확인
        source_contents = []
        for source in sources:
            content = source.get('content', '') or source.get('page_content', '')
            if content:
                source_contents.append(content)
        
        if len(source_contents) < 2:
            return 0.5
        
        # 소스 간 유사도 계산 (간단한 키워드 기반)
        source_keywords = [self._extract_key_phrases(content) for content in source_contents]
        
        # 모든 소스 쌍의 키워드 겹침 계산
        overlaps = []
        for i in range(len(source_keywords)):
            for j in range(i + 1, len(source_keywords)):
                overlap = len(set(source_keywords[i]) & set(source_keywords[j]))
                total = len(set(source_keywords[i]) | set(source_keywords[j]))
                if total > 0:
                    overlaps.append(overlap / total)
        
        source_consistency = sum(overlaps) / len(overlaps) if overlaps else 0.0
        
        # 답변-소스 일관성 (할루시네이션 감지 결과 재사용)
        answer_source_consistency = 1.0 - self._detect_hallucination(answer, sources)
        
        # 종합 일관성
        consistency = 0.6 * source_consistency + 0.4 * answer_source_consistency
        
        return max(0.0, min(1.0, consistency))
    
    def _evaluate_completeness(
        self,
        answer: str,
        sources: List[Dict[str, Any]],
        question: Optional[str] = None
    ) -> float:
        """
        답변 완전성 평가
        
        답변이 질문에 충분히 답했는지, 소스의 주요 정보를 포함했는지 확인합니다.
        
        Args:
            answer: 생성된 답변
            sources: 사용된 소스 문서 리스트
            question: 원본 질문 (선택적)
            
        Returns:
            완전성 점수 (0.0-1.0)
        """
        if not answer or len(answer.strip()) < 10:
            return 0.0
        
        completeness_factors = []
        
        # 1. 답변 길이 평가
        answer_length = len(answer.strip())
        if answer_length < 50:
            length_score = 0.3
        elif answer_length < 200:
            length_score = 0.7
        else:
            length_score = 1.0
        completeness_factors.append(length_score)
        
        # 2. 구조화 정도 평가 (목록, 제목 등)
        has_structure = bool(
            re.search(r'[0-9]+[\.\)]', answer) or  # 번호 목록
            '**' in answer or  # 마크다운 볼드
            '\n' in answer and answer.count('\n') >= 2  # 여러 줄
        )
        structure_score = 1.0 if has_structure else 0.5
        completeness_factors.append(structure_score)
        
        # 3. 질문 키워드 포함 여부 (질문이 있는 경우)
        if question:
            question_keywords = self._extract_key_phrases(question)
            answer_keywords = self._extract_key_phrases(answer)
            
            matched = sum(1 for qk in question_keywords if any(qk in ak for ak in answer_keywords))
            keyword_coverage = matched / len(question_keywords) if question_keywords else 0.0
            completeness_factors.append(keyword_coverage)
        
        # 4. 소스 정보 활용도
        if sources:
            source_keywords = set()
            for source in sources:
                content = source.get('content', '') or source.get('page_content', '')
                if content:
                    source_keywords.update(self._extract_key_phrases(content))
            
            answer_keywords = set(self._extract_key_phrases(answer))
            source_utilization = len(answer_keywords & source_keywords) / len(source_keywords) if source_keywords else 0.0
            completeness_factors.append(source_utilization)
        
        # 평균 계산
        completeness = sum(completeness_factors) / len(completeness_factors) if completeness_factors else 0.0
        
        return max(0.0, min(1.0, completeness))
    
    def _evaluate_source_quality(
        self,
        sources: List[Dict[str, Any]]
    ) -> float:
        """
        소스 품질 평가
        
        소스의 점수, 개수, 다양성을 평가합니다.
        
        Args:
            sources: 소스 문서 리스트
            
        Returns:
            소스 품질 점수 (0.0-1.0)
        """
        if not sources:
            return 0.0
        
        quality_factors = []
        
        # 1. 소스 점수 평균
        scores = [float(source.get('score', 0.0)) for source in sources if 'score' in source]
        if scores:
            avg_score = sum(scores) / len(scores)
            quality_factors.append(avg_score)
        
        # 2. 소스 개수 (적절한 수인지)
        source_count = len(sources)
        if source_count == 0:
            count_score = 0.0
        elif source_count < 2:
            count_score = 0.5
        elif source_count <= 5:
            count_score = 1.0
        else:
            count_score = 0.8  # 너무 많으면 오히려 낮음
        quality_factors.append(count_score)
        
        # 3. 소스 다양성 (다른 파일에서 온 소스인지)
        source_files = set()
        for source in sources:
            file_path = source.get('source_file', '') or source.get('metadata', {}).get('source', '')
            if file_path:
                source_files.add(file_path)
        
        diversity_score = min(1.0, len(source_files) / max(1, len(sources)))
        quality_factors.append(diversity_score)
        
        # 평균 계산
        quality = sum(quality_factors) / len(quality_factors) if quality_factors else 0.0
        
        return max(0.0, min(1.0, quality))
    
    def _calculate_confidence(
        self,
        hallucination_score: float,
        source_consistency: float,
        answer_completeness: float,
        source_quality: float
    ) -> float:
        """
        종합 신뢰도 계산
        
        Args:
            hallucination_score: 할루시네이션 점수 (0.0=없음, 1.0=심각)
            source_consistency: 소스 일관성 (0.0-1.0)
            answer_completeness: 답변 완전성 (0.0-1.0)
            source_quality: 소스 품질 (0.0-1.0)
            
        Returns:
            종합 신뢰도 (0.0-1.0)
        """
        # 할루시네이션 점수를 신뢰도로 변환
        hallucination_confidence = 1.0 - hallucination_score
        
        # 가중 평균
        confidence = (
            0.4 * hallucination_confidence +  # 할루시네이션 방지가 가장 중요
            0.3 * source_consistency +
            0.2 * answer_completeness +
            0.1 * source_quality
        )
        
        return max(0.0, min(1.0, confidence))
    
    def _extract_key_phrases(self, text: str, max_phrases: int = 20) -> List[str]:
        """텍스트에서 주요 구문 추출"""
        if not text:
            return []
        
        # 간단한 키워드 추출 (실제로는 형태소 분석 사용 가능)
        # 한국어 명사, 숫자, 영문 단어 추출
        words = re.findall(r'[가-힣]{2,}|[0-9]+(?:\.[0-9]+)?|[A-Za-z]{3,}', text)
        
        # 빈도 기반 상위 키워드 선택
        from collections import Counter
        word_counts = Counter(words)
        top_words = [word for word, _ in word_counts.most_common(max_phrases)]
        
        return top_words
    
    def _extract_numbers(self, text: str) -> List[str]:
        """텍스트에서 숫자 추출"""
        numbers = re.findall(r'[0-9]+(?:\.[0-9]+)?', text)
        return numbers
    
    def _match_numbers(
        self,
        answer_numbers: List[str],
        source_numbers: List[str]
    ) -> float:
        """답변의 숫자가 소스에 있는지 확인"""
        if not answer_numbers:
            return 1.0  # 숫자가 없으면 매칭 문제 없음
        
        matched = sum(1 for num in answer_numbers if num in source_numbers)
        return matched / len(answer_numbers) if answer_numbers else 0.0

