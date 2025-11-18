"""
Cross-Encoder 리랭커 모듈 (로컬 모델 경로 지원)
"""

from typing import List, Dict, Any, Optional

from src.utils.logger import get_logger


class CrossEncoderReranker:
    """로컬 Cross-Encoder 기반 리랭커"""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        batch_size: int = 32,
    ) -> None:
        self.logger = get_logger()
        self.model_path = model_path
        self.batch_size = batch_size

        # 디바이스 자동 선택
        resolved_device = device
        if device == "cuda":
            try:
                import torch  # lazy import
                if not torch.cuda.is_available():
                    self.logger.warning("CUDA를 사용할 수 없습니다. 리랭커를 CPU로 전환합니다.")
                    resolved_device = "cpu"
            except Exception:
                resolved_device = "cpu"
        self.device = resolved_device

        # 모델 로드 (로컬 경로)
        try:
            # 경로 검증 (폴더 존재 확인)
            import os
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"리랭커 모델 경로가 존재하지 않습니다: {self.model_path}")

            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_path, device=self.device)
            self.logger.info(
                f"리랭커 모델 로드 완료: {self.model_path} (device={self.device})"
            )
        except Exception as e:
            self.logger.error(
                "리랭커 모델 로드 실패: %s\n"
                "확인사항: (1) 경로가 올바른가? (2) Cross-Encoder 호환 모델인가? "
                "(3) transformers/sentence-transformers 버전 호환성 (4) CPU/GPU 환경 설정",
                str(e)
            )
            raise

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        문서 리스트를 리랭킹하여 점수 높은 순으로 반환

        Args:
            query: 사용자 질의
            documents: 'content' 키를 포함한 문서 리스트
            top_k: 상위 몇 개까지 리랭킹 결과를 사용할지
        """
        if not documents:
            return []

        # CrossEncoder 입력 쌍 구성
        pairs = [(query, doc.get("content", "")) for doc in documents]

        # 배치 스코어링 (배치별 예외 격리)
        scores: List[float] = []
        for i in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[i : i + self.batch_size]
            try:
                batch_scores = self.model.predict(batch_pairs)
                scores.extend([float(s) for s in batch_scores])
            except Exception as e:
                self.logger.error(f"리랭커 배치 스코어 실패(offset={i}, size={len(batch_pairs)}): {str(e)}")
                # 실패 배치는 0.0 점수로 대체하여 전체 파이프라인을 유지
                scores.extend([0.0 for _ in batch_pairs])

        # 문서에 리랭커 점수 부여
        for doc, score in zip(documents, scores):
            doc["reranker_score"] = float(score)

        # 리랭커 점수 기준 정렬
        documents.sort(key=lambda d: d.get("reranker_score", 0.0), reverse=True)

        if top_k is not None and top_k > 0:
            documents = documents[:top_k]

        return documents
    
    # ========== 비동기 메서드 (Phase 3: 리랭커 비동기화) ==========
    
    async def rerank_async(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        비동기 문서 리스트 리랭킹 (CPU/GPU 연산을 asyncio.to_thread로 비동기화)
        
        Args:
            query: 사용자 질의
            documents: 'content' 키를 포함한 문서 리스트
            top_k: 상위 몇 개까지 리랭킹 결과를 사용할지
        """
        # CPU/GPU 연산이므로 asyncio.to_thread 사용
        import asyncio
        return await asyncio.to_thread(self.rerank, query, documents, top_k)


