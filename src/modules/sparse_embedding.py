"""
Sparse 임베딩 모듈
BM25 기반 sparse 벡터 생성 (Qdrant 하이브리드 검색용)
"""

from typing import List, Dict, Optional, Callable
from collections import Counter
import math
import re
import json
from pathlib import Path

from langchain_qdrant.sparse_embeddings import SparseEmbeddings, SparseVector

from src.utils.logger import get_logger
from src.modules.langchain_retrievers import korean_preprocessing


class BM25SparseEmbedding(SparseEmbeddings):
    """BM25 기반 sparse 벡터 생성"""
    
    def __init__(
        self,
        preprocess_func: Optional[Callable[[str], List[str]]] = None,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
        use_morphological: bool = True
    ):
        """
        Args:
            preprocess_func: 텍스트 전처리 함수 (토큰화)
            k1: BM25 파라미터 k1
            b: BM25 파라미터 b
            epsilon: BM25 파라미터 epsilon (음수 IDF 방지)
            use_morphological: 형태소 분석 사용 여부 (기본값: False)
        """
        self.logger = get_logger()
        
        # 형태소 분석 사용 시 전처리 함수 래핑
        if preprocess_func is None:
            if use_morphological:
                # 형태소 분석을 사용하는 전처리 함수 생성
                def morphological_preprocess(text: str) -> List[str]:
                    return korean_preprocessing(text, use_morphological=True)
                self.preprocess_func = morphological_preprocess
            else:
                self.preprocess_func = korean_preprocessing
        else:
            self.preprocess_func = preprocess_func
        
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.use_morphological = use_morphological
        
        # 문서 집합 통계 (학습 후 사용)
        self.corpus_size: int = 0
        self.avgdl: float = 0.0
        self.doc_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}
        self.doc_len: List[int] = []
        
        # Vocabulary (토큰 -> 인덱스 매핑)
        self.vocabulary: Dict[str, int] = {}
        self.vocabulary_reverse: Dict[int, str] = {}
        
        morphological_status = "활성화" if use_morphological else "비활성화"
        self.logger.info(f"BM25SparseEmbedding 초기화: k1={k1}, b={b}, epsilon={epsilon}, 형태소분석={morphological_status}")
    
    def fit(self, documents: List[str], deduplicate: bool = True) -> None:
        """
        문서 집합으로 BM25 모델 학습
        
        Args:
            documents: 학습할 문서 리스트
            deduplicate: 중복 문서 제거 여부 (기본값: True)
        """
        if not documents:
            self.logger.warning("학습할 문서가 없습니다.")
            return
        
        # 중복 문서 제거 (안정성을 위한 이중 체크)
        unique_documents = documents
        if deduplicate:
            import hashlib
            seen_hashes = set()
            unique_documents = []
            for doc in documents:
                if not doc or not doc.strip():
                    continue  # 빈 문서 제외
                doc_hash = hashlib.sha256(doc.encode('utf-8')).hexdigest()
                if doc_hash not in seen_hashes:
                    seen_hashes.add(doc_hash)
                    unique_documents.append(doc)
            
            if len(unique_documents) < len(documents):
                removed_count = len(documents) - len(unique_documents)
                self.logger.info(f"중복 문서 제거: {removed_count}개 제외 (전체: {len(documents)}개 → 고유: {len(unique_documents)}개)")
        
        if not unique_documents:
            self.logger.error("유효한 문서가 없습니다.")
            return
        
        self.logger.info(f"BM25 모델 학습 시작: {len(unique_documents)}개 문서")
        
        # 문서 토큰화
        tokenized_docs = []
        for doc in unique_documents:
            tokens = self.preprocess_func(doc)
            if tokens:  # 빈 토큰 리스트 제외
                tokenized_docs.append(tokens)
        
        if not tokenized_docs:
            self.logger.error("토큰화된 문서가 없습니다.")
            return
        
        # Vocabulary 구축
        all_tokens = set()
        for tokens in tokenized_docs:
            all_tokens.update(tokens)
        
        # 토큰을 인덱스로 매핑
        self.vocabulary = {token: idx for idx, token in enumerate(sorted(all_tokens))}
        self.vocabulary_reverse = {idx: token for token, idx in self.vocabulary.items()}
        
        self.logger.info(f"Vocabulary 크기: {len(self.vocabulary)}")
        
        # BM25 통계 계산
        self.corpus_size = len(tokenized_docs)
        total_length = 0
        
        self.doc_freqs = []
        self.doc_len = []
        nd: Dict[str, int] = {}  # word -> number of documents with word
        
        for tokens in tokenized_docs:
            doc_len = len(tokens)
            self.doc_len.append(doc_len)
            total_length += doc_len
            
            # 문서 내 토큰 빈도
            frequencies = Counter(tokens)
            self.doc_freqs.append(dict(frequencies))
            
            # 문서 빈도 계산
            for word in set(tokens):
                nd[word] = nd.get(word, 0) + 1
        
        self.avgdl = total_length / self.corpus_size if self.corpus_size > 0 else 0.0
        
        # IDF 계산 (BM25Okapi 방식)
        idf_sum = 0.0
        negative_idfs = []
        
        for word, freq in nd.items():
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        
        if len(self.idf) > 0:
            average_idf = idf_sum / len(self.idf)
            eps = self.epsilon * average_idf
            for word in negative_idfs:
                self.idf[word] = eps
        
        self.logger.info(f"BM25 모델 학습 완료: corpus_size={self.corpus_size}, avgdl={self.avgdl:.2f}")
    
    def save_vocabulary(self, file_path: str, include_doc_stats: bool = False) -> bool:
        """
        Vocabulary와 IDF를 파일에 저장
        
        Args:
            file_path: 저장할 파일 경로
            include_doc_stats: doc_freqs와 doc_len도 저장할지 여부 (기본값: False)
            
        Returns:
            저장 성공 여부
        """
        try:
            if self.corpus_size == 0:
                self.logger.warning("학습되지 않은 모델은 저장할 수 없습니다.")
                return False
            
            # 저장할 데이터 준비
            data = {
                'corpus_size': self.corpus_size,
                'avgdl': self.avgdl,
                'vocabulary': self.vocabulary,
                'vocabulary_reverse': self.vocabulary_reverse,
                'idf': {k: float(v) for k, v in self.idf.items()},  # float로 변환 (JSON 호환)
                'k1': self.k1,
                'b': self.b,
                'epsilon': self.epsilon,
            }
            
            # doc_freqs와 doc_len 저장 (선택적)
            if include_doc_stats:
                data['doc_freqs'] = self.doc_freqs
                data['doc_len'] = self.doc_len
                self.logger.info("문서 통계(doc_freqs, doc_len)도 함께 저장합니다 (파일 크기 증가 가능)")
            
            # 디렉토리 생성
            file_path_obj = Path(file_path)
            file_path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            # JSON 파일로 저장
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Vocabulary 저장 완료: {file_path} (vocabulary_size={len(self.vocabulary)})")
            return True
            
        except Exception as e:
            self.logger.error(f"Vocabulary 저장 실패: {str(e)}")
            import traceback
            self.logger.error(f"상세 오류: {traceback.format_exc()}")
            return False
    
    def load_vocabulary(self, file_path: str) -> bool:
        """
        파일에서 Vocabulary와 IDF 로드
        
        Args:
            file_path: 로드할 파일 경로
            
        Returns:
            로드 성공 여부
        """
        try:
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                self.logger.debug(f"Vocabulary 파일이 없습니다: {file_path}")
                return False
            
            # JSON 파일에서 로드
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 데이터 복원
            self.corpus_size = data.get('corpus_size', 0)
            self.avgdl = data.get('avgdl', 0.0)
            self.vocabulary = data.get('vocabulary', {})
            self.vocabulary_reverse = data.get('vocabulary_reverse', {})
            self.idf = data.get('idf', {})
            
            # 파라미터 복원 (선택적)
            if 'k1' in data:
                self.k1 = data['k1']
            if 'b' in data:
                self.b = data['b']
            if 'epsilon' in data:
                self.epsilon = data['epsilon']
            
            # doc_freqs와 doc_len 복원 (저장된 경우)
            if 'doc_freqs' in data and 'doc_len' in data:
                self.doc_freqs = data['doc_freqs']
                self.doc_len = data['doc_len']
                self.logger.info("문서 통계(doc_freqs, doc_len) 복원 완료")
            else:
                # 저장되지 않은 경우 빈 리스트로 초기화
                self.doc_freqs = []
                self.doc_len = []
                self.logger.debug("문서 통계(doc_freqs, doc_len)가 저장되지 않아 빈 리스트로 초기화")
            
            if self.corpus_size > 0 and len(self.vocabulary) > 0:
                self.logger.info(f"Vocabulary 로드 완료: {file_path} (vocabulary_size={len(self.vocabulary)}, corpus_size={self.corpus_size})")
                return True
            else:
                self.logger.warning(f"Vocabulary 파일이 비어있거나 손상되었습니다: {file_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"Vocabulary 로드 실패: {str(e)}")
            import traceback
            self.logger.error(f"상세 오류: {traceback.format_exc()}")
            return False
    
    def _compute_bm25_vector(self, tokens: List[str]) -> SparseVector:
        """
        토큰 리스트에 대해 BM25 점수를 계산하여 sparse 벡터 생성
        
        Args:
            tokens: 토큰 리스트
            
        Returns:
            SparseVector: sparse 벡터
        """
        if not tokens or self.corpus_size == 0:
            return SparseVector(indices=[], values=[])
        
        # 문서 내 토큰 빈도
        token_freqs = Counter(tokens)
        doc_len = len(tokens)
        
        # BM25 점수 계산 (단일 문서에 대한 점수)
        # 실제로는 쿼리와 문서 간 점수를 계산하지만,
        # 여기서는 문서 자체의 토큰 가중치를 계산
        indices = []
        values = []
        
        for token, freq in token_freqs.items():
            if token not in self.vocabulary:
                continue
            
            token_idx = self.vocabulary[token]
            idf = self.idf.get(token, 0.0)
            
            # BM25 스타일 점수 계산
            # 실제 BM25는 쿼리-문서 점수이지만, 여기서는 문서의 토큰 가중치를 계산
            # TF * IDF 스타일로 계산하되, BM25의 정규화 적용
            tf = freq
            normalized_tf = (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            )
            score = idf * normalized_tf
            
            if score > 0:
                indices.append(token_idx)
                values.append(float(score))
        
        return SparseVector(indices=indices, values=values)
    
    def embed_documents(self, texts: List[str]) -> List[SparseVector]:
        """
        문서 리스트에 대해 sparse 벡터 생성
        
        Args:
            texts: 문서 리스트
            
        Returns:
            SparseVector 리스트
        """
        if self.corpus_size == 0:
            self.logger.warning("BM25 모델이 학습되지 않았습니다. 빈 벡터를 반환합니다.")
            return [SparseVector(indices=[], values=[]) for _ in texts]
        
        results = []
        for text in texts:
            tokens = self.preprocess_func(text)
            sparse_vec = self._compute_bm25_vector(tokens)
            results.append(sparse_vec)
        
        return results
    
    def embed_query(self, text: str = None, keywords: Optional[List[str]] = None, entities: Optional[List[str]] = None) -> SparseVector:
        """
        쿼리 텍스트 또는 키워드/엔티티로 sparse 벡터 생성
        
        Args:
            text: 쿼리 텍스트 (하위 호환성 유지, keywords/entities가 없을 때만 사용)
            keywords: query_refiner에서 추출한 키워드 리스트
            entities: query_refiner에서 추출한 엔티티 리스트
            
        Returns:
            SparseVector: sparse 벡터
        """
        if self.corpus_size == 0:
            self.logger.warning("BM25 모델이 학습되지 않았습니다. 빈 벡터를 반환합니다.")
            return SparseVector(indices=[], values=[])
        
        # keywords와 entities가 제공되면 직접 사용 (query_refiner에서 추출한 결과)
        if keywords is not None or entities is not None:
            all_tokens = []
            if keywords:
                all_tokens.extend(keywords)
            if entities:
                all_tokens.extend(entities)
            
            # vocabulary에 있는 토큰만 필터링
            indices = []
            values = []
            for token in set(all_tokens):  # 중복 제거
                if token not in self.vocabulary:
                    continue  # vocabulary에 없으면 제외
                
                token_idx = self.vocabulary[token]
                idf = self.idf.get(token, 0.0)
                
                if idf > 0:
                    indices.append(token_idx)
                    values.append(float(idf))
            
            return SparseVector(indices=indices, values=values)
        
        # 하위 호환성: text가 제공되면 기존 방식 사용
        if text:
            tokens = self.preprocess_func(text)
            
            # 쿼리의 경우 단순히 토큰의 IDF를 사용
            indices = []
            values = []
            
            for token in set(tokens):  # 쿼리에서는 중복 제거
                if token not in self.vocabulary:
                    continue
                
                token_idx = self.vocabulary[token]
                idf = self.idf.get(token, 0.0)
                
                if idf > 0:
                    indices.append(token_idx)
                    values.append(float(idf))
            
            return SparseVector(indices=indices, values=values)
        
        # text도 keywords/entities도 없는 경우 빈 벡터 반환
        self.logger.warning("text, keywords, entities 모두 제공되지 않았습니다. 빈 벡터를 반환합니다.")
        return SparseVector(indices=[], values=[])


class SparseEmbeddingManager:
    """Sparse 임베딩 관리자"""
    
    def __init__(
        self,
        preprocess_func: Optional[Callable[[str], List[str]]] = None,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
        vocabulary_path: Optional[str] = None,
        use_morphological: bool = True
    ):
        """
        Args:
            preprocess_func: 텍스트 전처리 함수
            k1: BM25 파라미터 k1
            b: BM25 파라미터 b
            epsilon: BM25 파라미터 epsilon
            vocabulary_path: Vocabulary 저장 경로 (None이면 자동 로드 안 함)
            use_morphological: 형태소 분석 사용 여부 (기본값: True)
        """
        self.logger = get_logger()
        self.vocabulary_path = vocabulary_path
        self.sparse_embedding = BM25SparseEmbedding(
            preprocess_func=preprocess_func,
            k1=k1,
            b=b,
            epsilon=epsilon,
            use_morphological=use_morphological
        )
        self.is_fitted = False
        
        # Vocabulary 파일이 있으면 자동 로드
        if self.vocabulary_path:
            if self.sparse_embedding.load_vocabulary(self.vocabulary_path):
                self.is_fitted = True
                self.logger.info("저장된 Vocabulary를 자동으로 로드했습니다.")
    
    def fit(self, documents: List[str], include_doc_stats: bool = False) -> None:
        """
        문서 집합으로 모델 학습
        
        Args:
            documents: 학습할 문서 리스트
            include_doc_stats: doc_freqs/doc_len도 저장할지 여부 (기본값: False)
        """
        self.sparse_embedding.fit(documents)
        self.is_fitted = True
        self.logger.info("Sparse 임베딩 모델 학습 완료")
        
        # 학습 후 자동 저장
        if self.vocabulary_path:
            if self.sparse_embedding.save_vocabulary(self.vocabulary_path, include_doc_stats=include_doc_stats):
                stats_info = " (문서 통계 포함)" if include_doc_stats else ""
                self.logger.info(f"Vocabulary가 영구 저장되었습니다: {self.vocabulary_path}{stats_info}")
            else:
                self.logger.warning("Vocabulary 저장에 실패했습니다.")
    
    def get_sparse_embedding(self) -> BM25SparseEmbedding:
        """Sparse 임베딩 인스턴스 반환"""
        if not self.is_fitted:
            self.logger.warning("Sparse 임베딩 모델이 학습되지 않았습니다.")
        return self.sparse_embedding
    
    def embed_documents(self, texts: List[str]) -> List[SparseVector]:
        """문서 리스트에 대해 sparse 벡터 생성"""
        return self.sparse_embedding.embed_documents(texts)
    
    def embed_query(self, text: str) -> SparseVector:
        """쿼리 텍스트에 대해 sparse 벡터 생성"""
        return self.sparse_embedding.embed_query(text)

