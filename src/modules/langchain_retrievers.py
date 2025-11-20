"""
LangChain 기반 검색기 모듈
BM25Retriever, FAISS, EnsembleRetriever 통합
"""

from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
import pickle
import inspect

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.embeddings import Embeddings as LangChainEmbeddings

from src.utils.logger import get_logger
from src.modules.document_processor import DocumentChunk
from src.modules.langchain_embedding_wrapper import EmbeddingManagerWrapper


def document_chunk_to_langchain_document(chunk: DocumentChunk) -> Document:
    """
    DocumentChunk를 LangChain Document로 변환
    
    Args:
        chunk: DocumentChunk 객체
        
    Returns:
        LangChain Document 객체
    """
    # 메타데이터에 chunk_id와 source_file 추가
    metadata = chunk.metadata.copy()
    metadata.update({
        'chunk_id': chunk.chunk_id,
        'source_file': chunk.source_file,
        'chunk_index': chunk.chunk_index,
        'doc_id': chunk.doc_id,
        'section_id': chunk.section_id,
        'chunk_type': chunk.chunk_type,
        'heading_path': chunk.heading_path,
        'page_start': chunk.page_start,
        'page_end': chunk.page_end,
        'language': chunk.language,
        'domain': chunk.domain,
        'embedding_version': chunk.embedding_version,
        'document': chunk.doc_metadata.to_dict() if chunk.doc_metadata else None,
    })
    
    return Document(
        page_content=chunk.content,
        metadata=metadata,
        id=chunk.chunk_id
    )


def langchain_document_to_dict(doc: Document) -> Dict[str, Any]:
    """
    LangChain Document를 기존 형식의 딕셔너리로 변환
    
    Args:
        doc: LangChain Document 객체
        
    Returns:
        기존 검색 결과 형식의 딕셔너리
    """
    return {
        'chunk_id': doc.id or doc.metadata.get('chunk_id', ''),
        'content': doc.page_content,
        'metadata': doc.metadata,
        'source_file': doc.metadata.get('source_file', ''),
        'chunk_index': doc.metadata.get('chunk_index', 0),
        'score': doc.metadata.get('score', 0.0)  # score가 있으면 사용, 없으면 0
    }


def korean_preprocessing(text: str, use_morphological: bool = False) -> List[str]:
    """
    한국어 텍스트 전처리 함수 (BM25용)
    
    Args:
        text: 원본 텍스트
        use_morphological: 형태소 분석 사용 여부 (기본값: False)
        
    Returns:
        토큰 리스트
    """
    import re
    
    # 형태소 분석기 사용 (선택적)
    if use_morphological:
        try:
            from konlpy.tag import Okt
            okt = Okt()
            # 형태소 분석 (morphs: 형태소 단위로 분리)
            tokens = okt.morphs(text)
            # 길이 1 이상인 토큰만 반환
            return [token for token in tokens if len(token) > 1]
        except ImportError:
            # KoNLPy가 설치되지 않은 경우 기본 방식으로 폴백
            logger = get_logger()
            logger.debug("KoNLPy가 설치되지 않아 기본 토크나이저 사용 (형태소 분석 비활성화)")
            # 기본 방식으로 계속 진행
        except Exception as e:
            # 형태소 분석 중 오류 발생 시 기본 방식으로 폴백
            logger = get_logger()
            logger.warning(f"형태소 분석 실패: {str(e)}. 기본 토크나이저 사용")
            # 기본 방식으로 계속 진행
    
    # 기본 방식: 개선된 정규식 기반 토큰화
    # 한국어 어절 패턴
    korean_pattern = r'[가-힣]+'
    # 영문 단어 패턴
    english_pattern = r'[a-zA-Z]+'
    # 숫자 패턴
    number_pattern = r'[0-9]+'
    
    tokens = []
    tokens.extend(re.findall(korean_pattern, text))
    tokens.extend(re.findall(english_pattern, text))
    tokens.extend(re.findall(number_pattern, text))
    
    # 영문은 소문자 변환, 한국어는 그대로
    result = []
    for token in tokens:
        if len(token) > 1:
            if token.isascii():
                result.append(token.lower())
            else:
                result.append(token)
    
    return result


class LangChainRetrievalManager:
    """LangChain 기반 검색 관리자"""
    
    def __init__(
        self,
        embedding_function: LangChainEmbeddings,
        faiss_storage_path: str = "data/faiss_index",
        bm25_storage_path: str = "data/bm25_index",
        bm25_preprocess_func: Optional[Callable[[str], List[str]]] = None,
        faiss_use_gpu: bool = True
    ):
        self.logger = get_logger()
        self.embedding_function = embedding_function
        self.faiss_storage_path = Path(faiss_storage_path)
        self.faiss_storage_path.mkdir(parents=True, exist_ok=True)
        
        # BM25 저장 경로
        self.bm25_storage_path = Path(bm25_storage_path)
        self.bm25_storage_path.mkdir(parents=True, exist_ok=True)
        
        # BM25 전처리 함수
        self.bm25_preprocess_func = bm25_preprocess_func or korean_preprocessing
        
        # FAISS GPU 설정
        self.faiss_use_gpu = faiss_use_gpu
        self._faiss_gpu_resource = None
        if self.faiss_use_gpu:
            self._init_faiss_gpu()
        
        # 검색기들
        self.faiss_store: Optional[FAISS] = None
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.ensemble_retriever: Optional[EnsembleRetriever] = None
        
        # FAISS save_local 메서드 시그니처 확인 (allow_dangerous_deserialization 지원 여부)
        self._faiss_supports_dangerous_deserialization = self._check_faiss_save_local_signature()
        # load_local 메서드도 확인 (load는 항상 지원하므로 별도 체크)
        self._faiss_load_supports_dangerous = self._check_faiss_load_local_signature()
    
    def _init_faiss_gpu(self):
        """FAISS GPU 리소스 초기화"""
        try:
            import faiss
            if hasattr(faiss, 'StandardGpuResources'):
                self._faiss_gpu_resource = faiss.StandardGpuResources()
                self.logger.info("FAISS GPU 리소스 초기화 완료")
            else:
                self.logger.warning("FAISS GPU가 설치되지 않았습니다. CPU를 사용합니다.")
                self.faiss_use_gpu = False
        except ImportError:
            self.logger.warning("faiss-gpu 패키지가 설치되지 않았습니다. CPU를 사용합니다.")
            self.faiss_use_gpu = False
        except Exception as e:
            self.logger.warning(f"FAISS GPU 초기화 실패: {str(e)}. CPU를 사용합니다.")
            self.faiss_use_gpu = False
    
    def _check_faiss_save_local_signature(self) -> bool:
        """
        FAISS.save_local() 메서드가 allow_dangerous_deserialization 파라미터를 지원하는지 확인
        
        Returns:
            지원 여부
        """
        try:
            sig = inspect.signature(FAISS.save_local)
            return 'allow_dangerous_deserialization' in sig.parameters
        except Exception:
            # 시그니처 확인 실패 시 False 반환 (안전하게 처리)
            return False
    
    def _check_faiss_load_local_signature(self) -> bool:
        """
        FAISS.load_local() 메서드가 allow_dangerous_deserialization 파라미터를 지원하는지 확인
        
        Returns:
            지원 여부
        """
        try:
            sig = inspect.signature(FAISS.load_local)
            return 'allow_dangerous_deserialization' in sig.parameters
        except Exception:
            # 시그니처 확인 실패 시 False 반환 (안전하게 처리)
            return False
    
    def initialize_faiss_from_chunks(
        self,
        chunks: List[DocumentChunk],
        index_name: str = "default"
    ) -> bool:
        """
        DocumentChunk 리스트로부터 FAISS 인덱스 생성
        
        Args:
            chunks: DocumentChunk 리스트
            index_name: 인덱스 이름
            
        Returns:
            생성 성공 여부
        """
        try:
            self.logger.info(f"FAISS 인덱스 생성 시작: {len(chunks)}개 청크")
            
            # DocumentChunk를 LangChain Document로 변환
            documents = [document_chunk_to_langchain_document(chunk) for chunk in chunks]
            
            # FAISS 인덱스 생성
            self.faiss_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embedding_function
            )
            
            # GPU 사용 시 인덱스를 GPU로 이동 (LangChain FAISS는 제한적 지원)
            # 주의: LangChain FAISS는 내부적으로 관리하므로 직접 GPU 변환이 제한적일 수 있음
            if self.faiss_use_gpu and self._faiss_gpu_resource is not None:
                try:
                    import faiss
                    # LangChain FAISS의 내부 인덱스 접근 시도
                    if hasattr(self.faiss_store, '_index') and self.faiss_store._index is not None:
                        # GPU 인덱스로 변환 시도
                        # 주의: LangChain이 인덱스를 재생성할 수 있으므로 주의 필요
                        cpu_index = self.faiss_store._index
                        if not isinstance(cpu_index, faiss.GpuIndex):
                            try:
                                gpu_index = faiss.index_cpu_to_gpu(self._faiss_gpu_resource, 0, cpu_index)
                                # LangChain FAISS가 이를 인식할 수 있도록 속성 설정 시도
                                # 실제로는 LangChain이 이를 다시 CPU로 변환할 수 있음
                                self.logger.info("FAISS 인덱스를 GPU로 이동 시도 (LangChain 제한으로 실제 효과는 제한적일 수 있음)")
                            except Exception as e:
                                self.logger.debug(f"FAISS GPU 변환 실패 (정상): {str(e)}")
                except Exception as e:
                    self.logger.debug(f"FAISS GPU 처리 시도 (정상): {str(e)}")
                    # LangChain FAISS의 GPU 지원이 제한적이므로, 에러를 경고로만 처리
            
            # 인덱스 저장
            index_path = self.faiss_storage_path / index_name
            if self._faiss_supports_dangerous_deserialization:
                self.faiss_store.save_local(
                    str(index_path),
                    allow_dangerous_deserialization=True
                )
            else:
                # 파라미터를 지원하지 않는 경우 파라미터 없이 저장 시도
                try:
                    self.faiss_store.save_local(str(index_path))
                except TypeError:
                    # 그래도 실패하면 다른 방법 시도 (fallback)
                    self.faiss_store.save_local(
                        str(index_path),
                        allow_dangerous_deserialization=True
                    )
            
            self.logger.info(f"FAISS 인덱스 생성 및 저장 완료: {index_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"FAISS 인덱스 생성 실패: {str(e)}")
            import traceback
            self.logger.error(f"상세 오류: {traceback.format_exc()}")
            return False
    
    def load_faiss_index(self, index_name: str = "default") -> bool:
        """
        저장된 FAISS 인덱스 로드
        
        Args:
            index_name: 인덱스 이름
            
        Returns:
            로드 성공 여부
        """
        try:
            index_path = self.faiss_storage_path / index_name
            
            if not index_path.exists():
                self.logger.debug(f"FAISS 인덱스 파일이 없습니다: {index_path}")
                return False
            
            # allow_dangerous_deserialization 파라미터 조건부 처리
            # load_local은 일반적으로 allow_dangerous_deserialization=True가 필요함
            if self._faiss_load_supports_dangerous:
                try:
                    self.faiss_store = FAISS.load_local(
                        str(index_path),
                        self.embedding_function,
                        allow_dangerous_deserialization=True
                    )
                except TypeError as e:
                    # 파라미터를 지원하지 않거나 다른 문제가 있는 경우
                    self.logger.debug(f"allow_dangerous_deserialization 파라미터 없이 재시도: {str(e)}")
                    try:
                        self.faiss_store = FAISS.load_local(
                            str(index_path),
                            self.embedding_function
                        )
                    except Exception as e2:
                        # 최신 LangChain은 allow_dangerous_deserialization=True가 필수
                        self.logger.error(f"FAISS 로드 실패 (allow_dangerous_deserialization 필요): {str(e2)}")
                        raise
            else:
                # 파라미터를 지원하지 않는 경우 파라미터 없이 로드 시도
                try:
                    self.faiss_store = FAISS.load_local(
                        str(index_path),
                        self.embedding_function
                    )
                except Exception as e:
                    # 최신 버전에서는 allow_dangerous_deserialization이 필요할 수 있음
                    self.logger.debug(f"allow_dangerous_deserialization=True로 재시도: {str(e)}")
                    try:
                        self.faiss_store = FAISS.load_local(
                            str(index_path),
                            self.embedding_function,
                            allow_dangerous_deserialization=True
                        )
                    except Exception as e2:
                        self.logger.error(f"FAISS 로드 실패: {str(e2)}")
                        raise
            
            # GPU 사용 시 로드된 인덱스를 GPU로 이동
            if self.faiss_use_gpu and self._faiss_gpu_resource is not None and self.faiss_store is not None:
                try:
                    import faiss
                    if hasattr(self.faiss_store, '_index') and self.faiss_store._index is not None:
                        # CPU 인덱스를 GPU로 변환
                        gpu_index = faiss.index_cpu_to_gpu(self._faiss_gpu_resource, 0, self.faiss_store._index)
                        self.faiss_store._index = gpu_index
                        self.logger.info("로드된 FAISS 인덱스를 GPU로 이동 완료")
                except Exception as e:
                    self.logger.warning(f"FAISS GPU 변환 실패: {str(e)}. CPU로 계속 사용합니다.")
            
            self.logger.info(f"FAISS 인덱스 로드 완료: {index_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"FAISS 인덱스 로드 실패: {str(e)}")
            return False
    
    def initialize_bm25_from_chunks(
        self,
        chunks: List[DocumentChunk],
        k: int = 4
    ) -> bool:
        """
        DocumentChunk 리스트로부터 BM25Retriever 생성
        
        Args:
            chunks: DocumentChunk 리스트
            k: 반환할 문서 수
            
        Returns:
            생성 성공 여부
        """
        try:
            self.logger.info(f"BM25Retriever 생성 시작: {len(chunks)}개 청크")
            
            # DocumentChunk를 LangChain Document로 변환
            documents = [document_chunk_to_langchain_document(chunk) for chunk in chunks]
            
            # BM25Retriever 생성
            self.bm25_retriever = BM25Retriever.from_documents(
                documents=documents,
                preprocess_func=self.bm25_preprocess_func,
                k=k
            )
            
            self.logger.info(f"BM25Retriever 생성 완료: {len(documents)}개 문서")
            
            # 자동 저장
            self.save_bm25_index()
            
            return True
            
        except Exception as e:
            self.logger.error(f"BM25Retriever 생성 실패: {str(e)}")
            import traceback
            self.logger.error(f"상세 오류: {traceback.format_exc()}")
            return False
    
    def save_bm25_index(self, index_name: str = "default") -> bool:
        """
        BM25Retriever 인덱스 저장 (문서 리스트를 pickle로 저장)
        
        Args:
            index_name: 인덱스 이름
            
        Returns:
            저장 성공 여부
        """
        if self.bm25_retriever is None:
            self.logger.debug("BM25Retriever가 초기화되지 않아 저장할 수 없습니다.")
            return False
        
        try:
            index_file = self.bm25_storage_path / f"{index_name}.pkl"
            
            # BM25Retriever의 문서 리스트 저장
            data = {
                'documents': self.bm25_retriever.docs,
                'preprocess_func_name': self.bm25_preprocess_func.__name__ if hasattr(self.bm25_preprocess_func, '__name__') else None,
                'k': self.bm25_retriever.k
            }
            
            with open(index_file, 'wb') as f:
                pickle.dump(data, f)
            
            self.logger.info(f"BM25 인덱스 저장 완료: {index_file} (문서 {len(self.bm25_retriever.docs)}개)")
            return True
            
        except Exception as e:
            self.logger.error(f"BM25 인덱스 저장 실패: {str(e)}")
            import traceback
            self.logger.error(f"상세 오류: {traceback.format_exc()}")
            return False
    
    def load_bm25_index(self, index_name: str = "default", k: int = 4) -> bool:
        """
        저장된 BM25 인덱스 로드 (문서 리스트를 pickle로 로드하여 BM25Retriever 재구축)
        
        Args:
            index_name: 인덱스 이름
            k: 반환할 문서 수 (저장된 값이 있으면 우선 사용)
            
        Returns:
            로드 성공 여부
        """
        try:
            index_file = self.bm25_storage_path / f"{index_name}.pkl"
            
            if not index_file.exists():
                self.logger.debug(f"BM25 인덱스 파일이 없습니다: {index_file}")
                return False
            
            with open(index_file, 'rb') as f:
                data = pickle.load(f)
            
            documents = data.get('documents', [])
            saved_k = data.get('k', k)
            
            if not documents:
                self.logger.warning(f"BM25 인덱스 파일은 존재하지만 문서가 없습니다: {index_file}")
                return False
            
            # 저장된 k 값 사용 (없으면 파라미터 값 사용)
            k = saved_k if saved_k else k
            
            # BM25Retriever 재구축
            self.bm25_retriever = BM25Retriever.from_documents(
                documents=documents,
                preprocess_func=self.bm25_preprocess_func,
                k=k
            )
            
            self.logger.info(f"BM25 인덱스 로드 완료: {index_file} (문서 {len(documents)}개)")
            return True
            
        except Exception as e:
            self.logger.error(f"BM25 인덱스 로드 실패: {str(e)}")
            import traceback
            self.logger.error(f"상세 오류: {traceback.format_exc()}")
            return False
    
    def create_ensemble_retriever(
        self,
        faiss_weight: float = 0.7,
        bm25_weight: float = 0.3,
        c: int = 60,
        k: int = 10
    ) -> Optional[EnsembleRetriever]:
        """
        EnsembleRetriever 생성 (FAISS + BM25)
        
        Args:
            faiss_weight: FAISS 검색 가중치
            bm25_weight: BM25 검색 가중치
            c: RRF 상수
            k: 반환할 문서 수
            
        Returns:
            EnsembleRetriever 객체 또는 None
        """
        if self.faiss_store is None:
            self.logger.warning("FAISS 저장소가 초기화되지 않았습니다.")
            return None
        
        if self.bm25_retriever is None:
            self.logger.warning("BM25Retriever가 초기화되지 않았습니다.")
            return None
        
        try:
            # FAISS Retriever 생성
            faiss_retriever = self.faiss_store.as_retriever(
                search_kwargs={"k": k}
            )
            
            # BM25 Retriever 설정 업데이트
            self.bm25_retriever.k = k
            
            # EnsembleRetriever 생성
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[faiss_retriever, self.bm25_retriever],
                weights=[faiss_weight, bm25_weight],
                c=c
            )
            
            self.logger.info(
                f"EnsembleRetriever 생성 완료: "
                f"weights=[{faiss_weight}, {bm25_weight}], c={c}, k={k}"
            )
            
            return self.ensemble_retriever
            
        except Exception as e:
            self.logger.error(f"EnsembleRetriever 생성 실패: {str(e)}")
            import traceback
            self.logger.error(f"상세 오류: {traceback.format_exc()}")
            return None
    
    def search_with_ensemble(
        self,
        query: str,
        k: int = 10,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        EnsembleRetriever를 사용하여 검색
        
        Args:
            query: 검색 쿼리
            k: 반환할 문서 수
            score_threshold: 최소 점수 임계값
            
        Returns:
            검색 결과 리스트 (기존 형식)
        """
        if self.ensemble_retriever is None:
            self.logger.warning("EnsembleRetriever가 초기화되지 않았습니다.")
            return []
        
        try:
            # EnsembleRetriever 검색
            documents = self.ensemble_retriever.invoke(query)
            
            # LangChain Document를 기존 형식으로 변환
            # EnsembleRetriever는 RRF 점수 순서로 정렬된 결과를 반환
            # 순위를 점수로 변환 (높은 순위 = 높은 점수)
            results = []
            total_docs = len(documents)
            
            for rank, doc in enumerate(documents, 1):
                result = langchain_document_to_dict(doc)
                
                # RRF 점수 기반 점수 계산 (순위가 높을수록 점수 높음)
                # 점수 범위: 0.9 ~ 0.1 (순위에 따라 감소)
                rrf_score = 0.9 - (rank - 1) * (0.8 / max(total_docs, 1))
                result['score'] = max(0.1, rrf_score)
                
                # score_threshold 필터링
                if score_threshold is None or result['score'] >= score_threshold:
                    results.append(result)
            
            # k개로 제한
            results = results[:k]
            
            self.logger.debug(f"EnsembleRetriever 검색 완료: 쿼리='{query}', 결과={len(results)}개")
            return results
            
        except Exception as e:
            self.logger.error(f"EnsembleRetriever 검색 실패: {str(e)}")
            import traceback
            self.logger.error(f"상세 오류: {traceback.format_exc()}")
            return []
    
    def search_with_faiss_only(
        self,
        query: str,
        k: int = 10,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        FAISS만 사용하여 검색
        
        Args:
            query: 검색 쿼리
            k: 반환할 문서 수
            score_threshold: 최소 점수 임계값
            
        Returns:
            검색 결과 리스트 (기존 형식)
        """
        if self.faiss_store is None:
            self.logger.warning("FAISS 저장소가 초기화되지 않았습니다.")
            return []
        
        try:
            # FAISS 유사도 검색 (점수 포함)
            self.logger.debug(f"FAISS similarity_search_with_score 호출: query='{query[:50]}...', k={k}")
            docs_with_scores = self.faiss_store.similarity_search_with_score(
                query=query,
                k=k
            )
            
            self.logger.debug(f"FAISS 검색된 문서 수: {len(docs_with_scores)}개")
            
            # 결과 변환
            results = []
            filtered_count = 0
            for doc, score in docs_with_scores:
                # 코사인 거리를 유사도 점수로 변환 (0~1 범위)
                # FAISS는 코사인 거리(0~2)를 반환하므로, 유사도는 1 - (distance/2) 또는 1/(1+distance)로 변환
                # 코사인 거리: 0에 가까울수록 유사, 2에 가까울수록 다름
                # numpy 타입을 Python float로 변환 (JSON 직렬화 오류 방지)
                score_float = float(score)
                similarity_score = 1.0 - (score_float / 2.0) if score_float <= 2.0 else 0.0
                similarity_score = max(0.0, min(1.0, similarity_score))  # 0~1 범위로 제한
                similarity_score = float(similarity_score)  # Python float로 명시적 변환
                
                result = langchain_document_to_dict(doc)
                result['score'] = similarity_score
                
                # score_threshold 필터링
                if score_threshold is None or similarity_score >= score_threshold:
                    results.append(result)
                else:
                    filtered_count += 1
            
            if filtered_count > 0:
                self.logger.debug(f"FAISS 검색: {filtered_count}개 결과가 threshold={score_threshold}로 필터링됨")
            
            self.logger.info(f"FAISS 검색 완료: 쿼리='{query[:50]}...', 검색된 문서={len(docs_with_scores)}개, 최종 결과={len(results)}개 (threshold={score_threshold})")
            return results
            
        except Exception as e:
            self.logger.error(f"FAISS 검색 실패: {str(e)}")
            import traceback
            self.logger.error(f"상세 오류: {traceback.format_exc()}")
            return []
    
    def search_with_bm25_only(
        self,
        query: str,
        k: int = 10,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        BM25만 사용하여 검색
        
        Args:
            query: 검색 쿼리
            k: 반환할 문서 수
            score_threshold: 최소 점수 임계값
            
        Returns:
            검색 결과 리스트 (기존 형식)
        """
        if self.bm25_retriever is None:
            self.logger.warning("BM25Retriever가 초기화되지 않았습니다.")
            return []
        
        try:
            # BM25 검색 (k 파라미터는 retriever 초기화 시 설정됨)
            # invoke 메서드는 쿼리만 받음
            self.bm25_retriever.k = k  # 검색 전에 k 설정
            documents = self.bm25_retriever.invoke(query)
            
            # 결과 변환
            results = []
            max_score = 0.0
            
            # BM25 점수 추정 (순위 기반)
            for rank, doc in enumerate(documents, 1):
                result = langchain_document_to_dict(doc)
                
                # BM25 점수를 유사도 점수로 변환 (순위 기반, 0~1 범위)
                # 높은 순위일수록 높은 점수 (1.0 ~ 0.3 범위)
                bm25_score = max(0.3, 1.0 - (rank - 1) * 0.1)
                result['score'] = bm25_score
                
                if bm25_score > max_score:
                    max_score = bm25_score
                
                # score_threshold 필터링
                if score_threshold is None or bm25_score >= score_threshold:
                    results.append(result)
            
            # 점수 정규화 (최고 점수를 1.0으로)
            if max_score > 0 and results:
                for result in results:
                    result['score'] = result['score'] / max_score
            
            self.logger.debug(f"BM25 검색 완료: 쿼리='{query}', 결과={len(results)}개")
            return results
            
        except Exception as e:
            self.logger.error(f"BM25 검색 실패: {str(e)}")
            import traceback
            self.logger.error(f"상세 오류: {traceback.format_exc()}")
            return []
    
    def add_documents_to_faiss(self, chunks: List[DocumentChunk]) -> bool:
        """
        FAISS에 문서 추가
        
        Args:
            chunks: 추가할 DocumentChunk 리스트
            
        Returns:
            추가 성공 여부
        """
        if self.faiss_store is None:
            self.logger.warning("FAISS 저장소가 초기화되지 않았습니다.")
            return False
        
        try:
            documents = [document_chunk_to_langchain_document(chunk) for chunk in chunks]
            self.faiss_store.add_documents(documents)
            
            # FAISS 인덱스 저장 (추가된 문서 반영)
            try:
                index_path = self.faiss_storage_path / "default"
                if self._faiss_supports_dangerous_deserialization:
                    self.faiss_store.save_local(
                        str(index_path),
                        allow_dangerous_deserialization=True
                    )
                else:
                    try:
                        self.faiss_store.save_local(str(index_path))
                    except TypeError:
                        self.faiss_store.save_local(
                            str(index_path),
                            allow_dangerous_deserialization=True
                        )
                self.logger.debug(f"FAISS 인덱스 저장 완료: {index_path}")
            except Exception as save_error:
                self.logger.warning(f"FAISS 인덱스 저장 실패 (문서는 추가됨): {str(save_error)}")
            
            self.logger.info(f"FAISS에 문서 추가 완료: {len(chunks)}개")
            return True
            
        except Exception as e:
            self.logger.error(f"FAISS 문서 추가 실패: {str(e)}")
            return False
    
    def add_documents_to_bm25(self, chunks: List[DocumentChunk]) -> bool:
        """
        BM25Retriever에 문서 추가
        
        Args:
            chunks: 추가할 DocumentChunk 리스트
            
        Returns:
            추가 성공 여부
        """
        if self.bm25_retriever is None:
            self.logger.warning("BM25Retriever가 초기화되지 않았습니다.")
            return False
        
        try:
            documents = [document_chunk_to_langchain_document(chunk) for chunk in chunks]
            
            # BM25Retriever는 인메모리이므로 새로 재구축해야 함
            # 기존 문서와 새 문서 합치기
            all_documents = self.bm25_retriever.docs + documents
            
            # BM25Retriever 재생성
            self.bm25_retriever = BM25Retriever.from_documents(
                documents=all_documents,
                preprocess_func=self.bm25_preprocess_func,
                k=self.bm25_retriever.k
            )
            
            self.logger.info(f"BM25Retriever 문서 추가 완료: {len(chunks)}개")
            
            # 자동 저장
            self.save_bm25_index()
            
            return True
            
        except Exception as e:
            self.logger.error(f"BM25Retriever 문서 추가 실패: {str(e)}")
            return False
    
    def delete_documents_by_source(self, source_file: str) -> bool:
        """
        특정 파일의 문서를 FAISS 및 BM25에서 삭제
        
        Args:
            source_file: 삭제할 문서의 소스 파일 경로
            
        Returns:
            삭제 성공 여부
        """
        deleted_count = 0
        
        # FAISS에서 삭제
        if self.faiss_store is not None:
            try:
                # FAISS는 직접 삭제를 지원하지 않으므로 재구축 필요
                # 현재는 문서 삭제 기능이 제한적이므로, 전체 재구축이 필요할 수 있음
                # 하지만 일단 로그만 남기고 실제 삭제는 재구축 시 처리
                self.logger.warning(
                    f"FAISS에서 특정 파일 삭제는 지원되지 않습니다. "
                    f"전체 재구축이 필요합니다: {source_file}"
                )
            except Exception as e:
                self.logger.error(f"FAISS 문서 삭제 실패: {str(e)}")
        
        # BM25에서 삭제
        if self.bm25_retriever is not None:
            try:
                # source_file과 일치하는 문서만 필터링
                remaining_documents = [
                    doc for doc in self.bm25_retriever.docs
                    if doc.metadata.get('source_file', '') != source_file
                ]
                
                deleted_count = len(self.bm25_retriever.docs) - len(remaining_documents)
                
                if deleted_count > 0:
                    # BM25Retriever 재생성
                    self.bm25_retriever = BM25Retriever.from_documents(
                        documents=remaining_documents,
                        preprocess_func=self.bm25_preprocess_func,
                        k=self.bm25_retriever.k
                    )
                    
                    # 자동 저장
                    self.save_bm25_index()
                    
                    self.logger.info(f"BM25에서 문서 삭제 완료: {source_file} ({deleted_count}개)")
                else:
                    self.logger.debug(f"BM25에서 삭제할 문서가 없습니다: {source_file}")
                
            except Exception as e:
                self.logger.error(f"BM25 문서 삭제 실패: {str(e)}")
                return False
        
        return deleted_count > 0
    
    def validate_indexes(self, qdrant_document_count: Optional[int] = None) -> Dict[str, Any]:
        """
        인덱스 상태 검증 (Qdrant 문서 수와 FAISS/BM25 인덱스 문서 수 비교)
        
        Args:
            qdrant_document_count: Qdrant의 문서 수 (None이면 비교하지 않음)
            
        Returns:
            검증 결과 딕셔너리
        """
        result = {
            'faiss_available': False,
            'faiss_document_count': 0,
            'bm25_available': False,
            'bm25_document_count': 0,
            'qdrant_document_count': qdrant_document_count,
            'is_consistent': True,
            'warnings': []
        }
        
        # FAISS 검증
        if self.faiss_store is not None:
            try:
                # FAISS에서 문서 수 확인
                # FAISS는 _index 속성에 인덱스가 있고, ntotal으로 문서 수를 알 수 있음
                if hasattr(self.faiss_store, '_index') and self.faiss_store._index is not None:
                    result['faiss_available'] = True
                    result['faiss_document_count'] = self.faiss_store._index.ntotal
            except Exception as e:
                result['warnings'].append(f"FAISS 문서 수 확인 실패: {str(e)}")
        
        # BM25 검증
        if self.bm25_retriever is not None:
            try:
                result['bm25_available'] = True
                result['bm25_document_count'] = len(self.bm25_retriever.docs)
            except Exception as e:
                result['warnings'].append(f"BM25 문서 수 확인 실패: {str(e)}")
        
        # 일관성 검증
        if qdrant_document_count is not None:
            if result['faiss_available']:
                if result['faiss_document_count'] != qdrant_document_count:
                    result['is_consistent'] = False
                    result['warnings'].append(
                        f"FAISS 문서 수({result['faiss_document_count']})와 "
                        f"Qdrant 문서 수({qdrant_document_count})가 일치하지 않습니다."
                    )
            
            if result['bm25_available']:
                if result['bm25_document_count'] != qdrant_document_count:
                    result['is_consistent'] = False
                    result['warnings'].append(
                        f"BM25 문서 수({result['bm25_document_count']})와 "
                        f"Qdrant 문서 수({qdrant_document_count})가 일치하지 않습니다."
                    )
        
        # FAISS와 BM25 간 일관성 검증
        if result['faiss_available'] and result['bm25_available']:
            if result['faiss_document_count'] != result['bm25_document_count']:
                result['is_consistent'] = False
                result['warnings'].append(
                    f"FAISS 문서 수({result['faiss_document_count']})와 "
                    f"BM25 문서 수({result['bm25_document_count']})가 일치하지 않습니다."
                )
        
        return result

