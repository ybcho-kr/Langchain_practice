"""
벡터 저장소 모듈
BGE-m3 기반 Qdrant 하이브리드 검색 (Dense + Sparse)
"""

from typing import List, Dict, Any, Optional
import uuid
import json
import os
import re
from pathlib import Path
import numpy as np

from qdrant_client import QdrantClient
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams, SparseIndexParams, Filter, FieldCondition, 
    MatchValue, Query, NamedSparseVector, Prefetch, SparseVector as QdrantSparseVector,
    PointStruct, FusionQuery, Fusion
)
from FlagEmbedding import BGEM3FlagModel

from src.utils.logger import get_logger
from src.utils.config import get_qdrant_config, get_embedding_config
from src.modules.document_processor import DocumentChunk
from src.modules.langchain_retrievers import LangChainRetrievalManager
from src.modules.kiwipiepy_preprocessor import KiwipiepyPreprocessor


class QdrantVectorStore:
    """BGE-m3 기반 Qdrant 하이브리드 벡터 저장소 (Dense + Sparse)"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, bge_model: Optional[BGEM3FlagModel] = None):
        """
        Args:
            config: Qdrant 설정
            bge_model: 기존 BGE-m3 모델 인스턴스 (선택적, 중복 로드 방지용)
        """
        self.logger = get_logger()
        
        if config is None:
            config = get_qdrant_config()
        
        self.collection_name = config.collection_name
        self.vector_size = config.vector_size  # BGE-m3는 1024차원
        self.distance_metric = Distance.COSINE if config.distance_metric.lower() == 'cosine' else Distance.EUCLIDEAN
        self.storage_path = config.storage_path
        self.use_local_storage = config.use_local_storage
        
        # Sparse 벡터 설정
        self.sparse_enabled = getattr(config, 'sparse_enabled', True)
        self.sparse_vector_name = getattr(config, 'sparse_vector_name', 'sparse')
        self.dense_vector_name = 'dense'  # BGE-m3 dense 벡터 이름
        self.hybrid_search_dense_weight = getattr(config, 'hybrid_search_dense_weight', 0.7)
        self.hybrid_search_sparse_weight = getattr(config, 'hybrid_search_sparse_weight', 0.3)
        # Pydantic 모델에서는 직접 속성 접근 (getattr는 속성이 없을 때만 기본값 사용)
        self.use_kiwipiepy_preprocessing = config.sparse_use_kiwipiepy if hasattr(config, 'sparse_use_kiwipiepy') else True
        self.kiwipiepy_dictionary_path = config.kiwipiepy_dictionary_path if hasattr(config, 'kiwipiepy_dictionary_path') else None
        self.logger.info(f"Kiwipiepy 설정 로드: use_kiwipiepy_preprocessing={self.use_kiwipiepy_preprocessing}, dictionary_path={self.kiwipiepy_dictionary_path}")
        
        # KIWIPIEPY_AVAILABLE 상태 확인
        from src.modules.kiwipiepy_preprocessor import KIWIPIEPY_AVAILABLE
        self.logger.info(f"Kiwipiepy 모듈 사용 가능 여부: KIWIPIEPY_AVAILABLE={KIWIPIEPY_AVAILABLE}")
        
        self.kiwipiepy_preprocessor: Optional[KiwipiepyPreprocessor] = None
        if self.use_kiwipiepy_preprocessing:
            try:
                self.kiwipiepy_preprocessor = KiwipiepyPreprocessor(
                    use_kiwipiepy=True,
                    dictionary_path=self.kiwipiepy_dictionary_path,
                )
                self.logger.info(f"KiwipiepyPreprocessor 생성 완료: use_kiwipiepy={self.kiwipiepy_preprocessor.use_kiwipiepy}, kiwi={self.kiwipiepy_preprocessor.kiwi is not None}")
                if not self.kiwipiepy_preprocessor.use_kiwipiepy:
                    self.logger.warning("Kiwipiepy 전처리를 사용할 수 없어 기능을 비활성화합니다. (Kiwipiepy가 설치되어 있지 않거나 초기화에 실패했습니다)")
                    self.use_kiwipiepy_preprocessing = False
                else:
                    self.logger.info("Kiwipiepy 형태소 전처리가 활성화되었습니다.")
            except Exception as exc:
                self.logger.warning(f"Kiwipiepy 전처리 초기화 실패: {exc}")
                import traceback
                self.logger.error(f"Kiwipiepy 초기화 상세 오류: {traceback.format_exc()}")
                self.use_kiwipiepy_preprocessing = False
        
        # 검색 기본값 설정
        self.default_limit = config.default_limit
        self.max_scroll_limit = config.max_scroll_limit
        self.connection_timeout = config.connection_timeout
        self.request_timeout = config.request_timeout
        
        # 로컬 파일 시스템 사용
        if self.use_local_storage:
            from pathlib import Path
            storage_path = Path(self.storage_path)
            storage_path.mkdir(parents=True, exist_ok=True)
            
            self.client = QdrantClient(path=str(storage_path))
            # 비동기 클라이언트는 지연 초기화 (로컬 저장소 동시 접근 문제 방지)
            self.async_client = None
            self._async_client_path = str(storage_path)
            self._async_client_host = None
            self._async_client_port = None
            self.logger.info(f"Qdrant 로컬 저장소 초기화: {storage_path}")
        else:
            self.client = QdrantClient(host=config.host, port=config.port)
            # 비동기 클라이언트는 지연 초기화
            self.async_client = None
            self._async_client_path = None
            self._async_client_host = config.host
            self._async_client_port = config.port
            self.logger.info(f"Qdrant 서버 클라이언트 초기화: {config.host}:{config.port}")
        
        # BGE-m3 모델 초기화
        if bge_model is not None:
            self.bge_model = bge_model
            self.logger.info("기존 BGE-m3 모델 인스턴스 재사용 (중복 로드 방지)")
        else:
            from src.utils.config import get_embedding_config
            embedding_config = get_embedding_config()
            
            # BGE-m3 모델 경로 확인
            model_path = embedding_config.model_path or embedding_config.name
            if not model_path:
                raise ValueError(
                    "BGE-m3 모델 경로가 설정되지 않았습니다. "
                    "config.yaml의 model.embedding.model_path를 BGE-m3 모델 경로로 설정하세요. "
                    "예: C:\\Users\\a003219048\\Desktop\\models\\BGE-m3-ko"
                )
            
            # 모델 경로 존재 확인
            if not os.path.exists(model_path):
                self.logger.warning(f"BGE-m3 모델 경로가 존재하지 않습니다: {model_path}")
                self.logger.warning("모델을 다운로드하거나 경로를 확인하세요.")
                self.logger.warning("BGE-m3 모델은 FlagEmbedding 라이브러리를 통해 자동 다운로드될 수 있습니다.")
            else:
                # 모델 디렉토리 내부 파일 확인 (config.json, tokenizer 등)
                config_file = os.path.join(model_path, "config.json")
                if not os.path.exists(config_file):
                    self.logger.warning(f"BGE-m3 모델 디렉토리 내 config.json을 찾을 수 없습니다: {model_path}")
                    self.logger.warning("모델이 완전히 다운로드되지 않았을 수 있습니다.")
                else:
                    self.logger.info(f"BGE-m3 모델 경로 검증 완료: {model_path}")
            
            # GPU 사용 가능 여부 확인
            use_fp16 = True
            try:
                import torch
                if not torch.cuda.is_available():
                    use_fp16 = False
                    self.logger.info("CUDA를 사용할 수 없어 FP16을 비활성화합니다.")
            except ImportError:
                use_fp16 = False
                self.logger.info("PyTorch를 찾을 수 없어 FP16을 비활성화합니다.")
            
            self.logger.info(f"BGE-m3 모델 초기화 중: {model_path} (FP16: {use_fp16})")
            try:
                # transformers 라이브러리 경고 억제 (XLMRobertaTokenizerFast 관련)
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*XLMRobertaTokenizerFast.*")
                    warnings.filterwarnings("ignore", message=".*fast tokenizer.*")
                    # transformers 라이브러리의 경고도 억제
                    import transformers
                    transformers.logging.set_verbosity_error()
                    
                    self.bge_model = BGEM3FlagModel(model_path, use_fp16=use_fp16)
                
                self.logger.info("BGE-m3 모델 초기화 완료")
            except Exception as e:
                self.logger.error(f"BGE-m3 모델 초기화 실패: {str(e)}")
                raise
    
    def _get_async_client(self):
        """비동기 클라이언트 지연 초기화 (로컬 저장소 동시 접근 문제 방지)"""
        if self.async_client is None:
            if self.use_local_storage:
                # 로컬 저장소의 경우 AsyncQdrantLocal은 동기 클라이언트와 동시 접근 불가
                # 따라서 None을 반환하여 asyncio.to_thread 사용을 유도
                return None
            else:
                # 서버 모드의 경우 비동기 클라이언트 생성
                self.async_client = AsyncQdrantClient(host=self._async_client_host, port=self._async_client_port)
        return self.async_client
    
    def _check_connection(self) -> bool:
        """연결 상태 확인"""
        try:
            collections = self.client.get_collections()
            self.logger.debug(f"Qdrant 연결 확인: {len(collections.collections)}개 컬렉션")
            return True
        except Exception as e:
            self.logger.error(f"Qdrant 연결 실패: {str(e)}")
            return False
    
    def _get_metadata(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """메타데이터 추출 헬퍼 메서드 (중복 제거)"""
        if 'metadata' in payload:
            return payload['metadata']
        return payload
    
    def create_collection(self, force_recreate: bool = False) -> bool:
        """컬렉션 생성 (BGE-m3 기반 Dense + Sparse 벡터)"""
        try:
            self.logger.info(f"컬렉션 생성 시작: {self.collection_name}")
            
            # 기존 컬렉션 확인
            self.logger.info("기존 컬렉션 목록 조회 중...")
            collections = self.client.get_collections()
            self.logger.info(f"기존 컬렉션 수: {len(collections.collections)}")
            
            collection_exists = any(col.name == self.collection_name for col in collections.collections)
            self.logger.info(f"컬렉션 존재 여부: {collection_exists}")
            
            if collection_exists:
                if force_recreate:
                    self.logger.info(f"기존 컬렉션 삭제: {self.collection_name}")
                    self.client.delete_collection(self.collection_name)
                else:
                    self.logger.info(f"컬렉션이 이미 존재합니다: {self.collection_name}")
                    return True
            
            # 새 컬렉션 생성
            self.logger.info(f"새 컬렉션 생성 중: {self.collection_name}, 벡터 크기: {self.vector_size}")
            
            # Dense 벡터 설정 (BGE-m3는 1024차원)
            vectors_config = {
                self.dense_vector_name: VectorParams(
                    size=self.vector_size,  # 1024
                    distance=self.distance_metric
                )
            }
            
            # Sparse 벡터 설정 (sparse_enabled일 때만)
            sparse_vectors_config = None
            if self.sparse_enabled:
                sparse_vectors_config = {
                    self.sparse_vector_name: SparseVectorParams(
                        index=SparseIndexParams()
                    )
                }
                self.logger.info(f"Sparse 벡터 설정 추가: {self.sparse_vector_name}")
            
            # 컬렉션 생성
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=vectors_config,
                sparse_vectors_config=sparse_vectors_config
            )
            
            self.logger.info(f"컬렉션 생성 완료: {self.collection_name} (Dense: {self.dense_vector_name}, Sparse: {self.sparse_vector_name if self.sparse_enabled else 'N/A'})")
            
            return True
            
        except Exception as e:
            self.logger.error(f"컬렉션 생성 실패: {str(e)}")
            import traceback
            self.logger.error(f"상세 오류: {traceback.format_exc()}")
            return False
    
    def add_documents(self, documents: List[DocumentChunk], force_update: bool = False) -> bool:
        """문서 추가 (BGE-m3로 dense+sparse 임베딩 생성 후 Qdrant에 직접 저장)"""
        if not self._check_connection():
            return False
        
        try:
            # 중복 청크 방지
            seen_chunks = set()
            valid_documents = []
            
            for doc in documents:
                # source_file 검증
                if not doc.source_file or doc.source_file.strip() == '':
                    self.logger.warning(f"source_file이 비어있는 청크를 건너뜁니다. chunk_id: {doc.chunk_id}")
                    continue
                
                # 청크 고유 식별자 생성 (파일명 + 청크 인덱스)
                chunk_key = f"{doc.source_file}:{doc.chunk_index}"
                    
                if chunk_key in seen_chunks:
                    self.logger.warning(f"중복 청크 건너뛰기: {chunk_key}")
                    continue
                
                seen_chunks.add(chunk_key)
                valid_documents.append(doc)
            
            if not valid_documents:
                self.logger.warning("추가할 유효한 문서가 없습니다.")
                return False
            
            # BGE-m3로 임베딩 생성 (배치 처리)
            self.logger.info(f"BGE-m3로 dense + sparse 임베딩 생성 중: {len(valid_documents)}개 문서")
            texts = [doc.content for doc in valid_documents]
            # 전처리 없이 원본 텍스트를 그대로 사용
            embedding_texts = texts
            
            # 배치 크기 설정 (메모리 효율성)
            batch_size = getattr(self, '_bge_batch_size', 32)
            all_embeddings = {'dense': [], 'lexical_weights': []}
            
            for i in range(0, len(texts), batch_size):
                batch_texts = embedding_texts[i:i+batch_size]
                batch_embeddings = self.bge_model.encode(
                    batch_texts,
                    return_dense=True,
                    return_sparse=True,
                    return_colbert_vecs=False
                )
                
                # 반환 키 확인 및 처리
                if 'dense_vecs' in batch_embeddings:
                    dense_key = 'dense_vecs'
                elif 'dense' in batch_embeddings:
                    dense_key = 'dense'
                else:
                    self.logger.error(f"Available keys: {batch_embeddings.keys()}")
                    raise KeyError("Cannot find dense embeddings in BGE-m3 output")
                
                all_embeddings['dense'].extend(batch_embeddings[dense_key])
                
                # lexical_weights 확인 및 로깅
                batch_lexical_weights = batch_embeddings.get('lexical_weights', [])
                if not batch_lexical_weights:
                    self.logger.warning(f"배치 {i//batch_size + 1}에서 lexical_weights가 없습니다. BGE-m3 반환값 키: {list(batch_embeddings.keys())}")
                else:
                    # 빈 sparse 벡터 개수 확인
                    empty_count = sum(1 for w in batch_lexical_weights if not w or (isinstance(w, dict) and len(w) == 0))
                    if empty_count > 0:
                        self.logger.warning(f"배치 {i//batch_size + 1}에서 {empty_count}/{len(batch_lexical_weights)}개 청크의 sparse 벡터가 비어있습니다")
                
                all_embeddings['lexical_weights'].extend(batch_lexical_weights)
                self.logger.debug(f"  진행: {min(i+batch_size, len(texts))}/{len(texts)}")
            
            self.logger.info("BGE-m3 임베딩 생성 완료")
            
            # Qdrant에 업로드
            self.logger.info("문서를 Qdrant에 업로드 중...")
            points = []
            
            for idx, (doc, dense_vec, sparse_dict) in enumerate(zip(valid_documents, all_embeddings['dense'], all_embeddings['lexical_weights'])):
                # Dense 벡터 변환
                if hasattr(dense_vec, 'tolist'):
                    dense_vector = dense_vec.tolist()
                elif isinstance(dense_vec, np.ndarray):
                    dense_vector = dense_vec.tolist()
                else:
                    dense_vector = list(dense_vec)
                
                # Sparse 벡터 변환 (lexical_weights는 딕셔너리 형태)
                sparse_vector = None
                if self.sparse_enabled:
                    if sparse_dict:
                        if isinstance(sparse_dict, dict):
                            # 딕셔너리 형태: {token_id: weight, ...}
                            indices = list(sparse_dict.keys())
                            values = list(sparse_dict.values())
                        else:
                            # 다른 형태인 경우 처리
                            self.logger.warning(f"청크 {doc.chunk_id} 예상치 못한 sparse_dict 형태: {type(sparse_dict)}")
                            indices = []
                            values = []
                        
                        if indices and values:
                            sparse_vector = QdrantSparseVector(
                                indices=indices,
                                values=values
                            )
                            self.logger.debug(f"청크 {doc.chunk_id} sparse 벡터 생성: {len(indices)}개 토큰")
                        else:
                            # 빈 sparse 벡터는 정상일 수 있음 (짧은 텍스트, 특수 문자만 있는 경우 등)
                            self.logger.debug(f"청크 {doc.chunk_id} sparse 벡터가 비어있음: indices={len(indices) if indices else 0}, values={len(values) if values else 0}, sparse_dict 타입={type(sparse_dict)}")
                    else:
                        # sparse_dict가 None이거나 비어있는 경우 (정상일 수 있음)
                        self.logger.debug(f"청크 {doc.chunk_id} sparse_dict가 None이거나 비어있음 (sparse_enabled={self.sparse_enabled})")
                
                # PointStruct 생성
                vector_dict = {
                    self.dense_vector_name: dense_vector
                }
                
                if sparse_vector:
                    vector_dict[self.sparse_vector_name] = sparse_vector
                    self.logger.debug(f"청크 {doc.chunk_id} vector_dict에 sparse 벡터 추가: {list(vector_dict.keys())}")
                else:
                    # sparse 벡터가 없는 경우는 정상일 수 있음 (짧은 텍스트 등)
                    # 경고 대신 DEBUG 레벨로 변경하여 로그 노이즈 감소
                    self.logger.debug(f"청크 {doc.chunk_id} sparse 벡터가 없어 vector_dict에 추가하지 않음 (정상일 수 있음)")
                
                # Point ID를 UUID 문자열로 변환 (Qdrant 로컬 저장소는 UUID 문자열을 요구)
                # chunk_id를 기반으로 일관된 UUID 생성 (uuid5 사용)
                if doc.chunk_id:
                    # chunk_id를 기반으로 UUID 생성 (일관성 유지)
                    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.chunk_id))
                else:
                    # chunk_id가 없으면 새 UUID 생성
                    point_id = str(uuid.uuid4())
                
                point = PointStruct(
                    id=point_id,
                    vector=vector_dict,
                    payload={
                        'page_content': doc.content,
                        'source_file': doc.source_file,
                        'chunk_id': doc.chunk_id,  # 원본 chunk_id는 payload에 저장
                        'chunk_index': doc.chunk_index,
                        'doc_id': doc.doc_id,
                        'section_id': doc.section_id,
                        'chunk_type': doc.chunk_type,
                        'heading_path': doc.heading_path,
                        'page_start': doc.page_start,
                        'page_end': doc.page_end,
                        'language': doc.language,
                        'domain': doc.domain,
                        'embedding_version': doc.embedding_version,
                        'document': doc.doc_metadata.to_dict() if doc.doc_metadata else {},
                        'metadata': {
                            **doc.metadata,
                            'chunk_id': doc.chunk_id,
                            'chunk_index': doc.chunk_index,
                            'doc_id': doc.doc_id,
                            'section_id': doc.section_id,
                            'chunk_type': doc.chunk_type,
                            'heading_path': doc.heading_path,
                            'page_start': doc.page_start,
                            'page_end': doc.page_end,
                            'language': doc.language,
                            'domain': doc.domain,
                            'embedding_version': doc.embedding_version,
                            'document': doc.doc_metadata.to_dict() if doc.doc_metadata else {},
                        },
                        **doc.metadata,
                    }
                )
                points.append(point)
            
            # 배치로 업로드
            upload_batch_size = 100
            for i in range(0, len(points), upload_batch_size):
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points[i:i+upload_batch_size]
                )
                self.logger.debug(f"  업로드: {min(i+upload_batch_size, len(points))}/{len(points)}")
            
            self.logger.info(f"문서 추가 완료: {len(valid_documents)}개 (중복 제거: {len(documents) - len(valid_documents)}개)")
            if self.sparse_enabled:
                self.logger.info("Dense + Sparse 벡터가 함께 저장되었습니다")
            return True
            
        except Exception as e:
            self.logger.error(f"문서 추가 실패: {str(e)}")
            import traceback
            self.logger.error(f"상세 오류: {traceback.format_exc()}")
            return False
    
    def replace_document_vectors(self, file_path: str, new_chunks: List[DocumentChunk]) -> bool:
        """특정 파일의 벡터를 완전히 교체"""
        try:
            self.logger.info(f"파일 벡터 교체 시작: {file_path}")
            
            # 1. 기존 벡터 삭제
            delete_success = self._delete_document_vectors(file_path)
            if not delete_success:
                self.logger.warning(f"기존 벡터 삭제 실패, 새 벡터만 추가: {file_path}")
            
            # 2. 새 벡터 추가
            add_success = self.add_documents(new_chunks, force_update=False)
            
            if add_success:
                self.logger.info(f"파일 벡터 교체 완료: {file_path}, 청크 수: {len(new_chunks)}")
                return True
            else:
                self.logger.error(f"파일 벡터 교체 실패: {file_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"파일 벡터 교체 실패: {file_path}, 오류: {str(e)}")
            return False
    
    def _delete_document_vectors(self, file_path: str) -> tuple:
        """
        특정 파일의 모든 벡터 삭제
        
        Args:
            file_path: 삭제할 파일 경로
            
        Returns:
            (성공 여부, 삭제된 청크 수) 튜플
        """
        try:
            # 삭제 전 청크 수 확인
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,  # 충분히 큰 수
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="source_file",
                            match=MatchValue(value=file_path)
                        )
                    ]
                ),
                with_payload=False,
                with_vectors=False
            )
            
            points_to_delete = scroll_result[0]
            chunk_count = len(points_to_delete)
            
            if chunk_count == 0:
                self.logger.warning(f"삭제할 포인트가 없습니다: {file_path}")
                return True, 0
            
            # Qdrant에서 해당 파일의 모든 포인트 삭제
            # source_file 필드로 필터링 (payload에 저장된 필드)
            result = self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="source_file",
                            match=MatchValue(value=file_path)
                        )
                    ]
                )
            )
            
            self.logger.info(f"파일 벡터 삭제 완료: {file_path}, {chunk_count}개 청크 삭제됨")
            return True, chunk_count
            
        except Exception as e:
            self.logger.error(f"파일 벡터 삭제 실패: {file_path}, 오류: {str(e)}")
            import traceback
            self.logger.error(f"상세 오류: {traceback.format_exc()}")
            return False, 0
    
    # ========== 비동기 메서드 (Phase 2: 벡터 검색 비동기화) ==========
    
    async def search_with_table_filter_async(self, 
                                            query: str, 
                                            table_title: Optional[str] = None,
                                            is_table_data: Optional[bool] = None,
                                            limit: Optional[int] = None,
                                            score_threshold: Optional[float] = None,
                                            dense_weight: Optional[float] = None,
                                            sparse_weight: Optional[float] = None,
                                            keywords: Optional[List[str]] = None,
                                            entities: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """비동기 표 관련 필터와 함께 검색 (BGE-m3 기반 Prefetch + RRF 또는 가중치 기반)"""
        if not self._check_connection():
            return []
        
        # 기본값 적용
        limit = limit if limit is not None else self.default_limit
        if score_threshold is None:
            from src.utils.config import get_rag_config
            rag_config = get_rag_config()
            score_threshold = rag_config.score_threshold
        
        # 필터 조건 구성
        filter_conditions = None
        if table_title or is_table_data is not None:
            must_conditions = []
            
            if table_title:
                must_conditions.append({
                    "key": "table_title",
                    "match": {"value": table_title}
                })
            
            if is_table_data is not None:
                must_conditions.append({
                    "key": "is_table_data",
                    "match": {"value": is_table_data}
                })
            
            if must_conditions:
                filter_conditions = {"must": must_conditions}
        
        # 가중치가 제공되면 가중치 기반 하이브리드 검색 사용
        if dense_weight is not None or sparse_weight is not None:
            # 기본값 적용
            from src.utils.config import get_qdrant_config
            qdrant_config = get_qdrant_config()
            effective_dense_weight = dense_weight if dense_weight is not None else getattr(qdrant_config, 'hybrid_search_dense_weight', 0.7)
            effective_sparse_weight = sparse_weight if sparse_weight is not None else getattr(qdrant_config, 'hybrid_search_sparse_weight', 0.3)
            
            self.logger.info(f"가중치 기반 검색 사용 (표 필터 포함): dense={effective_dense_weight:.2f}, sparse={effective_sparse_weight:.2f}")
            
            # 가중치 기반 하이브리드 검색 호출
            docs_with_scores = await self._hybrid_search_with_weights(
                query=query,
                limit=limit,
                filter_conditions=filter_conditions,
                dense_weight=effective_dense_weight,
                sparse_weight=effective_sparse_weight,
                keywords=keywords,
                entities=entities
            )
            
            # Document 형식에서 Dict 형식으로 변환
            results = []
            for doc, score in docs_with_scores:
                # score_threshold 필터링
                if score_threshold is not None and score < score_threshold:
                    continue
                
                metadata = doc.metadata
                results.append({
                    'content': doc.page_content,
                    'score': score,
                    'metadata': metadata,
                    'source_file': metadata.get('source_file', ''),
                    'chunk_index': metadata.get('chunk_index', 0),
                    'chunk_id': metadata.get('chunk_id', ''),
                    'table_title': metadata.get('table_title', ''),
                    'is_table_data': metadata.get('is_table_data', False)
                })
            
            return results
        
        # 가중치가 없으면 기존 RRF 방식 사용
        try:
            # 필터 조건 구성 (RRF 방식용)
            qdrant_filter = None
            if table_title or is_table_data is not None:
                must_conditions = []
                
                if table_title:
                    must_conditions.append(
                        FieldCondition(key="table_title", match=MatchValue(value=table_title))
                    )
                
                if is_table_data is not None:
                    must_conditions.append(
                        FieldCondition(key="is_table_data", match=MatchValue(value=is_table_data))
                    )
                
                if must_conditions:
                    qdrant_filter = Filter(must=must_conditions)
            
            self.logger.info(f"=== BGE-m3 비동기 하이브리드 검색 시작 ===")
            # 전처리 없이 원본 쿼리를 그대로 사용
            query_for_embedding = query
            self.logger.info(f"쿼리: {query[:100]}...")
            self.logger.info(f"Sparse 벡터 활성화: {self.sparse_enabled}")
            
            # BGE-m3로 쿼리 임베딩 생성 (비동기)
            import asyncio
            query_embeddings = await asyncio.to_thread(
                self.bge_model.encode,
                [query_for_embedding],
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False
            )
            
            # 키 이름 확인
            dense_key = 'dense_vecs' if 'dense_vecs' in query_embeddings else 'dense'
            dense_vec = query_embeddings[dense_key][0]
            sparse_weights = query_embeddings['lexical_weights'][0]
            
            # Dense 벡터 변환
            if hasattr(dense_vec, 'tolist'):
                dense_vector = dense_vec.tolist()
            elif isinstance(dense_vec, np.ndarray):
                dense_vector = dense_vec.tolist()
            else:
                dense_vector = list(dense_vec)
            
            # Sparse 벡터 변환
            sparse_vector = None
            if self.sparse_enabled and sparse_weights:
                if isinstance(sparse_weights, dict):
                    indices = list(sparse_weights.keys())
                    values = list(sparse_weights.values())
                else:
                    self.logger.warning(f"예상치 못한 sparse_weights 형태: {type(sparse_weights)}")
                    indices = []
                    values = []
                
                if indices and values:
                    sparse_vector = QdrantSparseVector(
                        indices=indices,
                        values=values
                    )
            
            # Prefetch + RRF 하이브리드 검색 (비동기)
            if self.sparse_enabled and sparse_vector:
                self.logger.info(f"✅ 하이브리드 검색 모드: Dense + Sparse 벡터 (RRF)")
                
                prefetch_list = [
                    Prefetch(
                        query=dense_vector,
                        using=self.dense_vector_name,
                        limit=limit * 2,
                        filter=qdrant_filter
                    ),
                    Prefetch(
                        query=sparse_vector,
                        using=self.sparse_vector_name,
                        limit=limit * 2,
                        filter=qdrant_filter
                    )
                ]
                
                if self.use_local_storage:
                    results = await asyncio.to_thread(
                        self.client.query_points,
                        collection_name=self.collection_name,
                        prefetch=prefetch_list,
                        query=FusionQuery(fusion=Fusion.RRF),
                        limit=limit,
                        with_payload=True,
                        with_vectors=False
                    )
                else:
                    if not hasattr(self, '_async_client') or self._async_client is None:
                        self._async_client = AsyncQdrantClient(host=self._async_client_host, port=self._async_client_port)
                    results = await self._async_client.query_points(
                        collection_name=self.collection_name,
                        prefetch=prefetch_list,
                        query=FusionQuery(fusion=Fusion.RRF),
                        limit=limit,
                        with_payload=True,
                        with_vectors=False
                    )
            else:
                # Dense만 사용
                self.logger.info(f"ℹ️  Dense 벡터만 사용")
                if self.use_local_storage:
                    results = await asyncio.to_thread(
                        self.client.query_points,
                        collection_name=self.collection_name,
                        query=dense_vector,
                        using=self.dense_vector_name,
                        limit=limit,
                        query_filter=qdrant_filter,
                        with_payload=True,
                        with_vectors=False
                    )
                else:
                    if not hasattr(self, '_async_client') or self._async_client is None:
                        self._async_client = AsyncQdrantClient(host=self._async_client_host, port=self._async_client_port)
                    results = await self._async_client.query_points(
                        collection_name=self.collection_name,
                        query=dense_vector,
                        using=self.dense_vector_name,
                        limit=limit,
                        query_filter=qdrant_filter,
                        with_payload=True,
                        with_vectors=False
                    )
            
            # 결과 변환
            search_results = []
            for point in results.points:
                payload = point.payload or {}
                
                # 점수 처리 (RRF 점수는 이미 유사도 형태)
                score = point.score if hasattr(point, 'score') else 0.0
                similarity_score = float(score)
                
                # 점수 범위 검증 (0-1 범위 확인)
                if similarity_score < 0.0 or similarity_score > 1.0:
                    self.logger.warning(
                        f"점수가 0-1 범위를 벗어남: {similarity_score:.4f}. "
                        f"자동 클리핑 적용"
                    )
                    similarity_score = max(0.0, min(1.0, similarity_score))
                
                # source_file 추출
                source_file = (
                    payload.get('source_file') or 
                    payload.get('file_path') or 
                    payload.get('file_name') or 
                    ''
                )
                
                # chunk_index 추출
                chunk_index = payload.get('chunk_index', 0)
                try:
                    chunk_index = int(chunk_index) if chunk_index is not None else 0
                except (ValueError, TypeError):
                    chunk_index = 0
                
                # chunk_id 추출
                chunk_id = payload.get('chunk_id', '')
                
                # chunk_index가 없으면 chunk_id에서 추출 시도
                if chunk_index == 0 and chunk_id:
                    match = re.search(r'_(\d+)$', chunk_id)
                    if match:
                        try:
                            chunk_index = int(match.group(1))
                        except (ValueError, TypeError):
                            chunk_index = 0
                
                # content 추출
                content = payload.get('page_content', '') or payload.get('content', '')
                
                search_results.append({
                    'content': content,
                    'score': similarity_score,
                    'metadata': payload,
                    'source_file': source_file,
                    'chunk_index': chunk_index,
                    'table_title': payload.get('table_title', ''),
                    'is_table_data': payload.get('is_table_data', False)
                })
            
            # 점수 임계값 필터링
            if score_threshold > 0:
                before_filter_count = len(search_results)
                search_results = [r for r in search_results if r['score'] >= score_threshold]
                filtered_out = before_filter_count - len(search_results)
                if filtered_out > 0:
                    self.logger.info(
                        f"비동기 점수 임계값({score_threshold:.3f}) 필터링: "
                        f"{before_filter_count}개 → {len(search_results)}개 (제외: {filtered_out}개)"
                    )
            
            self.logger.info(f"=== BGE-m3 비동기 하이브리드 검색 완료 ===")
            self.logger.info(f"최종 결과: {len(search_results)}개 (점수 임계값 필터링 후)")
            if self.sparse_enabled and sparse_vector:
                self.logger.info(f"✅ RRF 하이브리드 검색 완료: Dense + Sparse 벡터 통합 결과")
            
            return search_results
            
        except Exception as e:
            self.logger.error(f"비동기 필터 검색 실패: {str(e)}")
            import traceback
            self.logger.error(f"상세 오류: {traceback.format_exc()}")
            return []
    
    async def search_similar_async(self, 
                                  query: str, 
                                  limit: Optional[int] = None,
                                  score_threshold: Optional[float] = None,
                                  filter_conditions: Optional[Dict[str, Any]] = None,
                                  dense_weight: Optional[float] = None,
                                  sparse_weight: Optional[float] = None,
                                  keywords: Optional[List[str]] = None,
                                  entities: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        비동기 유사 문서 검색 (search_with_table_filter_async의 래퍼)
        
        Args:
            query: 검색 쿼리
            limit: 반환할 최대 결과 수
            score_threshold: 최소 점수 임계값
            filter_conditions: 필터 조건 (현재 미사용, 호환성 유지)
            dense_weight: Dense 벡터 가중치 (None이면 config 기본값 사용)
            sparse_weight: Sparse 벡터 가중치 (None이면 config 기본값 사용)
            keywords: query_refiner에서 추출한 키워드 리스트
            entities: query_refiner에서 추출한 엔티티 리스트
        """
        # search_with_table_filter_async에 모든 로직 위임 (가중치 체크 포함)
        return await self.search_with_table_filter_async(
            query=query,
            table_title=None,
            is_table_data=None,
            limit=limit,
            score_threshold=score_threshold,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            keywords=keywords,
            entities=entities
        )
    
    
    def _distance_to_similarity(self, distance: float, vector_type: str = 'dense') -> float:
        """
        거리(distance)를 유사도(similarity)로 변환
        
        Args:
            distance: Qdrant에서 반환한 거리 값
            vector_type: 벡터 타입 ('dense' 또는 'sparse')
            
        Returns:
            유사도 점수 (0.0 ~ 1.0)
        """
        if distance is None or distance < 0:
            return 0.0
        
        # Sparse 벡터는 BM25 기반이므로 이미 유사도 점수일 수 있음
        # 하지만 Qdrant는 거리로 반환하므로 변환 필요
        if vector_type == 'sparse':
            # Sparse 벡터는 보통 BM25 점수를 사용하지만, Qdrant에서는 거리로 반환
            # 거리 값이 0에 가까울수록 유사하므로, COSINE과 동일하게 처리
            # 단, Sparse 벡터의 경우 점수 범위가 다를 수 있으므로 주의
            if self.distance_metric == Distance.COSINE:
                similarity = 1.0 - (distance / 2.0)
            else:
                # 기본값: COSINE으로 처리
                similarity = 1.0 - (distance / 2.0)
        else:
            # Dense 벡터: Distance metric에 따라 변환
            if self.distance_metric == Distance.COSINE:
                # COSINE distance: 0(같음) ~ 2(다름)
                # 유사도: 1 - (distance / 2)
                similarity = 1.0 - (distance / 2.0)
            elif self.distance_metric == Distance.EUCLIDEAN:
                # EUCLIDEAN distance: 0(같음) ~ 무한대(다름)
                # 유사도: 1 / (1 + distance)
                similarity = 1.0 / (1.0 + distance)
            else:
                # 기본값: COSINE으로 처리
                similarity = 1.0 - (distance / 2.0)
        
        # 0-1 범위로 클리핑
        similarity = max(0.0, min(1.0, similarity))
        
        return similarity
    
    async def _hybrid_search_with_weights(self, 
                                         query: str, 
                                         limit: int,
                                         filter_conditions: Optional[Dict[str, Any]] = None,
                                         dense_weight: float = 0.7,
                                         sparse_weight: float = 0.3,
                                         keywords: Optional[List[str]] = None,
                                         entities: Optional[List[str]] = None) -> List[tuple]:
        """
        BGE-m3 기반 동적 가중치 하이브리드 검색 (Prefetch + 수동 가중치 결합)
        
        Returns:
            List[tuple]: (Document, score) 튜플 리스트
        """
        import asyncio
        from langchain_core.documents import Document
        
        try:
            # 전처리 없이 원본 쿼리를 그대로 사용
            query_for_embedding = query
            # BGE-m3로 쿼리 임베딩 생성 (비동기)
            query_embeddings = await asyncio.to_thread(
                self.bge_model.encode,
                [query_for_embedding],
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False
            )
            
            # 키 이름 확인
            dense_key = 'dense_vecs' if 'dense_vecs' in query_embeddings else 'dense'
            dense_vec = query_embeddings[dense_key][0]
            sparse_weights = query_embeddings['lexical_weights'][0]
            
            # Dense 벡터 변환
            if hasattr(dense_vec, 'tolist'):
                dense_vector = dense_vec.tolist()
            elif isinstance(dense_vec, np.ndarray):
                dense_vector = dense_vec.tolist()
            else:
                dense_vector = list(dense_vec)
            
            # Sparse 벡터 변환
            sparse_vector = None
            if self.sparse_enabled and sparse_weights:
                if isinstance(sparse_weights, dict):
                    indices = list(sparse_weights.keys())
                    values = list(sparse_weights.values())
                else:
                    self.logger.warning(f"예상치 못한 sparse_weights 형태: {type(sparse_weights)}")
                    indices = []
                    values = []
                
                if indices and values:
                    sparse_vector = QdrantSparseVector(
                        indices=indices,
                        values=values
                    )
            
            # 필터 구성
            qdrant_filter = None
            if filter_conditions:
                must_conditions = []
                for condition in filter_conditions.get("must", []):
                    key = condition.get("key")
                    match_value = condition.get("match", {}).get("value")
                    if key and match_value is not None:
                        must_conditions.append(
                            FieldCondition(key=key, match=MatchValue(value=match_value))
                        )
                if must_conditions:
                    qdrant_filter = Filter(must=must_conditions)
            
            self.logger.info(f"🔄 가중치 기반 하이브리드 검색 수행 (Dense={dense_weight}, Sparse={sparse_weight})")
            
            # Dense와 Sparse 검색 각각 수행 (가중치 결합을 위해)
            dense_results = None
            sparse_results = None
            
            # Dense 벡터 검색
            if dense_vector:
                if self.use_local_storage:
                    dense_results = await asyncio.to_thread(
                        self.client.query_points,
                        collection_name=self.collection_name,
                        query=dense_vector,
                        using=self.dense_vector_name,
                        limit=limit * 2,
                        query_filter=qdrant_filter,
                        with_payload=True,
                        with_vectors=False
                    )
                else:
                    if not hasattr(self, '_async_client') or self._async_client is None:
                        self._async_client = AsyncQdrantClient(host=self._async_client_host, port=self._async_client_port)
                    dense_results = await self._async_client.query_points(
                        collection_name=self.collection_name,
                        query=dense_vector,
                        using=self.dense_vector_name,
                        limit=limit * 2,
                        query_filter=qdrant_filter,
                        with_payload=True,
                        with_vectors=False
                    )
            
            # Sparse 벡터 검색
            if sparse_vector:
                if self.use_local_storage:
                    sparse_results = await asyncio.to_thread(
                        self.client.query_points,
                        collection_name=self.collection_name,
                        query=sparse_vector,
                        using=self.sparse_vector_name,
                        limit=limit * 2,
                        query_filter=qdrant_filter,
                        with_payload=True,
                        with_vectors=False
                    )
                else:
                    if not hasattr(self, '_async_client') or self._async_client is None:
                        self._async_client = AsyncQdrantClient(host=self._async_client_host, port=self._async_client_port)
                    sparse_results = await self._async_client.query_points(
                        collection_name=self.collection_name,
                        query=sparse_vector,
                        using=self.sparse_vector_name,
                        limit=limit * 2,
                        query_filter=qdrant_filter,
                        with_payload=True,
                        with_vectors=False
                    )
            
            # 가중치로 결과 결합
            combined_results = {}
            
            # Dense 결과 처리 (거리를 유사도로 변환)
            if dense_results and dense_results.points:
                for rank, point in enumerate(dense_results.points, 1):
                    point_id = str(point.id)
                    raw_distance = point.score if hasattr(point, 'score') else 0.0
                    # 거리를 유사도로 변환
                    dense_similarity = self._distance_to_similarity(raw_distance, 'dense')
                    
                    if point_id not in combined_results:
                        combined_results[point_id] = {
                            'point': point,
                            'dense_score': dense_similarity,
                            'dense_rank': rank,
                            'dense_distance': raw_distance,
                            'sparse_score': 0.0,
                            'sparse_rank': None,
                            'sparse_distance': 0.0,
                            'combined_score': 0.0
                        }
                    else:
                        combined_results[point_id]['dense_score'] = dense_similarity
                        combined_results[point_id]['dense_rank'] = rank
                        combined_results[point_id]['dense_distance'] = raw_distance
            
            # Sparse 결과 처리 (거리를 유사도로 변환)
            if sparse_results and sparse_results.points:
                for rank, point in enumerate(sparse_results.points, 1):
                    point_id = str(point.id)
                    raw_distance = point.score if hasattr(point, 'score') else 0.0
                    # 거리를 유사도로 변환
                    sparse_similarity = self._distance_to_similarity(raw_distance, 'sparse')
                    
                    if point_id not in combined_results:
                        combined_results[point_id] = {
                            'point': point,
                            'dense_score': 0.0,
                            'dense_rank': None,
                            'dense_distance': 0.0,
                            'sparse_score': sparse_similarity,
                            'sparse_rank': rank,
                            'sparse_distance': raw_distance,
                            'combined_score': 0.0
                        }
                    else:
                        combined_results[point_id]['sparse_score'] = sparse_similarity
                        combined_results[point_id]['sparse_rank'] = rank
                        combined_results[point_id]['sparse_distance'] = raw_distance
            
            # 가중치로 최종 점수 계산
            for point_id, result in combined_results.items():
                dense_score = result['dense_score']
                sparse_score = result['sparse_score']
                
                # 가중치 결합
                if dense_score > 0 and sparse_score > 0:
                    combined_score = (dense_score * dense_weight) + (sparse_score * sparse_weight)
                elif dense_score > 0:
                    combined_score = dense_score * dense_weight / (dense_weight + sparse_weight) if (dense_weight + sparse_weight) > 0 else dense_score
                elif sparse_score > 0:
                    combined_score = sparse_score * sparse_weight / (dense_weight + sparse_weight) if (dense_weight + sparse_weight) > 0 else sparse_score
                else:
                    combined_score = 0.0
                
                combined_score = max(0.0, min(1.0, combined_score))
                result['combined_score'] = combined_score
            
            # 최종 점수 기준으로 정렬
            sorted_results = sorted(
                combined_results.values(),
                key=lambda x: x['combined_score'],
                reverse=True
            )[:limit]
            
            # Document 형식으로 변환
            docs = []
            for result in sorted_results:
                point = result['point']
                payload = point.payload or {}
                
                page_content = payload.get('page_content', '') or payload.get('content', '')
                source_file = (
                    payload.get('source_file') or 
                    payload.get('file_path') or 
                    payload.get('file_name') or 
                    ''
                )
                chunk_index = payload.get('chunk_index', 0)
                try:
                    chunk_index = int(chunk_index) if chunk_index is not None else 0
                except (ValueError, TypeError):
                    chunk_index = 0
                chunk_id = payload.get('chunk_id', '')
                
                if chunk_index == 0 and chunk_id:
                    match = re.search(r'_(\d+)$', chunk_id)
                    if match:
                        try:
                            chunk_index = int(match.group(1))
                        except (ValueError, TypeError):
                            chunk_index = 0
                
                doc_metadata = {
                    'chunk_id': chunk_id,
                    'source_file': source_file,
                    'chunk_index': chunk_index,
                    **{k: v for k, v in payload.items() if k not in ['page_content', 'content', 'chunk_id', 'source_file', 'chunk_index']}
                }
                
                doc = Document(
                    page_content=page_content,
                    metadata=doc_metadata
                )
                docs.append((doc, result['combined_score']))
            
            self.logger.info(f"✅ 가중치 기반 하이브리드 검색 완료: {len(docs)}개 결과")
            self.logger.info(f"   - Dense 가중치: {dense_weight}, Sparse 가중치: {sparse_weight}")
            return docs
            
        except Exception as e:
            self.logger.error(f"가중치 기반 하이브리드 검색 실패: {str(e)}")
            import traceback
            self.logger.error(f"상세 오류: {traceback.format_exc()}")
            return []
    
    def get_documents_info(self) -> List[Dict[str, Any]]:
        """저장된 문서들의 정보 반환"""
        try:
            # 모든 포인트 조회 (메타데이터만)
            points = self.client.scroll(
                collection_name=self.collection_name,
                limit=self.max_scroll_limit,
                with_payload=True,
                with_vectors=False
            )[0]
            
            # 문서별로 그룹화
            documents = {}
            for point in points:
                payload = point.payload
                
                # source_file 추출 (payload 최상위 레벨 또는 metadata 내부)
                # 실제 저장 시: payload['source_file']에 직접 저장됨
                source_file = (payload.get('source_file') or 
                             payload.get('metadata', {}).get('source_file') or
                             payload.get('source') or 
                             payload.get('file_path') or 
                             payload.get('file_name') or 
                             '')
                
                # 메타데이터 추출 (중복 제거)
                metadata = self._get_metadata(payload)
                
                # source_file이 없으면 건너뛰기 (unknown 문서 제외)
                if not source_file or source_file.strip() == '':
                    continue
                
                # 파일명 추출 (경로에서 파일명만 추출)
                file_name = metadata.get('file_name', '')
                if not file_name and source_file != 'unknown':
                    # source_file에서 파일명 추출
                    file_name = os.path.basename(source_file)
                
                if source_file not in documents:
                    documents[source_file] = {
                        'source_file': source_file,
                        'file_name': file_name,  # 파일명 추가
                        'total_chunks': 0,
                        'first_chunk_index': float('inf'),
                        'last_chunk_index': -1,
                        'file_type': metadata.get('file_type', ''),
                        'file_size': metadata.get('file_size', 0),
                        'upload_time': metadata.get('upload_time', ''),
                        'chunk_ids': []
                    }
                
                documents[source_file]['total_chunks'] += 1
                chunk_index = metadata.get('chunk_index', 0)
                documents[source_file]['first_chunk_index'] = min(
                    documents[source_file]['first_chunk_index'], 
                    chunk_index
                )
                documents[source_file]['last_chunk_index'] = max(
                    documents[source_file]['last_chunk_index'], 
                    chunk_index
                )
                documents[source_file]['chunk_ids'].append(point.id)
            
            # 리스트로 변환
            result = list(documents.values())
            self.logger.info(f"문서 정보 조회 완료: {len(result)}개 문서")
            return result
            
        except Exception as e:
            self.logger.error(f"문서 정보 조회 실패: {str(e)}")
            return []
    
    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """특정 문서의 청크 정보 반환"""
        try:
            # 입력 검증
            if not document_id or document_id.strip() == '':
                self.logger.warning("get_document_chunks: document_id가 비어있습니다.")
                return []
            
            if document_id == 'N/A':
                self.logger.warning("get_document_chunks: 'N/A'는 유효한 문서 ID가 아닙니다.")
                return []
            
            # 모든 청크를 가져온 후 Python에서 필터링
            points = self.client.scroll(
                collection_name=self.collection_name,
                limit=self.max_scroll_limit,
                with_payload=True,
                with_vectors=False
            )[0]
            
            # 특정 문서의 청크들만 필터링
            filtered_points = []
            for point in points:
                metadata = self._get_metadata(point.payload)
                
                # source_file이 여러 위치에 있을 수 있으므로 모두 확인
                source_file = (
                    metadata.get('source_file') or 
                    metadata.get('file_path') or 
                    metadata.get('file_name') or 
                    ''
                )
                
                if source_file == document_id:
                    filtered_points.append(point)
            
            points = filtered_points
            
            chunks = []
            for point in points:
                payload = point.payload
                
                # 메타데이터 추출 (중복 제거)
                metadata = self._get_metadata(payload)
                
                chunks.append({
                    'chunk_id': point.id,
                    'chunk_index': metadata.get('chunk_index', 0),
                    'content_preview': payload.get('page_content', '')[:200] + '...',
                    'content_full': payload.get('page_content', ''),  # 전체 내용 추가
                    'content_length': len(payload.get('page_content', '')),
                    'metadata': metadata
                })
            
            # 청크 인덱스 순으로 정렬
            chunks.sort(key=lambda x: x['chunk_index'])
            
            self.logger.info(f"문서 청크 조회 완료: {document_id}, {len(chunks)}개 청크")
            return chunks
            
        except Exception as e:
            self.logger.error(f"문서 청크 조회 실패: {str(e)}")
            return []
    
    def _get_all_documents_from_qdrant(self) -> List[Any]:
        """
        Qdrant에서 모든 문서를 가져오기 (LangChain Document 형식)
        
        Returns:
            모든 문서의 LangChain Document 리스트
        """
        from langchain_core.documents import Document
        
        try:
            if not self._check_connection():
                return []
            
            all_documents = []
            offset = None
            
            # Scroll을 사용하여 모든 포인트 가져오기
            while True:
                scroll_result = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=self.max_scroll_limit,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False  # 벡터는 필요 없음
                )
                
                points, next_offset = scroll_result
                
                if not points:
                    break
                
                # Point를 LangChain Document로 변환
                for point in points:
                    payload = point.payload or {}
                    page_content = payload.get('page_content', '')
                    
                    if page_content:  # 내용이 있는 경우만 추가
                        doc = Document(
                            page_content=page_content,
                            metadata={
                                'chunk_id': payload.get('chunk_id', ''),
                                'source_file': payload.get('source_file', ''),
                                'chunk_index': payload.get('chunk_index', 0),
                                **{k: v for k, v in payload.items() 
                                   if k not in ['page_content', 'chunk_id', 'source_file', 'chunk_index']}
                            }
                        )
                        all_documents.append(doc)
                
                if next_offset is None:
                    break
                
                offset = next_offset
            
            self.logger.debug(f"Qdrant에서 총 {len(all_documents)}개 문서 가져옴")
            return all_documents
            
        except Exception as e:
            self.logger.error(f"Qdrant에서 모든 문서 가져오기 실패: {str(e)}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """컬렉션 정보 반환"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            
            # vectors가 딕셔너리인 경우 (다중 벡터 지원)와 단일 벡터인 경우 모두 처리
            vectors_config = collection_info.config.params.vectors
            if isinstance(vectors_config, dict):
                # 딕셔너리인 경우: 기본 벡터("") 또는 첫 번째 벡터 사용
                if "" in vectors_config:
                    vector_params = vectors_config[""]
                else:
                    # 기본 벡터가 없으면 첫 번째 벡터 사용
                    vector_params = next(iter(vectors_config.values()))
                vector_size = vector_params.size
                distance_metric = vector_params.distance.name
            else:
                # 단일 벡터인 경우 (레거시)
                vector_size = vectors_config.size
                distance_metric = vectors_config.distance.name
            
            # Sparse 벡터 정보 확인
            sparse_vectors_info = None
            if hasattr(collection_info.config.params, 'sparse_vectors') and \
               collection_info.config.params.sparse_vectors:
                sparse_vectors = collection_info.config.params.sparse_vectors
                if isinstance(sparse_vectors, dict):
                    sparse_vectors_info = list(sparse_vectors.keys())
            
            result = {
                'name': self.collection_name,
                'vector_size': vector_size,
                'distance_metric': distance_metric,
                'points_count': collection_info.points_count,
                'status': collection_info.status.name
            }
            
            if sparse_vectors_info:
                result['sparse_vectors'] = sparse_vectors_info
            
            return result
        except ValueError as e:
            # 컬렉션이 없는 경우 (초기화 시 정상적인 상황)
            if "not found" in str(e).lower():
                self.logger.debug(f"컬렉션이 아직 생성되지 않음: {self.collection_name} (초기화 중일 수 있음)")
                return {}
            else:
                self.logger.error(f"컬렉션 정보 조회 실패: {str(e)}")
                return {}
        except Exception as e:
            self.logger.error(f"컬렉션 정보 조회 실패: {str(e)}")
            import traceback
            self.logger.error(f"상세 오류: {traceback.format_exc()}")
            return {}
    
    def delete_collection(self) -> bool:
        """컬렉션 삭제"""
        try:
            self.client.delete_collection(self.collection_name)
            self.logger.info(f"컬렉션 삭제 완료: {self.collection_name}")
            return True
        except Exception as e:
            self.logger.error(f"컬렉션 삭제 실패: {str(e)}")
            return False
    
    '''샘플 벡터 DB확인'''
    def inspect_vectors(self, sample_size: int = 3, point_ids: Optional[List[Any]] = None) -> Dict[str, Any]:
        """
        샘플 포인트의 Dense와 Sparse 벡터 확인
        
        Args:
            sample_size: 확인할 샘플 포인트 수 (point_ids가 None일 때)
            point_ids: 확인할 특정 포인트 ID 리스트 (지정 시 sample_size 무시)
            
        Returns:
            벡터 정보 딕셔너리
        """
        try:
            # 컬렉션 정보 확인
            collection_info = self.client.get_collection(self.collection_name)
            
            # 벡터 설정 확인
            vectors_config = collection_info.config.params.vectors
            sparse_vectors_config = getattr(collection_info.config.params, 'sparse_vectors', None)
            
            result = {
                'collection_name': self.collection_name,
                'points_count': collection_info.points_count,
                'dense_vectors': {},
                'sparse_vectors': {},
                'samples': []
            }
            
            # Dense 벡터 설정 정보
            if isinstance(vectors_config, dict):
                for vec_name, vec_params in vectors_config.items():
                    result['dense_vectors'][vec_name or '(default)'] = {
                        'size': vec_params.size,
                        'distance': vec_params.distance.name
                    }
            elif vectors_config:
                result['dense_vectors']['(default)'] = {
                    'size': vectors_config.size,
                    'distance': vectors_config.distance.name
                }
            
            # Sparse 벡터 설정 정보
            if sparse_vectors_config and isinstance(sparse_vectors_config, dict):
                for sparse_name in sparse_vectors_config.keys():
                    result['sparse_vectors'][sparse_name] = {
                        'enabled': True
                    }
            
            # 샘플 포인트 가져오기
            if point_ids:
                # 특정 포인트 ID로 조회
                # Qdrant 다중 벡터 구조에서는 with_vectors를 딕셔너리로 지정해야 할 수 있음
                try:
                    # 방법 1: 모든 벡터 포함 (다중 벡터 구조)
                    points = self.client.retrieve(
                        collection_name=self.collection_name,
                        ids=point_ids,
                        with_payload=True,
                        with_vectors=True  # 모든 벡터 포함 (dense + sparse)
                    )
                except Exception as e:
                    self.logger.warning(f"retrieve with_vectors=True 실패: {str(e)}, 다른 방법 시도")
                    # 방법 2: 명시적으로 벡터 이름 지정
                    points = self.client.retrieve(
                        collection_name=self.collection_name,
                        ids=point_ids,
                        with_payload=True,
                        with_vectors={self.dense_vector_name: True, self.sparse_vector_name: True}
                    )
            else:
                # 샘플 포인트 스크롤
                try:
                    scroll_result = self.client.scroll(
                        collection_name=self.collection_name,
                        limit=sample_size,
                        with_payload=True,
                        with_vectors=True  # 모든 벡터 포함 (dense + sparse)
                    )
                    points = scroll_result[0]
                except Exception as e:
                    self.logger.warning(f"scroll with_vectors=True 실패: {str(e)}, 다른 방법 시도")
                    scroll_result = self.client.scroll(
                        collection_name=self.collection_name,
                        limit=sample_size,
                        with_payload=True,
                        with_vectors={self.dense_vector_name: True, self.sparse_vector_name: True}
                    )
                    points = scroll_result[0]
            
            # 샘플 포인트 분석
            for point in points:
                sample_info = {
                    'point_id': str(point.id),
                    'payload': point.payload,
                    'dense_vectors': {},
                    'sparse_vectors': {}
                }
                
                # 디버깅: point.vector 구조 확인
                self.logger.info(f"포인트 {point.id} 벡터 타입: {type(point.vector)}")
                if hasattr(point, 'vector'):
                    self.logger.info(f"포인트 {point.id} has vector: {point.vector is not None}")
                    if isinstance(point.vector, dict):
                        self.logger.info(f"포인트 {point.id} vector keys: {list(point.vector.keys())}")
                        for k, v in point.vector.items():
                            self.logger.info(f"포인트 {point.id} vector[{k}] 타입: {type(v)}")
                            if hasattr(v, 'indices'):
                                self.logger.info(f"포인트 {point.id} vector[{k}] indices 타입: {type(v.indices)}, 길이: {len(v.indices) if hasattr(v, '__len__') else 'N/A'}")
                    elif point.vector is not None:
                        self.logger.info(f"포인트 {point.id} vector는 리스트 타입, 길이: {len(point.vector) if hasattr(point.vector, '__len__') else 'N/A'}")
                
                # Dense 벡터 확인
                if point.vector:
                    if isinstance(point.vector, dict):
                        # 다중 벡터
                        for vec_name, vec_data in point.vector.items():
                            if isinstance(vec_data, list):
                                sample_info['dense_vectors'][vec_name or '(default)'] = {
                                    'size': len(vec_data),
                                    'preview': vec_data[:5] if len(vec_data) > 5 else vec_data,
                                    'has_data': True
                                }
                    elif isinstance(point.vector, list):
                        # 단일 벡터
                        sample_info['dense_vectors']['(default)'] = {
                            'size': len(point.vector),
                            'preview': point.vector[:5] if len(point.vector) > 5 else point.vector,
                            'has_data': True
                        }
                
                # Sparse 벡터 확인
                # 방법 1: point.vector 딕셔너리에서 sparse 벡터 확인 (Qdrant 다중 벡터 구조)
                if point.vector and isinstance(point.vector, dict):
                    self.logger.info(f"포인트 {point.id} vector 딕셔너리 확인: {list(point.vector.keys())}, sparse_vector_name={self.sparse_vector_name}")
                    for vec_name, vec_data in point.vector.items():
                        self.logger.info(f"포인트 {point.id} 벡터 이름: {vec_name}, 타입: {type(vec_data)}")
                        # Sparse 벡터는 QdrantSparseVector 객체 또는 dict 형태
                        is_sparse = (vec_name == self.sparse_vector_name) or (hasattr(vec_data, 'indices') and hasattr(vec_data, 'values'))
                        self.logger.info(f"포인트 {point.id} {vec_name} is_sparse: {is_sparse}, hasattr indices: {hasattr(vec_data, 'indices')}, hasattr values: {hasattr(vec_data, 'values')}")
                        if is_sparse:
                            sparse_info = {}
                            
                            if hasattr(vec_data, 'indices') and hasattr(vec_data, 'values'):
                                indices = list(vec_data.indices)
                                values = list(vec_data.values)
                            elif isinstance(vec_data, dict) and 'indices' in vec_data and 'values' in vec_data:
                                indices = list(vec_data['indices'])
                                values = list(vec_data['values'])
                            else:
                                continue
                            
                            sparse_info = {
                                'indices_count': len(indices),
                                'values_count': len(values),
                                'indices_preview': indices[:10] if len(indices) > 10 else indices,
                                'values_preview': values[:10] if len(values) > 10 else values,
                                'has_data': True
                            }
                            
                            # 토큰 인덱스를 실제 단어로 변환
                            tokens_info = []
                            try:
                                if hasattr(self, 'bge_model') and self.bge_model and hasattr(self.bge_model, 'tokenizer') and self.bge_model.tokenizer:
                                    for token_idx, weight in zip(indices, values):
                                        token_info = {
                                            'token_id': token_idx,
                                            'weight': float(weight)
                                        }
                                        
                                        try:
                                            # convert_ids_to_tokens: 서브워드 토큰 반환
                                            tokens = self.bge_model.tokenizer.convert_ids_to_tokens([token_idx])
                                            if tokens and len(tokens) > 0:
                                                token_info['token_text'] = tokens[0]
                                            
                                            # decode: 실제 단어로 디코딩
                                            token_word = self.bge_model.tokenizer.decode([token_idx], skip_special_tokens=True)
                                            if token_word:
                                                token_info['token_word'] = token_word.strip()
                                        except Exception as e:
                                            self.logger.debug(f"토큰 {token_idx} 변환 실패: {str(e)}")
                                        
                                        tokens_info.append(token_info)
                                    
                                    sparse_info['tokens'] = tokens_info
                                    sparse_info['tokens_preview'] = tokens_info[:10] if len(tokens_info) > 10 else tokens_info
                            except Exception as e:
                                self.logger.debug(f"토크나이저를 사용한 토큰 변환 실패: {str(e)}")
                            
                            sample_info['sparse_vectors'][vec_name or self.sparse_vector_name] = sparse_info
                
                # 방법 2: point.sparse_vectors 속성 확인 (레거시)
                if hasattr(point, 'sparse_vectors') and point.sparse_vectors:
                    if isinstance(point.sparse_vectors, dict):
                        for sparse_name, sparse_data in point.sparse_vectors.items():
                            if hasattr(sparse_data, 'indices') and hasattr(sparse_data, 'values'):
                                indices = list(sparse_data.indices)
                                values = list(sparse_data.values)
                                
                                sparse_info = {
                                    'indices_count': len(indices),
                                    'values_count': len(values),
                                    'indices_preview': indices[:10] if len(indices) > 10 else indices,
                                    'values_preview': values[:10] if len(values) > 10 else values,
                                    'has_data': True
                                }
                                
                                # 토큰 인덱스를 실제 단어로 변환
                                tokens_info = []
                                try:
                                    if hasattr(self, 'bge_model') and self.bge_model and hasattr(self.bge_model, 'tokenizer') and self.bge_model.tokenizer:
                                        for token_idx, weight in zip(indices, values):
                                            token_info = {
                                                'token_id': token_idx,
                                                'weight': float(weight)
                                            }
                                            
                                            try:
                                                tokens = self.bge_model.tokenizer.convert_ids_to_tokens([token_idx])
                                                if tokens and len(tokens) > 0:
                                                    token_info['token_text'] = tokens[0]
                                                
                                                token_word = self.bge_model.tokenizer.decode([token_idx], skip_special_tokens=True)
                                                if token_word:
                                                    token_info['token_word'] = token_word.strip()
                                            except Exception as e:
                                                self.logger.debug(f"토큰 {token_idx} 변환 실패: {str(e)}")
                                            
                                            tokens_info.append(token_info)
                                        
                                        sparse_info['tokens'] = tokens_info
                                        sparse_info['tokens_preview'] = tokens_info[:10] if len(tokens_info) > 10 else tokens_info
                                except Exception as e:
                                    self.logger.debug(f"토크나이저를 사용한 토큰 변환 실패: {str(e)}")
                                
                                sample_info['sparse_vectors'][sparse_name] = sparse_info
                
                result['samples'].append(sample_info)
            
            self.logger.info(f"벡터 확인 완료: {len(result['samples'])}개 샘플 포인트")
            return result
            
        except Exception as e:
            self.logger.error(f"벡터 확인 실패: {str(e)}")
            import traceback
            self.logger.error(f"상세 오류: {traceback.format_exc()}")
            return {
                'error': str(e),
                'collection_name': self.collection_name
            }
    
    def get_vector_statistics(self) -> Dict[str, Any]:
        """
        벡터 통계 정보 반환
        
        Returns:
            벡터 통계 딕셔너리
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            
            # 벡터 설정 확인
            vectors_config = collection_info.config.params.vectors
            sparse_vectors_config = getattr(collection_info.config.params, 'sparse_vectors', None)
            
            stats = {
                'collection_name': self.collection_name,
                'points_count': collection_info.points_count,
                'dense_vector_count': 0,
                'sparse_vector_count': 0,
                'dense_vectors_enabled': False,
                'sparse_vectors_enabled': False,
                'vector_configs': {}
            }
            
            # Dense 벡터 확인
            if vectors_config:
                stats['dense_vectors_enabled'] = True
                if isinstance(vectors_config, dict):
                    stats['dense_vector_count'] = len(vectors_config)
                    for vec_name, vec_params in vectors_config.items():
                        stats['vector_configs'][vec_name or '(default)'] = {
                            'type': 'dense',
                            'size': vec_params.size,
                            'distance': vec_params.distance.name
                        }
                else:
                    stats['dense_vector_count'] = 1
                    stats['vector_configs']['(default)'] = {
                        'type': 'dense',
                        'size': vectors_config.size,
                        'distance': vectors_config.distance.name
                    }
            
            # Sparse 벡터 확인
            if sparse_vectors_config and isinstance(sparse_vectors_config, dict):
                stats['sparse_vectors_enabled'] = True
                stats['sparse_vector_count'] = len(sparse_vectors_config)
                for sparse_name in sparse_vectors_config.keys():
                    stats['vector_configs'][sparse_name] = {
                        'type': 'sparse'
                    }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"벡터 통계 조회 실패: {str(e)}")
            return {
                'error': str(e),
                'collection_name': self.collection_name
            }
    
    def get_sparse_vocabulary(self, limit: Optional[int] = None, search_token: Optional[str] = None) -> Dict[str, Any]:
        """
        Qdrant에서 Sparse 벡터를 조회하여 Vocabulary 정보 집계 (BGE-m3 기반)
        
        Qdrant에 저장된 모든 포인트의 sparse 벡터를 조회하여:
        - 토큰 인덱스와 가중치 집계
        - 토큰별 문서 빈도 계산 (DF)
        - IDF 값 계산 (Inverse Document Frequency)
        - 통계 정보 제공
        
        Args:
            limit: 반환할 vocabulary 항목 수 (None이면 전체, 기본값: 1000개)
            search_token: 특정 토큰 검색 (토큰 인덱스가 포함된 항목만 반환)
        
        Returns:
            Vocabulary 정보 딕셔너리
        """
        try:
            result = {
                'sparse_enabled': self.sparse_enabled,
                'model_type': 'BGE-m3',
                'model_trained': True,
                'corpus_size': 0,
                'vocabulary_size': 0,
                'avgdl': 0.0,
                'vocabulary': {},
                'idf_values': {},
                'statistics': {},
                'message': ''
            }
            
            if not self.sparse_enabled:
                result['message'] = 'Sparse 벡터가 비활성화되어 있습니다.'
                return result
            
            if not self._check_connection():
                result['message'] = 'Qdrant 연결을 확인할 수 없습니다.'
                return result
            
            self.logger.info("Qdrant에서 sparse 벡터 vocabulary 집계 시작...")
            
            # 모든 포인트의 sparse 벡터 수집
            token_doc_freq: Dict[int, int] = {}  # 토큰 인덱스 -> 문서 수
            token_weights: Dict[int, List[float]] = {}  # 토큰 인덱스 -> 가중치 리스트
            total_documents = 0
            offset = None
            
            # Scroll을 사용하여 모든 포인트 가져오기
            while True:
                scroll_result = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=self.max_scroll_limit,
                    offset=offset,
                    with_payload=False,
                    with_vectors=True  # sparse 벡터를 가져오기 위해 필요
                )
                
                points, next_offset = scroll_result
                
                if not points:
                    break
                
                for point in points:
                    # Sparse 벡터 추출
                    if hasattr(point, 'vector') and point.vector:
                        vectors = point.vector
                        if isinstance(vectors, dict) and self.sparse_vector_name in vectors:
                            sparse_vec = vectors[self.sparse_vector_name]
                            if hasattr(sparse_vec, 'indices') and hasattr(sparse_vec, 'values'):
                                indices = sparse_vec.indices
                                values = sparse_vec.values
                                
                                # 각 토큰 인덱스에 대해 문서 빈도 및 가중치 수집
                                seen_tokens_in_doc = set()
                                for idx, weight in zip(indices, values):
                                    if idx not in seen_tokens_in_doc:
                                        token_doc_freq[idx] = token_doc_freq.get(idx, 0) + 1
                                        seen_tokens_in_doc.add(idx)
                                    
                                    if idx not in token_weights:
                                        token_weights[idx] = []
                                    token_weights[idx].append(float(weight))
                
                total_documents += len(points)
                
                if next_offset is None:
                    break
                
                offset = next_offset
            
            if total_documents == 0:
                result['message'] = 'Qdrant에 저장된 문서가 없습니다.'
                return result
            
            # IDF 계산: IDF(t) = log(N / df(t))
            # N: 전체 문서 수, df(t): 토큰 t를 포함하는 문서 수
            import math
            vocabulary = {}
            idf_values = {}
            
            # 토크나이저를 사용하여 토큰 인덱스를 실제 단어로 변환
            tokenizer_available = False
            try:
                if hasattr(self.bge_model, 'tokenizer') and self.bge_model.tokenizer is not None:
                    tokenizer_available = True
                    self.logger.info("토크나이저를 사용하여 토큰 인덱스를 단어로 변환합니다.")
            except Exception as e:
                self.logger.warning(f"토크나이저 접근 실패: {str(e)}. 토큰 인덱스만 반환합니다.")
            
            for token_idx in token_doc_freq.keys():
                df = token_doc_freq[token_idx]
                idf = math.log(total_documents / df) if df > 0 else 0.0
                
                # 평균 가중치 계산
                avg_weight = sum(token_weights[token_idx]) / len(token_weights[token_idx]) if token_idx in token_weights else 0.0
                max_weight = max(token_weights[token_idx]) if token_idx in token_weights else 0.0
                min_weight = min(token_weights[token_idx]) if token_idx in token_weights else 0.0
                
                # 토큰 인덱스를 실제 단어로 변환
                token_text = None
                token_text_decoded = None
                if tokenizer_available:
                    try:
                        # convert_ids_to_tokens: 서브워드 토큰 반환 (예: "전기" -> ["전", "##기"])
                        tokens = self.bge_model.tokenizer.convert_ids_to_tokens([token_idx])
                        if tokens and len(tokens) > 0:
                            token_text = tokens[0]
                        
                        # decode: 실제 단어로 디코딩 (서브워드 토큰을 합쳐서 단어로 만듦)
                        token_text_decoded = self.bge_model.tokenizer.decode([token_idx], skip_special_tokens=True)
                        # decode는 공백을 추가할 수 있으므로 strip
                        if token_text_decoded:
                            token_text_decoded = token_text_decoded.strip()
                    except Exception as e:
                        self.logger.debug(f"토큰 {token_idx} 변환 실패: {str(e)}")
                        token_text = None
                        token_text_decoded = None
                
                vocabulary[token_idx] = {
                    'index': token_idx,
                    'document_frequency': df,
                    'avg_weight': avg_weight,
                    'max_weight': max_weight,
                    'min_weight': min_weight,
                    'total_occurrences': len(token_weights[token_idx]),
                    'token_text': token_text,  # 서브워드 토큰 (예: "전", "##기")
                    'token_word': token_text_decoded  # 실제 단어 (예: "전기")
                }
                idf_values[token_idx] = idf
            
            # search_token 필터링 (토큰 인덱스, 토큰 텍스트, 실제 단어 모두 검색)
            if search_token:
                search_token_lower = search_token.lower()
                filtered_vocab = {}
                filtered_idf = {}
                for token_idx, vocab_info in vocabulary.items():
                    token_str = str(token_idx)
                    token_text = vocab_info.get('token_text', '') or ''
                    token_word = vocab_info.get('token_word', '') or ''
                    
                    # 토큰 인덱스, 토큰 텍스트, 실제 단어 중 하나라도 매칭되면 포함
                    if (search_token_lower in token_str.lower() or 
                        search_token_lower in token_text.lower() or 
                        search_token_lower in token_word.lower()):
                        filtered_vocab[token_idx] = vocab_info
                        filtered_idf[token_idx] = idf_values[token_idx]
                vocabulary = filtered_vocab
                idf_values = filtered_idf
            
            # limit 적용 및 정렬 (IDF 값 기준 내림차순)
            sorted_tokens = sorted(idf_values.items(), key=lambda x: x[1], reverse=True)
            if limit:
                sorted_tokens = sorted_tokens[:limit]
            
            # 최종 vocabulary 및 idf_values 구성
            final_vocabulary = {}
            final_idf_values = {}
            for token_idx, idf_val in sorted_tokens:
                final_vocabulary[token_idx] = vocabulary[token_idx]
                final_idf_values[token_idx] = idf_val
            
            # 통계 정보 계산
            if idf_values:
                idf_list = list(idf_values.values())
                statistics = {
                    'min_idf': min(idf_list),
                    'max_idf': max(idf_list),
                    'avg_idf': sum(idf_list) / len(idf_list),
                    'median_idf': sorted(idf_list)[len(idf_list) // 2] if idf_list else 0.0,
                    'top_tokens': [
                        {
                            'index': idx, 
                            'idf': idf_val, 
                            'df': vocabulary[idx]['document_frequency'],
                            'token_text': vocabulary[idx].get('token_text'),
                            'token_word': vocabulary[idx].get('token_word')
                        }
                        for idx, idf_val in sorted_tokens[:10]
                    ]
                }
            else:
                statistics = {}
            
            result.update({
                'corpus_size': total_documents,
                'vocabulary_size': len(final_vocabulary),
                'vocabulary': final_vocabulary,
                'idf_values': final_idf_values,
                'statistics': statistics,
                'message': f'Qdrant에서 {total_documents}개 문서의 sparse 벡터를 조회하여 {len(final_vocabulary)}개 토큰을 집계했습니다.'
            })
            
            self.logger.info(f"Vocabulary 집계 완료: {total_documents}개 문서, {len(final_vocabulary)}개 토큰")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Sparse vocabulary 조회 실패: {str(e)}")
            import traceback
            self.logger.error(f"상세 오류: {traceback.format_exc()}")
            return {
                'error': str(e),
                'sparse_enabled': self.sparse_enabled,
                'message': f'Vocabulary 조회 중 오류 발생: {str(e)}'
            }

    # ------------------------------------------------------------------
    def _preprocess_texts_for_embedding(self, texts: List[str]) -> List[str]:
        """BGE 임베딩 전에 Kiwipiepy 형태소 전처리를 적용"""
        if not self.use_kiwipiepy_preprocessing or not self.kiwipiepy_preprocessor:
            return texts
        processed: List[str] = []
        for text in texts:
            try:
                processed.append(self.kiwipiepy_preprocessor.preprocess(text))
            except Exception as exc:  # pragma: no cover - 방어적 처리
                self.logger.warning(f"Kiwipiepy 전처리 실패. 원본 텍스트 사용: {exc}")
                processed.append(text)
        return processed

    def _preprocess_query_text(self, text: str) -> str:
        """검색 쿼리 전처리"""
        if not self.use_kiwipiepy_preprocessing or not self.kiwipiepy_preprocessor:
            return text
        try:
            return self.kiwipiepy_preprocessor.preprocess(text)
        except Exception as exc:  # pragma: no cover
            self.logger.warning(f"Kiwipiepy 쿼리 전처리 실패. 원본 사용: {exc}")
            return text
    
    def analyze_sparse_quality(self) -> Dict[str, Any]:
        """
        Sparse 벡터 DB 품질 분석
        
        Returns:
            품질 분석 결과 딕셔너리
        """
        import statistics
        from collections import Counter
        
        try:
            result = {
                'sparse_enabled': self.sparse_enabled,
                'total_points': 0,
                'points_with_sparse': 0,
                'points_without_sparse': 0,
                'empty_sparse_vectors': 0,
                'token_statistics': {},
                'weight_statistics': {},
                'weight_distribution': {},
                'vocabulary_statistics': {},
                'quality_assessment': {},
                'recommendations': [],
                'issues': []
            }
            
            if not self.sparse_enabled:
                result['message'] = 'Sparse 벡터가 비활성화되어 있습니다.'
                return result
            
            if not self._check_connection():
                result['message'] = 'Qdrant 연결을 확인할 수 없습니다.'
                return result
            
            self.logger.info("Sparse 벡터 품질 분석 시작...")
            
            # 통계 수집
            token_counts = []
            weight_values = []
            token_frequency = Counter()
            low_weight_tokens = 0  # < 0.1
            medium_weight_tokens = 0  # 0.1-0.5
            high_weight_tokens = 0  # >= 0.5
            
            # 모든 포인트 스크롤
            offset = None
            total_points = 0
            points_with_sparse = 0
            points_without_sparse = 0
            empty_sparse_vectors = 0
            
            while True:
                scroll_result = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=self.max_scroll_limit,
                    offset=offset,
                    with_payload=False,
                    with_vectors=True
                )
                
                points, next_offset = scroll_result
                
                if not points:
                    break
                
                total_points += len(points)
                
                for point in points:
                    vectors = point.vector if hasattr(point, 'vector') else {}
                    
                    if isinstance(vectors, dict):
                        sparse_vector = vectors.get(self.sparse_vector_name)
                    else:
                        sparse_vector = None
                    
                    if sparse_vector is None:
                        points_without_sparse += 1
                        continue
                    
                    points_with_sparse += 1
                    
                    # Sparse 벡터 구조 확인
                    if hasattr(sparse_vector, 'indices') and hasattr(sparse_vector, 'values'):
                        indices = sparse_vector.indices
                        values = sparse_vector.values
                    elif isinstance(sparse_vector, dict):
                        indices = list(sparse_vector.keys())
                        values = list(sparse_vector.values())
                    else:
                        continue
                    
                    if not indices or not values:
                        empty_sparse_vectors += 1
                        continue
                    
                    # 토큰 수
                    token_count = len(indices)
                    token_counts.append(token_count)
                    
                    # 토큰 빈도
                    for token_idx in indices:
                        token_frequency[token_idx] += 1
                    
                    # 가중치 분석
                    for weight in values:
                        weight_values.append(float(weight))
                        
                        if weight < 0.1:
                            low_weight_tokens += 1
                        elif weight < 0.5:
                            medium_weight_tokens += 1
                        else:
                            high_weight_tokens += 1
                
                if next_offset is None:
                    break
                
                offset = next_offset
            
            # 통계 계산
            result['total_points'] = total_points
            result['points_with_sparse'] = points_with_sparse
            result['points_without_sparse'] = points_without_sparse
            result['empty_sparse_vectors'] = empty_sparse_vectors
            
            # 토큰 통계
            if token_counts:
                result['token_statistics'] = {
                    'mean': statistics.mean(token_counts),
                    'median': statistics.median(token_counts),
                    'min': min(token_counts),
                    'max': max(token_counts),
                    'stdev': statistics.stdev(token_counts) if len(token_counts) > 1 else 0.0,
                    'q25': sorted(token_counts)[len(token_counts) // 4] if token_counts else 0,
                    'q75': sorted(token_counts)[len(token_counts) * 3 // 4] if token_counts else 0
                }
            
            # 가중치 통계
            if weight_values:
                result['weight_statistics'] = {
                    'mean': statistics.mean(weight_values),
                    'median': statistics.median(weight_values),
                    'min': min(weight_values),
                    'max': max(weight_values),
                    'stdev': statistics.stdev(weight_values) if len(weight_values) > 1 else 0.0
                }
            
            # 가중치 분포
            total_tokens = low_weight_tokens + medium_weight_tokens + high_weight_tokens
            if total_tokens > 0:
                result['weight_distribution'] = {
                    'low_weight_count': low_weight_tokens,
                    'low_weight_percentage': low_weight_tokens / total_tokens * 100,
                    'medium_weight_count': medium_weight_tokens,
                    'medium_weight_percentage': medium_weight_tokens / total_tokens * 100,
                    'high_weight_count': high_weight_tokens,
                    'high_weight_percentage': high_weight_tokens / total_tokens * 100
                }
            
            # Vocabulary 통계
            unique_tokens = len(token_frequency)
            result['vocabulary_statistics'] = {
                'unique_tokens': unique_tokens,
                'most_common_tokens': [
                    {'token_id': token_id, 'frequency': count, 'percentage': count / points_with_sparse * 100 if points_with_sparse > 0 else 0}
                    for token_id, count in token_frequency.most_common(10)
                ]
            }
            
            # 품질 평가
            issues = []
            recommendations = []
            
            # 1. Sparse 벡터 누락 확인
            if total_points > 0:
                missing_rate = points_without_sparse / total_points
                if missing_rate > 0.1:
                    issues.append(f"{missing_rate*100:.1f}%의 포인트에 sparse 벡터가 없습니다.")
                    recommendations.append("문서 재처리 시 sparse 벡터 생성 확인 필요")
            
            # 2. 토큰 수 확인
            if token_counts:
                avg_tokens = statistics.mean(token_counts)
                if avg_tokens < 10:
                    issues.append(f"평균 토큰 수가 너무 적습니다 ({avg_tokens:.1f}개).")
                    recommendations.append("청크 크기 증가 또는 텍스트 전처리 개선 고려")
                elif avg_tokens > 200:
                    issues.append(f"평균 토큰 수가 너무 많습니다 ({avg_tokens:.1f}개).")
                    recommendations.append("청크 크기 감소 또는 더 세밀한 청킹 고려")
            
            # 3. 가중치 분포 확인
            if total_tokens > 0:
                low_weight_ratio = low_weight_tokens / total_tokens
                if low_weight_ratio > 0.5:
                    issues.append(f"낮은 가중치 토큰이 {low_weight_ratio*100:.1f}%를 차지합니다.")
                    recommendations.append("BGE-m3 모델의 lexical weight 임계값 조정 고려")
                
                high_weight_ratio = high_weight_tokens / total_tokens
                if high_weight_ratio < 0.1:
                    issues.append(f"높은 가중치 토큰이 {high_weight_ratio*100:.1f}%에 불과합니다.")
                    recommendations.append("텍스트 품질 개선 또는 모델 파라미터 조정 고려")
            
            # 4. Vocabulary 크기 확인
            if unique_tokens < 100:
                issues.append(f"Vocabulary가 너무 작습니다 ({unique_tokens}개).")
                recommendations.append("더 많은 문서 추가 또는 텍스트 다양성 증가 필요")
            elif unique_tokens > 100000:
                issues.append(f"Vocabulary가 너무 큽니다 ({unique_tokens:,}개).")
                recommendations.append("불필요한 토큰 필터링 또는 전처리 개선 고려")
            
            result['quality_assessment'] = {
                'overall_quality': 'good' if not issues else 'needs_improvement',
                'issues': issues
            }
            result['recommendations'] = recommendations
            
            self.logger.info("Sparse 벡터 품질 분석 완료")
            return result
            
        except Exception as e:
            self.logger.error(f"Sparse 벡터 품질 분석 실패: {str(e)}")
            import traceback
            self.logger.error(f"상세 오류: {traceback.format_exc()}")
            return {
                'error': str(e),
                'message': f'품질 분석 실패: {str(e)}'
            }


