"""
벡터 저장소 모듈
LangChain-Qdrant를 사용한 벡터 데이터베이스 관리
"""

from typing import List, Dict, Any, Optional
import uuid
import json
import os
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, Filter, FieldCondition, MatchValue, Query, NamedSparseVector, Prefetch, SparseVector as QdrantSparseVector
from langchain_qdrant import QdrantVectorStore as LangChainQdrantVectorStore
from langchain_qdrant.qdrant import RetrievalMode
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from src.utils.logger import get_logger
from src.utils.config import get_qdrant_config, get_embedding_config
from src.modules.document_processor import DocumentChunk
# 레거시 BM25Indexer 제거됨 - LangChain BM25Retriever 사용
# from src.modules.bm25_indexer import BM25Indexer
from src.modules.langchain_retrievers import LangChainRetrievalManager
from src.modules.langchain_embedding_wrapper import EmbeddingManagerWrapper
from src.modules.embedding_module import EmbeddingManager
from src.modules.sparse_embedding import BM25SparseEmbedding, SparseEmbeddingManager


class QdrantVectorStore:
    """LangChain-Qdrant 벡터 저장소 (로컬 파일 시스템)"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, embeddings: Optional[Any] = None):
        """
        Args:
            config: Qdrant 설정
            embeddings: 기존 임베딩 인스턴스 (선택적, 중복 로드 방지용)
        """
        self.logger = get_logger()
        
        if config is None:
            config = get_qdrant_config()
        
        self.collection_name = config.collection_name
        self.vector_size = config.vector_size
        self.distance_metric = Distance.COSINE if config.distance_metric.lower() == 'cosine' else Distance.EUCLIDEAN
        self.storage_path = config.storage_path
        self.use_local_storage = config.use_local_storage
        
        # Sparse 벡터 설정
        self.sparse_enabled = getattr(config, 'sparse_enabled', True)
        self.sparse_vector_name = getattr(config, 'sparse_vector_name', 'sparse')
        self.hybrid_search_dense_weight = getattr(config, 'hybrid_search_dense_weight', 0.7)
        self.hybrid_search_sparse_weight = getattr(config, 'hybrid_search_sparse_weight', 0.3)
        
        # 검색 기본값 설정
        self.default_limit = config.default_limit
        self.max_scroll_limit = config.max_scroll_limit
        # score_threshold는 RAG 설정에서 참조하도록 변경 (통일화)
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
        
        # 임베딩 설정: 기존 인스턴스가 있으면 재사용 (중복 로드 방지)
        if embeddings is not None:
            self.embeddings = embeddings
            self.logger.info("기존 임베딩 인스턴스 재사용 (중복 로드 방지)")
        else:
            # LangChain 임베딩 설정 (설정에서 가져오기)
            from src.utils.config import get_embedding_config
            embedding_config = get_embedding_config()
            
            # provider에 따라 적절한 임베딩 클래스 선택
            if embedding_config.provider == "huggingface":
                from langchain_huggingface import HuggingFaceEmbeddings
                # 디바이스 결정: CUDA 가용성 확인 후 불가하면 CPU로 강제 전환
                resolved_device = getattr(embedding_config, 'device', 'cuda') or 'cuda'
                if resolved_device == 'cuda':
                    try:
                        import torch  # 지연 임포트로 초기화 비용 최소화
                        import torchaudio
                        import torchvision
                        
                        # PyTorch 패키지 버전 정보 로깅
                        self.logger.info(f"PyTorch 패키지 버전 정보:")
                        self.logger.info(f"  torch: {torch.__version__}")
                        self.logger.info(f"  torchaudio: {torchaudio.__version__}")
                        self.logger.info(f"  torchvision: {torchvision.__version__}")
                        
                        if not torch.cuda.is_available():
                            self.logger.warning("CUDA를 사용할 수 없습니다. CPU로 전환합니다.")
                            resolved_device = 'cpu'
                    except Exception:
                        resolved_device = 'cpu'
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=embedding_config.model_path or embedding_config.name,
                    model_kwargs={'device': resolved_device},
                    encode_kwargs={'normalize_embeddings': True}
                )
                self.logger.info(f"HuggingFace 임베딩 초기화: {(embedding_config.model_path or embedding_config.name)} (device={resolved_device})")
            else:
                self.embeddings = OllamaEmbeddings(
                    model=embedding_config.name,
                    base_url=embedding_config.base_url
                )
                self.logger.info(f"Ollama 임베딩 초기화: {embedding_config.name}")
        
        # Sparse 벡터 설정 추가
        self.sparse_vocabulary_path = getattr(config, 'sparse_vocabulary_path', 'data/sparse_vocabulary')
        self.sparse_use_morphological = getattr(config, 'sparse_use_morphological', True)
        self.sparse_include_doc_stats = getattr(config, 'sparse_include_doc_stats', False)
        
        # Vocabulary 파일 경로 생성 (컬렉션별로 분리)
        vocabulary_file = f"{self.sparse_vocabulary_path}/{self.collection_name}_vocabulary.json"
        
        # Sparse 임베딩 초기화 (sparse_enabled일 때만)
        self.sparse_embedding = None
        self.sparse_embedding_manager = None
        if self.sparse_enabled:
            self.sparse_embedding_manager = SparseEmbeddingManager(
                vocabulary_path=vocabulary_file,
                use_morphological=self.sparse_use_morphological
            )
            self.sparse_embedding = self.sparse_embedding_manager.get_sparse_embedding()
            if self.sparse_embedding_manager.is_fitted:
                self.logger.info(f"Sparse 임베딩 초기화 완료 (저장된 Vocabulary 로드됨: {vocabulary_file})")
            else:
                morphological_status = "활성화" if self.sparse_use_morphological else "비활성화"
                self.logger.info(f"Sparse 임베딩 초기화 완료 (형태소 분석: {morphological_status}, 학습은 문서 추가 시 수행)")
        
        # LangChain Qdrant 벡터 저장소는 나중에 초기화 (컬렉션 생성 후)
        self.vector_store = None
    
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
        """컬렉션 생성"""
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
                    
                    # LangChain Qdrant 벡터 저장소 초기화
                    if self.vector_store is None:
                        self.logger.info("LangChain Qdrant 벡터 저장소 초기화 중...")
                        
                        # RetrievalMode 결정
                        if self.sparse_enabled and self.sparse_embedding is not None:
                            retrieval_mode = RetrievalMode.HYBRID
                            self.logger.info(f"RetrievalMode: HYBRID (dense + sparse)")
                        else:
                            retrieval_mode = RetrievalMode.DENSE
                            self.logger.info(f"RetrievalMode: DENSE")
                        
                        self.vector_store = LangChainQdrantVectorStore(
                            client=self.client,
                            collection_name=self.collection_name,
                            embedding=self.embeddings,
                            retrieval_mode=retrieval_mode,
                            sparse_embedding=self.sparse_embedding if self.sparse_enabled else None,
                            sparse_vector_name=self.sparse_vector_name if self.sparse_enabled else None
                        )
                        self.logger.info("LangChain Qdrant 벡터 저장소 초기화 완료")
                    
                    return True
            
            # 새 컬렉션 생성
            self.logger.info(f"새 컬렉션 생성 중: {self.collection_name}, 벡터 크기: {self.vector_size}")
            
            # Dense 벡터 설정
            vectors_config = {
                "": VectorParams(
                    size=self.vector_size,
                    distance=self.distance_metric
                )
            }
            
            # Sparse 벡터 설정 (sparse_enabled일 때만)
            sparse_vectors_config = None
            if self.sparse_enabled:
                sparse_vectors_config = {
                    self.sparse_vector_name: SparseVectorParams()
                }
                self.logger.info(f"Sparse 벡터 설정 추가: {self.sparse_vector_name}")
            
            # 컬렉션 생성
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=vectors_config,
                sparse_vectors_config=sparse_vectors_config
            )
            
            self.logger.info(f"컬렉션 생성 완료: {self.collection_name}")
            
            # LangChain Qdrant 벡터 저장소 초기화
            self.logger.info("LangChain Qdrant 벡터 저장소 초기화 중...")
            
            # RetrievalMode 결정
            if self.sparse_enabled and self.sparse_embedding is not None:
                retrieval_mode = RetrievalMode.HYBRID
                self.logger.info(f"RetrievalMode: HYBRID (dense + sparse)")
            else:
                retrieval_mode = RetrievalMode.DENSE
                self.logger.info(f"RetrievalMode: DENSE")
            
            self.vector_store = LangChainQdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embeddings,
                retrieval_mode=retrieval_mode,
                sparse_embedding=self.sparse_embedding if self.sparse_enabled else None,
                sparse_vector_name=self.sparse_vector_name if self.sparse_enabled else None
            )
            self.logger.info("LangChain Qdrant 벡터 저장소 초기화 완료")
            
            return True
            
        except Exception as e:
            self.logger.error(f"컬렉션 생성 실패: {str(e)}")
            import traceback
            self.logger.error(f"상세 오류: {traceback.format_exc()}")
            return False
    
    def add_documents(self, documents: List[DocumentChunk], force_update: bool = False) -> bool:
        """문서 추가 (LangChain Document 형식으로 변환, dense+sparse 벡터 함께 저장)"""
        if not self._check_connection():
            return False
        
        if self.vector_store is None:
            self.logger.error("LangChain Qdrant 벡터 저장소가 초기화되지 않았습니다")
            return False
        
        try:
            # DocumentChunk를 LangChain Document로 변환
            langchain_docs = []
            seen_chunks = set()  # 중복 청크 방지
            
            for doc in documents:
                # 청크 고유 식별자 생성 (파일명 + 청크 인덱스)
                chunk_key = f"{doc.source_file}:{doc.chunk_index}"
                    
                if chunk_key in seen_chunks:
                    self.logger.warning(f"중복 청크 건너뛰기: {chunk_key}")
                    continue
                
                seen_chunks.add(chunk_key)
                
                langchain_doc = Document(
                    page_content=doc.content,
                    metadata={
                        'chunk_id': doc.chunk_id,
                        'source_file': doc.source_file,
                        'chunk_index': doc.chunk_index,
                        **doc.metadata
                    }
                )
                langchain_docs.append(langchain_doc)
            
            # Sparse 임베딩 모델 학습
            if self.sparse_enabled and self.sparse_embedding_manager:
                if not self.sparse_embedding_manager.is_fitted:
                    # 첫 번째 학습: 현재 문서로 학습
                    self.logger.info("Sparse 임베딩 모델 학습 시작 (초기 학습)...")
                    document_texts = [doc.page_content for doc in langchain_docs]
                    self.sparse_embedding_manager.fit(document_texts, include_doc_stats=self.sparse_include_doc_stats)
                    self.sparse_embedding = self.sparse_embedding_manager.get_sparse_embedding()
                    # vector_store의 sparse_embedding도 업데이트 (LangChain QdrantVectorStore 내부 속성)
                    if self.vector_store:
                        # LangChain QdrantVectorStore는 _sparse_embeddings 속성에 저장
                        if hasattr(self.vector_store, '_sparse_embeddings'):
                            self.vector_store._sparse_embeddings = self.sparse_embedding
                            self.logger.info("vector_store._sparse_embeddings 업데이트 완료")
                        elif hasattr(self.vector_store, 'sparse_embedding'):
                            self.vector_store.sparse_embedding = self.sparse_embedding
                            self.logger.info("vector_store.sparse_embedding 업데이트 완료")
                    # 학습 상태 확인 로그
                    if self.sparse_embedding and hasattr(self.sparse_embedding, 'corpus_size'):
                        self.logger.info(f"Sparse 임베딩 모델 학습 완료: corpus_size={self.sparse_embedding.corpus_size}, vocabulary_size={len(self.sparse_embedding.vocabulary)}")
                    else:
                        self.logger.info("Sparse 임베딩 모델 학습 완료")
                else:
                    # 추가 문서 업로드 시: 전체 문서로 재학습 (vocabulary 업데이트)
                    self.logger.info("추가 문서 업로드 감지: Sparse 임베딩 모델 재학습 시작...")
                    try:
                        # Qdrant에서 모든 문서 가져오기
                        all_documents = self._get_all_documents_from_qdrant()
                        if all_documents:
                            # 내용 기반 중복 제거 (Vocabulary 학습 정확도 향상)
                            import hashlib
                            seen_content_hashes = set()
                            unique_documents = []
                            
                            for doc in all_documents:
                                content_hash = hashlib.sha256(doc.page_content.encode('utf-8')).hexdigest()
                                if content_hash not in seen_content_hashes:
                                    seen_content_hashes.add(content_hash)
                                    unique_documents.append(doc)
                            
                            # 기존 문서 (중복 제거됨)
                            existing_texts = [doc.page_content for doc in unique_documents]
                            
                            # 새 문서도 중복 체크 (기존 문서와의 중복 제거)
                            new_texts = [doc.page_content for doc in langchain_docs]
                            unique_new_texts = []
                            for text in new_texts:
                                content_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
                                if content_hash not in seen_content_hashes:
                                    seen_content_hashes.add(content_hash)
                                    unique_new_texts.append(text)
                            
                            all_texts = existing_texts + unique_new_texts
                            
                            if len(unique_new_texts) < len(new_texts):
                                self.logger.info(f"중복 문서 제거: {len(new_texts) - len(unique_new_texts)}개 중복 문서 제외")
                            
                            self.logger.info(f"전체 문서로 재학습: 기존 {len(existing_texts)}개 (중복 제거됨) + 새 {len(unique_new_texts)}개 = 총 {len(all_texts)}개")
                            # 재학습 (vocabulary 업데이트)
                            self.sparse_embedding_manager.fit(all_texts, include_doc_stats=self.sparse_include_doc_stats)
                            # 학습 후 sparse_embedding 참조 업데이트
                            self.sparse_embedding = self.sparse_embedding_manager.get_sparse_embedding()
                            # vector_store의 sparse_embedding도 업데이트
                            if self.vector_store:
                                if hasattr(self.vector_store, '_sparse_embeddings'):
                                    self.vector_store._sparse_embeddings = self.sparse_embedding
                                    self.logger.info("vector_store._sparse_embeddings 업데이트 완료")
                                elif hasattr(self.vector_store, 'sparse_embedding'):
                                    self.vector_store.sparse_embedding = self.sparse_embedding
                                    self.logger.info("vector_store.sparse_embedding 업데이트 완료")
                            # 학습 상태 확인 로그
                            if self.sparse_embedding and hasattr(self.sparse_embedding, 'corpus_size'):
                                self.logger.info(f"Sparse 임베딩 모델 재학습 완료: corpus_size={self.sparse_embedding.corpus_size}, vocabulary_size={len(self.sparse_embedding.vocabulary)}")
                            else:
                                self.logger.info("Sparse 임베딩 모델 재학습 완료 (vocabulary 업데이트됨)")
                        else:
                            # Qdrant에서 문서를 가져올 수 없으면 현재 문서만으로 재학습
                            self.logger.warning("Qdrant에서 기존 문서를 가져올 수 없어 현재 문서만으로 재학습합니다.")
                            document_texts = [doc.page_content for doc in langchain_docs]
                            self.sparse_embedding_manager.fit(document_texts, include_doc_stats=self.sparse_include_doc_stats)
                            # 학습 후 업데이트
                            self.sparse_embedding = self.sparse_embedding_manager.get_sparse_embedding()
                            if self.vector_store:
                                if hasattr(self.vector_store, '_sparse_embeddings'):
                                    self.vector_store._sparse_embeddings = self.sparse_embedding
                                elif hasattr(self.vector_store, 'sparse_embedding'):
                                    self.vector_store.sparse_embedding = self.sparse_embedding
                    except Exception as e:
                        self.logger.warning(f"Sparse 임베딩 모델 재학습 실패: {str(e)}. 현재 문서만으로 재학습합니다.")
                        document_texts = [doc.page_content for doc in langchain_docs]
                        self.sparse_embedding_manager.fit(document_texts, include_doc_stats=self.sparse_include_doc_stats)
                        # 학습 후 업데이트
                        self.sparse_embedding = self.sparse_embedding_manager.get_sparse_embedding()
                        if self.vector_store:
                            if hasattr(self.vector_store, '_sparse_embeddings'):
                                self.vector_store._sparse_embeddings = self.sparse_embedding
                            elif hasattr(self.vector_store, 'sparse_embedding'):
                                self.vector_store.sparse_embedding = self.sparse_embedding
            
            if force_update:
                # 기존 문서 삭제 후 새로 추가 (중복 방지)
                self.logger.info("기존 문서 삭제 후 새로 추가")
                # LangChain-Qdrant는 upsert를 지원하므로 중복 자동 처리
                # HYBRID 모드일 때 자동으로 dense+sparse 벡터 생성 및 저장
                self.vector_store.add_documents(langchain_docs)
            else:
                # 일반 추가 (중복 방지 로직 적용)
                # HYBRID 모드일 때 자동으로 dense+sparse 벡터 생성 및 저장
                self.vector_store.add_documents(langchain_docs)
            
            self.logger.info(f"문서 추가 완료: {len(langchain_docs)}개 (중복 제거: {len(documents) - len(langchain_docs)}개)")
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
    
    def _delete_document_vectors(self, file_path: str) -> bool:
        """특정 파일의 모든 벡터 삭제"""
        try:
            # Qdrant에서 해당 파일의 모든 포인트 삭제
            result = self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="metadata.source_file",
                            match=MatchValue(value=file_path)
                        )
                    ]
                )
            )
            
            self.logger.info(f"파일 벡터 삭제 완료: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"파일 벡터 삭제 실패: {file_path}, 오류: {str(e)}")
            return False
    
    def search_by_table_title(self, 
                             table_title: str, 
                             limit: Optional[int] = None,
                             score_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        표 제목으로 검색 (레거시 호환성 유지)
        
        Note: 내부적으로 search_with_table_filter를 호출하여 중복 코드 제거
        """
        # search_with_table_filter를 사용하여 동일한 기능 구현
        return self.search_with_table_filter(
            query=table_title,  # 표 제목을 쿼리로 사용
            table_title=table_title,  # 필터로도 사용
            limit=limit,
            score_threshold=score_threshold
        )
    
    def search_with_table_filter(self, 
                                query: str, 
                                table_title: Optional[str] = None,
                                is_table_data: Optional[bool] = None,
                                limit: Optional[int] = None,
                                score_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """표 관련 필터와 함께 검색"""
        if not self._check_connection():
            return []
        
        if self.vector_store is None:
            self.logger.error("LangChain Qdrant 벡터 저장소가 초기화되지 않았습니다")
            return []
        
        # 기본값 적용
        limit = limit if limit is not None else self.default_limit
        # score_threshold는 호출하는 쪽에서 항상 전달되므로 None 체크만 수행
        if score_threshold is None:
            # RAG 설정에서 기본값 가져오기 (호환성)
            from src.utils.config import get_rag_config
            rag_config = get_rag_config()
            score_threshold = rag_config.score_threshold
        
        try:
            filter_conditions = None
            
            # 필터 조건 구성
            if table_title or is_table_data is not None:
                filter_conditions = {"must": []}
                
                if table_title:
                    filter_conditions["must"].append({
                        "key": "table_title",
                        "match": {"value": table_title}
                    })
                
                if is_table_data is not None:
                    filter_conditions["must"].append({
                        "key": "is_table_data",
                        "match": {"value": is_table_data}
                    })
            
            # 검색 모드 및 설정 확인 및 로깅
            retrieval_mode = getattr(self.vector_store, 'retrieval_mode', None)
            sparse_embedding_available = self.sparse_embedding is not None
            
            # vector_store에서 실제 사용되는 sparse_embedding 확인
            vector_store_sparse_embedding = None
            if self.vector_store:
                if hasattr(self.vector_store, '_sparse_embeddings'):
                    vector_store_sparse_embedding = self.vector_store._sparse_embeddings
                elif hasattr(self.vector_store, 'sparse_embedding'):
                    vector_store_sparse_embedding = getattr(self.vector_store, 'sparse_embedding', None)
            
            # 학습 상태 확인
            sparse_model_trained = False
            corpus_size = 0
            vocabulary_size = 0
            if vector_store_sparse_embedding and hasattr(vector_store_sparse_embedding, 'corpus_size'):
                corpus_size = vector_store_sparse_embedding.corpus_size
                vocabulary_size = len(vector_store_sparse_embedding.vocabulary) if hasattr(vector_store_sparse_embedding, 'vocabulary') else 0
                sparse_model_trained = corpus_size > 0
            
            self.logger.info(f"=== Qdrant 검색 시작 ===")
            self.logger.info(f"쿼리: {query[:100]}...")
            self.logger.info(f"Sparse 벡터 활성화: {self.sparse_enabled}")
            self.logger.info(f"Sparse 임베딩 모델 사용 가능: {sparse_embedding_available}")
            self.logger.info(f"Sparse 모델 학습 상태: {sparse_model_trained} (corpus_size={corpus_size}, vocabulary_size={vocabulary_size})")
            self.logger.info(f"RetrievalMode: {retrieval_mode}")
            
            if self.sparse_enabled and retrieval_mode == RetrievalMode.HYBRID:
                self.logger.info(f"✅ 하이브리드 검색 모드: Dense + Sparse 벡터 모두 사용")
                self.logger.info(f"   - Dense 가중치: {self.hybrid_search_dense_weight}")
                self.logger.info(f"   - Sparse 가중치: {self.hybrid_search_sparse_weight}")
            elif self.sparse_enabled and retrieval_mode == RetrievalMode.DENSE:
                self.logger.warning(f"⚠️  Sparse 벡터 활성화되었지만 DENSE 모드로 검색 중 (Sparse 벡터 미사용)")
            elif not self.sparse_enabled:
                self.logger.info(f"ℹ️  Dense 벡터만 사용 (Sparse 벡터 비활성화)")
            else:
                self.logger.warning(f"⚠️  RetrievalMode를 확인할 수 없음: {retrieval_mode}")
            
            if filter_conditions:
                self.logger.debug(f"필터 조건 적용: {filter_conditions}")
                docs = self.vector_store.similarity_search_with_relevance_scores(
                    query=query,
                    k=limit,
                    filter=filter_conditions
                )
            else:
                docs = self.vector_store.similarity_search_with_relevance_scores(
                    query=query,
                    k=limit
                )
            
            self.logger.info(f"검색 결과: {len(docs)}개 문서 반환")
            
            # 결과 변환
            results = []
            for doc, score in docs:
                similarity_score = float(score)
                
                results.append({
                    'content': doc.page_content,
                    'score': similarity_score,
                    'metadata': doc.metadata,
                    'source_file': doc.metadata.get('source_file', ''),
                    'chunk_index': doc.metadata.get('chunk_index', 0),
                    'table_title': doc.metadata.get('table_title', ''),
                    'is_table_data': doc.metadata.get('is_table_data', False)
                })
            
            # 점수 임계값 필터링
            if score_threshold > 0:
                before_filter_count = len(results)
                results = [r for r in results if r['score'] >= score_threshold]
                filtered_out = before_filter_count - len(results)
                if filtered_out > 0:
                    self.logger.info(f"점수 임계값({score_threshold:.3f}) 필터링: {before_filter_count}개 → {len(results)}개 (제외: {filtered_out}개)")
            
            # 검색 결과 상세 로그
            self.logger.info(f"=== Qdrant 검색 완료 ===")
            self.logger.info(f"최종 결과: {len(results)}개 (점수 임계값 필터링 후)")
            if self.sparse_enabled and retrieval_mode == RetrievalMode.HYBRID:
                self.logger.info(f"✅ 하이브리드 검색 완료: Dense + Sparse 벡터 통합 결과")
            if results:
                self.logger.info(f"상위 결과 요약 (최대 3개):")
                for i, result in enumerate(results[:3], 1):
                    source_file = result.get('source_file', '')
                    filename = os.path.basename(source_file) if source_file else 'N/A'
                    chunk_idx = result.get('chunk_index', 'N/A')
                    table_title = result.get('table_title', '')
                    is_table = result.get('is_table_data', False)
                    score = result.get('score', 0.0)
                    content_preview = result.get('content', '')[:50].replace('\n', ' ') + '...' if result.get('content') else 'N/A'
                    
                    table_info = f", 표제목: {table_title}" if table_title else ""
                    table_type = ", 표데이터" if is_table else ""
                    
                    self.logger.info(
                        f"  [{i}] 점수: {score:.4f} | 파일: {filename} | 청크#{chunk_idx}"
                        f"{table_info}{table_type} | 내용: {content_preview}"
                    )
            else:
                self.logger.info("  검색 결과 없음 (점수 임계값 미달 또는 매칭 문서 없음)")
            
            return results
            
        except Exception as e:
            self.logger.error(f"필터 검색 실패: {str(e)}")
            return []
    
    # ========== 비동기 메서드 (Phase 2: 벡터 검색 비동기화) ==========
    
    async def search_with_table_filter_async(self, 
                                            query: str, 
                                            table_title: Optional[str] = None,
                                            is_table_data: Optional[bool] = None,
                                            limit: Optional[int] = None,
                                            score_threshold: Optional[float] = None,
                                            dense_weight: Optional[float] = None,
                                            sparse_weight: Optional[float] = None) -> List[Dict[str, Any]]:
        """비동기 표 관련 필터와 함께 검색"""
        if not self._check_connection():
            return []
        
        if self.vector_store is None:
            self.logger.error("LangChain Qdrant 벡터 저장소가 초기화되지 않았습니다")
            return []
        
        # 기본값 적용
        limit = limit if limit is not None else self.default_limit
        if score_threshold is None:
            from src.utils.config import get_rag_config
            rag_config = get_rag_config()
            score_threshold = rag_config.score_threshold
        
        try:
            filter_conditions = None
            
            # 필터 조건 구성
            if table_title or is_table_data is not None:
                filter_conditions = {"must": []}
                
                if table_title:
                    filter_conditions["must"].append({
                        "key": "table_title",
                        "match": {"value": table_title}
                    })
                
                if is_table_data is not None:
                    filter_conditions["must"].append({
                        "key": "is_table_data",
                        "match": {"value": is_table_data}
                    })
            
            # 검색 모드 및 설정 확인 및 로깅
            retrieval_mode = getattr(self.vector_store, 'retrieval_mode', None)
            sparse_embedding_available = self.sparse_embedding is not None
            
            # vector_store에서 실제 사용되는 sparse_embedding 확인
            vector_store_sparse_embedding = None
            if self.vector_store:
                if hasattr(self.vector_store, '_sparse_embeddings'):
                    vector_store_sparse_embedding = self.vector_store._sparse_embeddings
                elif hasattr(self.vector_store, 'sparse_embedding'):
                    vector_store_sparse_embedding = getattr(self.vector_store, 'sparse_embedding', None)
            
            # 학습 상태 확인
            sparse_model_trained = False
            corpus_size = 0
            vocabulary_size = 0
            if vector_store_sparse_embedding and hasattr(vector_store_sparse_embedding, 'corpus_size'):
                corpus_size = vector_store_sparse_embedding.corpus_size
                vocabulary_size = len(vector_store_sparse_embedding.vocabulary) if hasattr(vector_store_sparse_embedding, 'vocabulary') else 0
                sparse_model_trained = corpus_size > 0
            
            self.logger.info(f"=== Qdrant 비동기 검색 시작 ===")
            self.logger.info(f"쿼리: {query[:100]}...")
            self.logger.info(f"Sparse 벡터 활성화: {self.sparse_enabled}")
            self.logger.info(f"Sparse 임베딩 모델 사용 가능: {sparse_embedding_available}")
            self.logger.info(f"Sparse 모델 학습 상태: {sparse_model_trained} (corpus_size={corpus_size}, vocabulary_size={vocabulary_size})")
            self.logger.info(f"RetrievalMode: {retrieval_mode}")
            
            # 가중치 적용 (API에서 제공된 경우 사용, 없으면 config 기본값)
            effective_dense_weight = dense_weight if dense_weight is not None else self.hybrid_search_dense_weight
            effective_sparse_weight = sparse_weight if sparse_weight is not None else self.hybrid_search_sparse_weight
            
            if self.sparse_enabled and retrieval_mode == RetrievalMode.HYBRID:
                self.logger.info(f"✅ 하이브리드 검색 모드: Dense + Sparse 벡터 모두 사용")
                self.logger.info(f"   - Dense 가중치: {effective_dense_weight} {'(API 제공)' if dense_weight is not None else '(config 기본값)'}")
                self.logger.info(f"   - Sparse 가중치: {effective_sparse_weight} {'(API 제공)' if sparse_weight is not None else '(config 기본값)'}")
                
                # 동적 가중치 적용: Qdrant 클라이언트를 직접 사용하여 하이브리드 검색 수행
                if dense_weight is not None or sparse_weight is not None:
                    self.logger.info(f"🔄 동적 가중치 적용: Qdrant 클라이언트 직접 사용")
                    docs = await self._hybrid_search_with_weights(
                        query=query,
                        limit=limit,
                        filter_conditions=filter_conditions,
                        dense_weight=effective_dense_weight,
                        sparse_weight=effective_sparse_weight
                    )
                else:
                    # 가중치가 제공되지 않으면 LangChain QdrantVectorStore 사용
                    docs = await self._search_with_langchain(
                        query=query,
                        limit=limit,
                        filter_conditions=filter_conditions
                    )
            elif self.sparse_enabled and retrieval_mode == RetrievalMode.DENSE:
                self.logger.warning(f"⚠️  Sparse 벡터 활성화되었지만 DENSE 모드로 검색 중 (Sparse 벡터 미사용)")
                docs = await self._search_with_langchain(
                    query=query,
                    limit=limit,
                    filter_conditions=filter_conditions
                )
            elif not self.sparse_enabled:
                self.logger.info(f"ℹ️  Dense 벡터만 사용 (Sparse 벡터 비활성화)")
                docs = await self._search_with_langchain(
                    query=query,
                    limit=limit,
                    filter_conditions=filter_conditions
                )
            else:
                self.logger.warning(f"⚠️  RetrievalMode를 확인할 수 없음: {retrieval_mode}")
                docs = await self._search_with_langchain(
                    query=query,
                    limit=limit,
                    filter_conditions=filter_conditions
                )
            
            # 결과 변환
            results = []
            for doc, score in docs:
                similarity_score = float(score)
                
                results.append({
                    'content': doc.page_content,
                    'score': similarity_score,
                    'metadata': doc.metadata,
                    'source_file': doc.metadata.get('source_file', ''),
                    'chunk_index': doc.metadata.get('chunk_index', 0),
                    'table_title': doc.metadata.get('table_title', ''),
                    'is_table_data': doc.metadata.get('is_table_data', False)
                })
            
            # 점수 임계값 필터링
            if score_threshold > 0:
                before_filter_count = len(results)
                results = [r for r in results if r['score'] >= score_threshold]
                filtered_out = before_filter_count - len(results)
                if filtered_out > 0:
                    self.logger.info(f"비동기 점수 임계값({score_threshold:.3f}) 필터링: {before_filter_count}개 → {len(results)}개 (제외: {filtered_out}개)")
            
            self.logger.info(f"=== Qdrant 비동기 검색 완료 ===")
            self.logger.info(f"최종 결과: {len(results)}개 (점수 임계값 필터링 후)")
            if self.sparse_enabled and retrieval_mode == RetrievalMode.HYBRID:
                self.logger.info(f"✅ 하이브리드 검색 완료: Dense + Sparse 벡터 통합 결과")
            return results
            
        except Exception as e:
            self.logger.error(f"비동기 필터 검색 실패: {str(e)}")
            return []
    
    async def search_similar_async(self, 
                                  query: str, 
                                  limit: Optional[int] = None,
                                  score_threshold: Optional[float] = None,
                                  filter_conditions: Optional[Dict[str, Any]] = None,
                                  dense_weight: Optional[float] = None,
                                  sparse_weight: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        비동기 유사 문서 검색
        
        Args:
            query: 검색 쿼리
            limit: 반환할 최대 결과 수
            score_threshold: 최소 점수 임계값
            filter_conditions: 필터 조건 (현재 미사용, 호환성 유지)
            dense_weight: Dense 벡터 가중치 (None이면 config 기본값 사용)
            sparse_weight: Sparse 벡터 가중치 (None이면 config 기본값 사용)
        """
        return await self.search_with_table_filter_async(
            query=query,
            limit=limit,
            score_threshold=score_threshold,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight
        )
    
    async def _search_with_langchain(self, query: str, limit: int, filter_conditions: Optional[Dict[str, Any]] = None):
        """LangChain QdrantVectorStore를 사용한 검색 (가중치 미지원)"""
        import asyncio
        
        if self.use_local_storage:
            if filter_conditions:
                return await asyncio.to_thread(
                    self.vector_store.similarity_search_with_relevance_scores,
                    query=query,
                    k=limit,
                    filter=filter_conditions
                )
            else:
                return await asyncio.to_thread(
                    self.vector_store.similarity_search_with_relevance_scores,
                    query=query,
                    k=limit
                )
        else:
            if hasattr(self.vector_store, 'asimilarity_search_with_relevance_scores'):
                if filter_conditions:
                    return await self.vector_store.asimilarity_search_with_relevance_scores(
                        query=query,
                        k=limit,
                        filter=filter_conditions
                    )
                else:
                    return await self.vector_store.asimilarity_search_with_relevance_scores(
                        query=query,
                        k=limit
                    )
            else:
                if filter_conditions:
                    return await asyncio.to_thread(
                        self.vector_store.similarity_search_with_relevance_scores,
                        query=query,
                        k=limit,
                        filter=filter_conditions
                    )
                else:
                    return await asyncio.to_thread(
                        self.vector_store.similarity_search_with_relevance_scores,
                        query=query,
                        k=limit
                    )
    
    async def _hybrid_search_with_weights(self, 
                                         query: str, 
                                         limit: int,
                                         filter_conditions: Optional[Dict[str, Any]] = None,
                                         dense_weight: float = 0.7,
                                         sparse_weight: float = 0.3) -> List[tuple]:
        """
        Qdrant 클라이언트를 직접 사용하여 동적 가중치로 하이브리드 검색 수행
        
        Returns:
            List[tuple]: (Document, score) 튜플 리스트
        """
        import asyncio
        
        try:
            # Dense 벡터 생성
            dense_vector = await asyncio.to_thread(self.embeddings.embed_query, query)
            
            # Sparse 벡터 생성
            sparse_vector_obj = None
            if self.sparse_embedding:
                sparse_vector_obj = await asyncio.to_thread(self.sparse_embedding.embed_query, query)
            
            # Qdrant Query 구성
            query_vector = None
            sparse_query = None
            
            if dense_vector and sparse_vector_obj:
                # 하이브리드 검색: Dense + Sparse
                query_vector = dense_vector
                # SparseVector 객체를 딕셔너리로 변환
                sparse_vector_dict = {
                    "indices": sparse_vector_obj.indices,
                    "values": sparse_vector_obj.values
                }
                sparse_query = NamedSparseVector(
                    name=self.sparse_vector_name,
                    vector=sparse_vector_dict
                )
            elif dense_vector:
                # Dense만 사용
                query_vector = dense_vector
            elif sparse_vector_obj:
                # Sparse만 사용
                # SparseVector 객체를 딕셔너리로 변환
                sparse_vector_dict = {
                    "indices": sparse_vector_obj.indices,
                    "values": sparse_vector_obj.values
                }
                sparse_query = NamedSparseVector(
                    name=self.sparse_vector_name,
                    vector=sparse_vector_dict
                )
            else:
                self.logger.error("Dense와 Sparse 벡터 모두 생성 실패")
                return []
            
            # Qdrant 클라이언트를 직접 사용하여 가중치 기반 하이브리드 검색 수행
            # Prefetch를 사용하여 Dense와 Sparse 검색을 각각 수행한 후 가중치로 결합
            self.logger.info(f"🔄 가중치 기반 하이브리드 검색 수행 (Dense={dense_weight}, Sparse={sparse_weight})")
            
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
            
            # Prefetch를 사용하여 Dense와 Sparse 검색 각각 수행
            # Qdrant는 prefetch 결과를 자동으로 결합하지만, 가중치를 직접 적용하려면
            # 각각 검색 후 수동으로 가중치 결합해야 함
            import asyncio
            
            # Dense 벡터 검색
            dense_results = None
            if query_vector:
                if self.use_local_storage:
                    dense_results = await asyncio.to_thread(
                        self.client.query_points,
                        collection_name=self.collection_name,
                        query=query_vector,
                        using="",  # 기본 벡터 사용
                        limit=limit * 2,  # 가중치 결합을 위해 더 많은 결과 가져오기
                        query_filter=qdrant_filter,
                        with_payload=True,
                        with_vectors=False
                    )
                else:
                    if not hasattr(self, '_async_client') or self._async_client is None:
                        self._async_client = AsyncQdrantClient(host=self._async_client_host, port=self._async_client_port)
                    dense_results = await self._async_client.query_points(
                        collection_name=self.collection_name,
                        query=query_vector,
                        using="",
                        limit=limit * 2,
                        query_filter=qdrant_filter,
                        with_payload=True,
                        with_vectors=False
                    )
            
            # Sparse 벡터 검색
            sparse_results = None
            if sparse_vector_obj:
                sparse_vector_qdrant = QdrantSparseVector(
                    indices=sparse_vector_obj.indices,
                    values=sparse_vector_obj.values
                )
                if self.use_local_storage:
                    sparse_results = await asyncio.to_thread(
                        self.client.query_points,
                        collection_name=self.collection_name,
                        query=sparse_vector_qdrant,
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
                        query=sparse_vector_qdrant,
                        using=self.sparse_vector_name,
                        limit=limit * 2,
                        query_filter=qdrant_filter,
                        with_payload=True,
                        with_vectors=False
                    )
            
            # 가중치로 결과 결합
            combined_results = {}
            
            # Dense 결과 처리
            if dense_results and dense_results.points:
                for point in dense_results.points:
                    point_id = str(point.id)
                    if point_id not in combined_results:
                        combined_results[point_id] = {
                            'point': point,
                            'dense_score': point.score if hasattr(point, 'score') else 0.0,
                            'sparse_score': 0.0,
                            'combined_score': 0.0
                        }
                    else:
                        combined_results[point_id]['dense_score'] = point.score if hasattr(point, 'score') else 0.0
            
            # Sparse 결과 처리
            if sparse_results and sparse_results.points:
                for point in sparse_results.points:
                    point_id = str(point.id)
                    if point_id not in combined_results:
                        combined_results[point_id] = {
                            'point': point,
                            'dense_score': 0.0,
                            'sparse_score': point.score if hasattr(point, 'score') else 0.0,
                            'combined_score': 0.0
                        }
                    else:
                        combined_results[point_id]['sparse_score'] = point.score if hasattr(point, 'score') else 0.0
            
            # 가중치로 최종 점수 계산
            for point_id, result in combined_results.items():
                dense_score = result['dense_score']
                sparse_score = result['sparse_score']
                
                # 가중치 결합: (Dense 점수 × dense_weight) + (Sparse 점수 × sparse_weight)
                # 점수가 0인 경우 해당 검색에서 발견되지 않은 것이므로 가중치를 조정
                if dense_score > 0 and sparse_score > 0:
                    # 둘 다 발견된 경우: 가중치 그대로 적용
                    combined_score = (dense_score * dense_weight) + (sparse_score * sparse_weight)
                elif dense_score > 0:
                    # Dense만 발견된 경우: Dense 가중치만 적용 (정규화)
                    combined_score = dense_score * dense_weight / (dense_weight + sparse_weight) if (dense_weight + sparse_weight) > 0 else dense_score
                elif sparse_score > 0:
                    # Sparse만 발견된 경우: Sparse 가중치만 적용 (정규화)
                    combined_score = sparse_score * sparse_weight / (dense_weight + sparse_weight) if (dense_weight + sparse_weight) > 0 else sparse_score
                else:
                    combined_score = 0.0
                
                result['combined_score'] = combined_score
            
            # 최종 점수 기준으로 정렬
            sorted_results = sorted(
                combined_results.values(),
                key=lambda x: x['combined_score'],
                reverse=True
            )[:limit]
            
            # LangChain Document 형식으로 변환
            docs = []
            for result in sorted_results:
                point = result['point']
                payload = point.payload or {}
                # LangChain QdrantVectorStore는 기본적으로 'page_content' 키를 사용하지만,
                # 우리가 저장할 때는 DocumentChunk의 content를 그대로 저장하므로
                # payload에서 직접 텍스트를 가져오거나, metadata에서 가져와야 함
                # LangChain의 기본 동작을 따라 'page_content' 키를 먼저 확인
                page_content = payload.get('page_content', '')
                if not page_content:
                    # 'page_content'가 없으면 payload의 다른 키들을 확인
                    # 또는 metadata에서 가져오기
                    for key in ['content', 'text', 'body']:
                        if key in payload:
                            page_content = payload[key]
                            break
                
                doc = Document(
                    page_content=page_content,
                    metadata={
                        'chunk_id': payload.get('chunk_id', ''),
                        'source_file': payload.get('source_file', ''),
                        'chunk_index': payload.get('chunk_index', 0),
                        'table_title': payload.get('table_title', ''),
                        'is_table_data': payload.get('is_table_data', False),
                        **{k: v for k, v in payload.items() 
                           if k not in ['page_content', 'chunk_id', 'source_file', 'chunk_index', 'table_title', 'is_table_data']}
                    }
                )
                # 결합된 점수 사용
                docs.append((doc, result['combined_score']))
            
            self.logger.info(f"✅ 가중치 기반 하이브리드 검색 완료: {len(docs)}개 결과")
            self.logger.info(f"   - Dense 가중치: {dense_weight}, Sparse 가중치: {sparse_weight}")
            if docs:
                self.logger.debug(f"   - 상위 결과 점수 범위: {docs[0][1]:.4f} ~ {docs[-1][1]:.4f}")
            return docs
            
        except Exception as e:
            self.logger.error(f"Qdrant 직접 하이브리드 검색 실패: {str(e)}")
            import traceback
            self.logger.error(f"상세 오류: {traceback.format_exc()}")
            # 폴백: LangChain 사용
            self.logger.warning("LangChain QdrantVectorStore로 폴백")
            return await self._search_with_langchain(query, limit, filter_conditions)
    
    def search_similar(self, 
                      query: str, 
                      limit: Optional[int] = None,
                      score_threshold: Optional[float] = None,
                      filter_conditions: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        유사 문서 검색 (기존 호환성 유지)
        
        Args:
            query: 검색 쿼리
            limit: 반환할 최대 결과 수
            score_threshold: 최소 점수 임계값
            filter_conditions: 필터 조건 (현재 미사용, 호환성 유지)
        
        Note: filter_conditions는 현재 사용되지 않음.
             필터가 필요한 경우 search_with_table_filter를 사용하세요.
        """
        return self.search_with_table_filter(
            query=query,
            limit=limit,
            score_threshold=score_threshold
        )
    
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
                
                # 메타데이터 추출 (중복 제거)
                metadata = self._get_metadata(payload)
                
                # 다양한 가능한 키 이름 시도
                source_file = (metadata.get('source_file') or 
                             metadata.get('source') or 
                             metadata.get('file_path') or 
                             metadata.get('file_name') or 
                             'unknown')
                
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
                
                if metadata.get('source_file') == document_id:
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
    
    def _get_all_documents_from_qdrant(self) -> List[Document]:
        """
        Qdrant에서 모든 문서를 가져오기 (LangChain Document 형식)
        
        Returns:
            모든 문서의 LangChain Document 리스트
        """
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
                points = self.client.retrieve(
                    collection_name=self.collection_name,
                    ids=point_ids,
                    with_payload=True,
                    with_vectors=True  # 벡터 데이터 포함
                )
            else:
                # 샘플 포인트 스크롤
                scroll_result = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=sample_size,
                    with_payload=True,
                    with_vectors=True  # 벡터 데이터 포함
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
                if hasattr(point, 'sparse_vectors') and point.sparse_vectors:
                    if isinstance(point.sparse_vectors, dict):
                        for sparse_name, sparse_data in point.sparse_vectors.items():
                            if hasattr(sparse_data, 'indices') and hasattr(sparse_data, 'values'):
                                sample_info['sparse_vectors'][sparse_name] = {
                                    'indices_count': len(sparse_data.indices),
                                    'values_count': len(sparse_data.values),
                                    'indices_preview': list(sparse_data.indices[:10]) if len(sparse_data.indices) > 10 else list(sparse_data.indices),
                                    'values_preview': list(sparse_data.values[:10]) if len(sparse_data.values) > 10 else list(sparse_data.values),
                                    'has_data': True
                                }
                
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
        Sparse 벡터의 Vocabulary 정보 반환
        
        Args:
            limit: 반환할 vocabulary 항목 수 (None이면 전체, 최대 1000개)
            search_token: 특정 토큰 검색 (토큰이 포함된 항목만 반환)
        
        Returns:
            Vocabulary 정보 딕셔너리
        """
        try:
            result = {
                'sparse_enabled': self.sparse_enabled,
                'model_trained': False,
                'corpus_size': 0,
                'vocabulary_size': 0,
                'avgdl': 0.0,
                'vocabulary': {},
                'idf_values': {},
                'statistics': {}
            }
            
            if not self.sparse_enabled:
                result['message'] = 'Sparse 벡터가 비활성화되어 있습니다.'
                return result
            
            # Sparse 임베딩 모델 확인
            sparse_embedding = None
            if self.sparse_embedding:
                sparse_embedding = self.sparse_embedding
            elif self.sparse_embedding_manager:
                sparse_embedding = self.sparse_embedding_manager.get_sparse_embedding()
            
            if not sparse_embedding:
                result['message'] = 'Sparse 임베딩 모델이 초기화되지 않았습니다.'
                return result
            
            # 학습 상태 확인
            if not hasattr(sparse_embedding, 'corpus_size') or sparse_embedding.corpus_size == 0:
                result['message'] = 'Sparse 임베딩 모델이 아직 학습되지 않았습니다. 문서를 먼저 업로드해주세요.'
                return result
            
            result['model_trained'] = True
            result['corpus_size'] = sparse_embedding.corpus_size
            result['avgdl'] = getattr(sparse_embedding, 'avgdl', 0.0)
            
            # Vocabulary 정보 추출
            vocabulary = getattr(sparse_embedding, 'vocabulary', {})
            idf = getattr(sparse_embedding, 'idf', {})
            vocabulary_reverse = getattr(sparse_embedding, 'vocabulary_reverse', {})
            
            result['vocabulary_size'] = len(vocabulary)
            
            # Vocabulary 항목 준비
            vocab_items = []
            for token, idx in vocabulary.items():
                idf_value = idf.get(token, 0.0)
                vocab_items.append({
                    'token': token,
                    'index': idx,
                    'idf': float(idf_value)
                })
            
            # 검색 필터링
            if search_token:
                search_token_lower = search_token.lower()
                vocab_items = [
                    item for item in vocab_items 
                    if search_token_lower in item['token'].lower()
                ]
                result['search_token'] = search_token
                result['filtered_count'] = len(vocab_items)
            
            # 정렬 (IDF 값 기준 내림차순)
            vocab_items.sort(key=lambda x: x['idf'], reverse=True)
            
            # 제한 적용
            if limit is not None:
                vocab_items = vocab_items[:limit]
                result['limit_applied'] = limit
            else:
                # 기본값: 최대 1000개
                if len(vocab_items) > 1000:
                    vocab_items = vocab_items[:1000]
                    result['limit_applied'] = 1000
                    result['message'] = f'Vocabulary가 너무 커서 상위 1000개만 반환합니다. (전체: {len(vocabulary)}개)'
            
            # 결과 구성
            result['vocabulary'] = {item['token']: item['index'] for item in vocab_items}
            result['idf_values'] = {item['token']: item['idf'] for item in vocab_items}
            
            # 통계 정보
            if vocab_items:
                idf_values_list = [item['idf'] for item in vocab_items]
                result['statistics'] = {
                    'total_vocabulary_size': len(vocabulary),
                    'returned_count': len(vocab_items),
                    'idf_min': float(min(idf_values_list)),
                    'idf_max': float(max(idf_values_list)),
                    'idf_mean': float(sum(idf_values_list) / len(idf_values_list)),
                    'top_tokens': [item['token'] for item in vocab_items[:10]]  # 상위 10개 토큰
                }
            
            self.logger.info(f"Sparse vocabulary 조회 완료: {len(vocab_items)}개 항목 반환 (전체: {len(vocabulary)}개)")
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


class VectorStoreManager:
    """벡터 저장소 관리자 (LangChain 기반 또는 Qdrant 레거시)"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, embedding_manager: Optional[EmbeddingManager] = None):
        """
        Args:
            config: Qdrant 설정
            embedding_manager: 기존 EmbeddingManager 인스턴스 (선택적, 중복 로드 방지용)
        """
        self.logger = get_logger()
        
        # 설정 로드
        if config is None:
            from src.utils.config import get_qdrant_config
            qdrant_config = get_qdrant_config()
        else:
            qdrant_config = config
        
        # EmbeddingManager 생성 또는 재사용 (중복 로드 방지)
        if embedding_manager is None:
            embedding_manager = EmbeddingManager()
            self.logger.info("EmbeddingManager 새로 생성")
        else:
            self.logger.info("기존 EmbeddingManager 재사용 (중복 로드 방지)")
        
        # EmbeddingManager를 LangChain Embeddings로 래핑
        langchain_embeddings = EmbeddingManagerWrapper(embedding_manager)
        
        # Qdrant 레거시 지원 (필요시) - 기존 임베딩 재사용
        self.store = QdrantVectorStore(qdrant_config, embeddings=langchain_embeddings)
        
        # LangChain Retrieval Manager 초기화
        try:
            # 설정에서 경로 가져오기
            if hasattr(qdrant_config, 'faiss_storage_path'):
                faiss_storage_path = qdrant_config.faiss_storage_path
            elif isinstance(qdrant_config, dict):
                faiss_storage_path = qdrant_config.get('faiss_storage_path', 'data/faiss_index')
            else:
                faiss_storage_path = 'data/faiss_index'
            
            # BM25 저장 경로 가져오기
            if hasattr(qdrant_config, 'bm25_storage_path'):
                bm25_storage_path = qdrant_config.bm25_storage_path
            elif isinstance(qdrant_config, dict):
                bm25_storage_path = qdrant_config.get('bm25_storage_path', 'data/bm25_index')
            else:
                bm25_storage_path = 'data/bm25_index'
            
            # FAISS GPU 설정 가져오기
            if hasattr(qdrant_config, 'faiss_use_gpu'):
                faiss_use_gpu = qdrant_config.faiss_use_gpu
            elif isinstance(qdrant_config, dict):
                faiss_use_gpu = qdrant_config.get('faiss_use_gpu', True)
            else:
                faiss_use_gpu = True  # 기본값
            
            self.langchain_retrieval_manager = LangChainRetrievalManager(
                embedding_function=langchain_embeddings,
                faiss_storage_path=faiss_storage_path,
                bm25_storage_path=bm25_storage_path,
                faiss_use_gpu=faiss_use_gpu
            )
            
            # 기존 FAISS 인덱스 로드 시도
            faiss_loaded = self.langchain_retrieval_manager.load_faiss_index()
            if faiss_loaded:
                self.logger.info("FAISS 인덱스 자동 로드 완료")
            else:
                self.logger.debug("FAISS 인덱스 없음 (문서 업로드 시 생성됨)")
            
            # 기존 BM25 인덱스 로드 시도
            bm25_loaded = self.langchain_retrieval_manager.load_bm25_index()
            if bm25_loaded:
                self.logger.info("BM25 인덱스 자동 로드 완료")
            else:
                self.logger.debug("BM25 인덱스 없음 (문서 업로드 시 생성됨)")
            
            self.logger.info("LangChain Retrieval Manager 초기화 완료")
            
        except Exception as e:
            self.logger.warning(f"LangChain Retrieval Manager 초기화 실패: {str(e)}. 레거시 모드로 동작합니다.")
            self.langchain_retrieval_manager = None
        
        # 하이브리드 검색 활성화 여부 확인 (레거시 BM25Indexer 제거됨)
        if hasattr(qdrant_config, 'hybrid_search_enabled'):
            hybrid_search_enabled = qdrant_config.hybrid_search_enabled
        elif isinstance(qdrant_config, dict):
            hybrid_search_enabled = qdrant_config.get('hybrid_search_enabled', True)
        else:
            hybrid_search_enabled = True
        
        self.hybrid_search_enabled = hybrid_search_enabled
    
    def setup_collection(self, force_recreate: bool = False) -> bool:
        """컬렉션 설정"""
        return self.store.create_collection(force_recreate)
    
    def add_chunks(self, chunks: List[DocumentChunk], force_update: bool = False) -> bool:
        """청크를 벡터 저장소에 추가 (Qdrant + FAISS + BM25)"""
        # Qdrant에 추가 (레거시 호환)
        qdrant_success = self.store.add_documents(chunks, force_update)
        
        # LangChain FAISS 및 BM25에 추가 (문서 업로드 시)
        # 주의: add_chunks는 업로드 시 호출되지만, FAISS는 전체 청크로 초기 생성하는 것이 더 효율적
        # 따라서 FAISS 인덱스는 rag_system.py의 process_and_store_documents에서 초기화됨
        # 여기서는 FAISS 인덱스가 이미 존재할 때만 문서 추가 시도
        if self.langchain_retrieval_manager:
            try:
                # FAISS 인덱스가 이미 존재하면 문서 추가
                if self.langchain_retrieval_manager.faiss_store is not None:
                    faiss_success = self.langchain_retrieval_manager.add_documents_to_faiss(chunks)
                    if faiss_success:
                        self.logger.debug(f"FAISS에 {len(chunks)}개 청크 추가 완료")
                # FAISS 인덱스가 없으면 초기 생성은 rag_system에서 처리
                
                # BM25 인덱스가 이미 존재하면 문서 추가
                if self.langchain_retrieval_manager.bm25_retriever is not None:
                    bm25_success = self.langchain_retrieval_manager.add_documents_to_bm25(chunks)
                    if bm25_success:
                        self.logger.debug(f"BM25에 {len(chunks)}개 청크 추가 완료")
                # BM25 인덱스가 없으면 초기 생성은 rag_system에서 처리
                
            except Exception as e:
                self.logger.warning(f"FAISS/BM25 문서 추가 실패: {str(e)}")
        
        return qdrant_success
    
    def replace_document_vectors(self, file_path: str, new_chunks: List[DocumentChunk]) -> bool:
        """특정 파일의 벡터를 완전히 교체 (Qdrant + FAISS + BM25)"""
        # Qdrant에서 교체
        qdrant_success = self.store.replace_document_vectors(file_path, new_chunks)
        
        if not qdrant_success:
            return False
        
        # FAISS 및 BM25에서도 교체
        if self.langchain_retrieval_manager:
            try:
                # FAISS에서 기존 문서 삭제 후 새 문서 추가
                # FAISS는 직접 삭제를 지원하지 않으므로, 전체 재구축이 필요하거나
                # 일단 새 문서를 추가하고 나중에 재구축하는 방식을 사용
                if self.langchain_retrieval_manager.faiss_store is not None:
                    # 새 문서 추가
                    faiss_success = self.langchain_retrieval_manager.add_documents_to_faiss(new_chunks)
                    if faiss_success:
                        self.logger.info(f"FAISS에 새 문서 추가 완료: {file_path} ({len(new_chunks)}개)")
                    else:
                        self.logger.warning(f"FAISS 문서 추가 실패: {file_path}")
                        # FAISS 재구축 권장
                        self.logger.warning("FAISS 인덱스 재구축을 권장합니다. /rebuild-indexes API를 사용하세요.")
                
                # BM25에서 기존 문서 삭제 후 새 문서 추가
                if self.langchain_retrieval_manager.bm25_retriever is not None:
                    # 기존 문서 삭제
                    delete_success = self.langchain_retrieval_manager.delete_documents_by_source(file_path)
                    if delete_success:
                        self.logger.debug(f"BM25에서 기존 문서 삭제 완료: {file_path}")
                    
                    # 새 문서 추가
                    bm25_success = self.langchain_retrieval_manager.add_documents_to_bm25(new_chunks)
                    if bm25_success:
                        self.logger.info(f"BM25에 새 문서 추가 완료: {file_path} ({len(new_chunks)}개)")
                    else:
                        self.logger.warning(f"BM25 문서 추가 실패: {file_path}")
                elif self.langchain_retrieval_manager.faiss_store is not None:
                    # BM25 인덱스가 없지만 FAISS는 있는 경우, BM25만 초기화
                    self.logger.info(f"BM25 인덱스가 없어 새로 구축합니다: {file_path}")
                    # Qdrant에서 전체 문서 가져와서 BM25 구축하는 것은 별도 로직 필요
                    # 일단 경고만 남기고, 나중에 재구축하도록 안내
                    self.logger.warning("BM25 인덱스를 구축하려면 전체 문서로 재구축이 필요합니다.")
                
            except Exception as e:
                self.logger.error(f"FAISS/BM25 문서 교체 실패: {str(e)}")
                # Qdrant는 성공했으므로 True 반환 (FAISS/BM25는 경고만)
        
        return qdrant_success
    
    def search_similar(self, 
                      query: str, 
                      limit: Optional[int] = None,
                      score_threshold: Optional[float] = None,
                      filter_conditions: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """유사 문서 검색 (FAISS 우선, 없으면 Qdrant)"""
        # LangChain FAISS 사용 가능 시 우선 사용
        if self.langchain_retrieval_manager and self.langchain_retrieval_manager.faiss_store:
            try:
                results = self.langchain_retrieval_manager.search_with_faiss_only(
                    query=query,
                    k=limit or 10,
                    score_threshold=score_threshold
                )
                if results:
                    self.logger.debug("FAISS 검색 사용")
                    return results
            except Exception as e:
                self.logger.warning(f"FAISS 검색 실패, Qdrant 사용: {str(e)}")
        
        # 레거시 Qdrant 검색
        return self.store.search_similar(query, limit, score_threshold, filter_conditions)
    
    def search_by_table_title(self, 
                             table_title: str, 
                             limit: Optional[int] = None,
                             score_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """표 제목으로 검색"""
        return self.store.search_by_table_title(table_title, limit, score_threshold)
    
    def search_with_table_filter(self, 
                                query: str, 
                                table_title: Optional[str] = None,
                                is_table_data: Optional[bool] = None,
                                limit: Optional[int] = None,
                                score_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """표 관련 필터와 함께 검색"""
        return self.store.search_with_table_filter(
            query, table_title, is_table_data, limit, score_threshold
        )
    
    def get_collection_info(self) -> Dict[str, Any]:
        """컬렉션 정보 반환"""
        return self.store.get_collection_info()
    
    def get_stats(self) -> Dict[str, Any]:
        """저장소 통계 반환 (get_collection_info의 별칭)"""
        return self.get_collection_info()
    
    def inspect_vectors(self, sample_size: int = 3, point_ids: Optional[List[Any]] = None) -> Dict[str, Any]:
        """
        샘플 포인트의 Dense와 Sparse 벡터 확인
        
        Args:
            sample_size: 확인할 샘플 포인트 수 (point_ids가 None일 때)
            point_ids: 확인할 특정 포인트 ID 리스트 (지정 시 sample_size 무시)
            
        Returns:
            벡터 정보 딕셔너리
        """
        return self.store.inspect_vectors(sample_size, point_ids)
    
    def get_vector_statistics(self) -> Dict[str, Any]:
        """
        벡터 통계 정보 반환
        
        Returns:
            벡터 통계 딕셔너리
        """
        return self.store.get_vector_statistics()
    
    def get_sparse_vocabulary(self, limit: Optional[int] = None, search_token: Optional[str] = None) -> Dict[str, Any]:
        """
        Sparse 벡터의 Vocabulary 정보 반환
        
        Args:
            limit: 반환할 vocabulary 항목 수 (None이면 전체)
            search_token: 특정 토큰 검색 (토큰이 포함된 항목만 반환)
        
        Returns:
            Vocabulary 정보 딕셔너리
        """
        return self.store.get_sparse_vocabulary(limit, search_token)
    
    def get_documents_info(self) -> List[Dict[str, Any]]:
        """저장된 문서들의 정보 반환"""
        return self.store.get_documents_info()
    
    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """특정 문서의 청크 정보 반환"""
        return self.store.get_document_chunks(document_id)
    
    def delete_document(self, source_file: str) -> Dict[str, Any]:
        """
        문서 삭제 (Qdrant + FAISS + BM25)
        
        Args:
            source_file: 삭제할 문서의 source_file 경로
            
        Returns:
            삭제 결과 딕셔너리
        """
        result = {
            'success': False,
            'deleted_chunks_count': 0,
            'qdrant_deleted': False,
            'qdrant_success': False,  # 호환성을 위해 유지
            'faiss_deleted': False,
            'faiss_handled': False,  # FAISS 처리 여부
            'bm25_deleted': False,
            'bm25_success': False,  # 호환성을 위해 유지
            'message': '',
            'warnings': []
        }
        
        try:
            # 1. Qdrant에서 삭제 전에 청크 수 확인
            deleted_chunks_count = self.store.get_document_chunks_count(source_file)
            
            # 2. Qdrant에서 삭제
            qdrant_success = self.store._delete_document_vectors(source_file)
            result['qdrant_deleted'] = qdrant_success
            result['qdrant_success'] = qdrant_success
            result['deleted_chunks_count'] = deleted_chunks_count
            
            if not qdrant_success:
                result['message'] = f"Qdrant에서 문서 삭제 실패: {source_file}"
                return result
            
            # 3. BM25에서 삭제
            if self.langchain_retrieval_manager:
                try:
                    bm25_success = self.langchain_retrieval_manager.delete_documents_by_source(source_file)
                    result['bm25_deleted'] = bm25_success
                    result['bm25_success'] = bm25_success
                    if bm25_success:
                        self.logger.info(f"BM25에서 문서 삭제 완료: {source_file}")
                    else:
                        result['warnings'].append("BM25에서 문서를 찾을 수 없거나 이미 삭제되었습니다.")
                except Exception as e:
                    self.logger.error(f"BM25 문서 삭제 실패: {str(e)}")
                    result['warnings'].append(f"BM25 삭제 실패: {str(e)}")
                
                # 4. FAISS에서 삭제 (FAISS는 직접 삭제를 지원하지 않으므로 재구축 필요)
                if self.langchain_retrieval_manager.faiss_store is not None:
                    result['faiss_handled'] = True
                    result['warnings'].append(
                        "FAISS는 직접 삭제를 지원하지 않습니다. "
                        "인덱스 재구축을 권장합니다. /rebuild-indexes API를 사용하세요."
                    )
                    # FAISS 재구축은 사용자가 수동으로 해야 함
            
            result['success'] = True
            result['message'] = f"문서 삭제 완료: {source_file}"
            
            return result
            
        except Exception as e:
            self.logger.error(f"문서 삭제 중 오류: {str(e)}")
            result['message'] = f"문서 삭제 실패: {str(e)}"
            return result
    
    def build_bm25_index(self, chunks: List[DocumentChunk]) -> bool:
        """BM25 인덱스 구축 (LangChain BM25Retriever만 사용)"""
        if not self.langchain_retrieval_manager:
            self.logger.warning("LangChain Retrieval Manager가 초기화되지 않았습니다. BM25 인덱스를 구축할 수 없습니다.")
            return False
        
        try:
            success = self.langchain_retrieval_manager.initialize_bm25_from_chunks(chunks)
            if success:
                self.logger.info("LangChain BM25Retriever 초기화 완료")
            return success
        except Exception as e:
            self.logger.error(f"LangChain BM25Retriever 초기화 실패: {str(e)}")
            import traceback
            self.logger.error(f"상세 오류: {traceback.format_exc()}")
            return False
    
    def hybrid_search(self,
                     query: str,
                     limit: int = 10,
                     score_threshold: Optional[float] = None,
                     vector_weight: Optional[float] = None,
                     bm25_weight: Optional[float] = None,
                     rrf_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        하이브리드 검색 (EnsembleRetriever 또는 레거시 RRF)
        
        Args:
            query: 검색 쿼리
            limit: 반환할 최대 결과 수
            score_threshold: 최소 점수 임계값
            vector_weight: 벡터 검색 가중치 (None이면 설정값 사용)
            bm25_weight: BM25 검색 가중치 (None이면 설정값 사용)
            rrf_k: RRF 알고리즘 상수 (None이면 설정값 사용)
            
        Returns:
            통합된 검색 결과 리스트
        """
        if not self.hybrid_search_enabled:
            self.logger.warning("하이브리드 검색이 비활성화되어 있습니다. 벡터 검색만 수행합니다.")
            return self.search_similar(query, limit, score_threshold)
        
        # LangChain EnsembleRetriever 사용 시도
        if self.langchain_retrieval_manager:
            try:
                # 설정값 가져오기
                from src.utils.config import get_qdrant_config
                qdrant_config = get_qdrant_config()
                
                faiss_weight = vector_weight if vector_weight is not None else (qdrant_config.hybrid_search_vector_weight if hasattr(qdrant_config, 'hybrid_search_vector_weight') else 0.7)
                bm25_weight_val = bm25_weight if bm25_weight is not None else (qdrant_config.hybrid_search_bm25_weight if hasattr(qdrant_config, 'hybrid_search_bm25_weight') else 0.3)
                rrf_c = rrf_k if rrf_k is not None else (qdrant_config.hybrid_search_rrf_k if hasattr(qdrant_config, 'hybrid_search_rrf_k') else 60)
                
                # EnsembleRetriever 생성 (없으면 생성)
                if self.langchain_retrieval_manager.ensemble_retriever is None:
                    self.langchain_retrieval_manager.create_ensemble_retriever(
                        faiss_weight=faiss_weight,
                        bm25_weight=bm25_weight_val,
                        c=rrf_c,
                        k=limit
                    )
                
                # EnsembleRetriever 검색
                if self.langchain_retrieval_manager.ensemble_retriever:
                    results = self.langchain_retrieval_manager.search_with_ensemble(
                        query=query,
                        k=limit,
                        score_threshold=score_threshold
                    )
                    if results:
                        self.logger.info(f"EnsembleRetriever 검색 완료: {len(results)}개 결과")
                        return results
                
            except Exception as e:
                self.logger.error(f"EnsembleRetriever 검색 실패: {str(e)}")
                import traceback
                self.logger.error(f"상세 오류: {traceback.format_exc()}")
        
        # EnsembleRetriever가 사용 불가능한 경우 벡터 검색만 사용
        self.logger.warning("EnsembleRetriever를 사용할 수 없어 벡터 검색만 수행합니다.")
        return self.search_similar(query, limit, score_threshold)
    
    # ========== 비동기 메서드 (Phase 2: 벡터 검색 비동기화) ==========
    
    async def search_similar_async(self, 
                                  query: str, 
                                  limit: Optional[int] = None,
                                  score_threshold: Optional[float] = None,
                                  filter_conditions: Optional[Dict[str, Any]] = None,
                                  dense_weight: Optional[float] = None,
                                  sparse_weight: Optional[float] = None) -> List[Dict[str, Any]]:
        """비동기 유사 문서 검색 (FAISS 우선, 없으면 Qdrant)"""
        # LangChain FAISS 사용 가능 시 우선 사용 (동기 유지 - FAISS는 CPU/GPU 연산)
        if self.langchain_retrieval_manager and self.langchain_retrieval_manager.faiss_store:
            try:
                # FAISS는 CPU/GPU 연산이므로 asyncio.to_thread로 비동기화
                import asyncio
                results = await asyncio.to_thread(
                    self.langchain_retrieval_manager.search_with_faiss_only,
                    query=query,
                    k=limit or 10,
                    score_threshold=score_threshold
                )
                if results:
                    self.logger.debug("비동기 FAISS 검색 사용")
                    return results
            except Exception as e:
                self.logger.warning(f"비동기 FAISS 검색 실패, Qdrant 사용: {str(e)}")
        
        # Qdrant 비동기 검색
        return await self.store.search_similar_async(
            query, limit, score_threshold, filter_conditions, dense_weight, sparse_weight
        )
    
    async def hybrid_search_async(self,
                                 query: str,
                                 limit: int = 10,
                                 score_threshold: Optional[float] = None,
                                 vector_weight: Optional[float] = None,
                                 bm25_weight: Optional[float] = None,
                                 rrf_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        비동기 하이브리드 검색 (EnsembleRetriever 또는 레거시 RRF)
        """
        if not self.hybrid_search_enabled:
            self.logger.warning("하이브리드 검색이 비활성화되어 있습니다. 벡터 검색만 수행합니다.")
            return await self.search_similar_async(query, limit, score_threshold)
        
        # LangChain EnsembleRetriever 사용 시도 (동기 메서드를 비동기로 실행)
        if self.langchain_retrieval_manager:
            try:
                from src.utils.config import get_qdrant_config
                qdrant_config = get_qdrant_config()
                
                faiss_weight = vector_weight if vector_weight is not None else (qdrant_config.hybrid_search_vector_weight if hasattr(qdrant_config, 'hybrid_search_vector_weight') else 0.7)
                bm25_weight_val = bm25_weight if bm25_weight is not None else (qdrant_config.hybrid_search_bm25_weight if hasattr(qdrant_config, 'hybrid_search_bm25_weight') else 0.3)
                rrf_c = rrf_k if rrf_k is not None else (qdrant_config.hybrid_search_rrf_k if hasattr(qdrant_config, 'hybrid_search_rrf_k') else 60)
                
                # EnsembleRetriever 생성 (없으면 생성)
                if self.langchain_retrieval_manager.ensemble_retriever is None:
                    self.langchain_retrieval_manager.create_ensemble_retriever(
                        faiss_weight=faiss_weight,
                        bm25_weight=bm25_weight_val,
                        c=rrf_c,
                        k=limit
                    )
                
                # EnsembleRetriever 검색 (비동기로 실행)
                if self.langchain_retrieval_manager.ensemble_retriever:
                    import asyncio
                    results = await asyncio.to_thread(
                        self.langchain_retrieval_manager.search_with_ensemble,
                        query=query,
                        k=limit,
                        score_threshold=score_threshold
                    )
                    if results:
                        self.logger.info(f"비동기 EnsembleRetriever 검색 완료: {len(results)}개 결과")
                        return results
                
            except Exception as e:
                self.logger.error(f"비동기 EnsembleRetriever 검색 실패: {str(e)}")
                import traceback
                self.logger.error(f"상세 오류: {traceback.format_exc()}")
        
        # EnsembleRetriever가 사용 불가능한 경우 벡터 검색만 사용
        self.logger.warning("EnsembleRetriever를 사용할 수 없어 벡터 검색만 수행합니다.")
        return await self.search_similar_async(query, limit, score_threshold)
    
    def _merge_with_rrf(self,
                       vector_results: List[Dict[str, Any]],
                       bm25_results: List[Dict[str, Any]],
                       rrf_k: int = 60,
                       limit: int = 10,
                       score_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion (RRF) 알고리즘으로 검색 결과 통합 (레거시 - 더 이상 사용되지 않음)
        
        주의: 이 메서드는 레거시 BM25Indexer와 함께 사용되었지만,
        현재는 LangChain EnsembleRetriever가 RRF를 자동으로 처리하므로 사용되지 않습니다.
        
        RRF 점수 = Σ 1 / (k + rank)
        - k: 상수 (일반적으로 60)
        - rank: 각 검색 방법에서의 순위
        """
        # 청크 ID를 키로 하는 RRF 점수 딕셔너리
        rrf_scores: Dict[str, float] = {}
        result_data: Dict[str, Dict[str, Any]] = {}
        
        # 벡터 검색 결과 점수 추가
        for rank, result in enumerate(vector_results, 1):
            # chunk_id 추출 (여러 가능한 키 시도)
            chunk_id = (result.get('chunk_id') or 
                       result.get('id') or 
                       result.get('metadata', {}).get('chunk_id'))
            
            if chunk_id:
                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (rrf_k + rank)
                result_data[chunk_id] = result
        
        # BM25 검색 결과 점수 추가
        for rank, result in enumerate(bm25_results, 1):
            chunk_id = result.get('chunk_id')
            if chunk_id:
                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (rrf_k + rank)
                
                # 벡터 검색에 없던 결과이면 데이터 추가
                if chunk_id not in result_data:
                    # BM25 결과를 벡터 검색 형식으로 변환
                    result_data[chunk_id] = {
                        'content': result.get('content', ''),
                        'score': result.get('score', 0),
                        'metadata': result.get('metadata', {}),
                        'source_file': result.get('source_file', ''),
                        'chunk_index': result.get('chunk_index', 0),
                        'chunk_id': chunk_id,
                        'rrf_score': rrf_scores[chunk_id]
                    }
        
        # RRF 점수 기준 정렬
        sorted_chunk_ids = sorted(
            rrf_scores.keys(),
            key=lambda x: rrf_scores[x],
            reverse=True
        )
        
        # 상위 K개 결과 반환
        results = []
        for chunk_id in sorted_chunk_ids:
            result = result_data.get(chunk_id)
            if not result:
                continue
            
            # RRF 점수를 최종 점수로 사용
            result['score'] = rrf_scores[chunk_id]
            result['rrf_score'] = rrf_scores[chunk_id]
            result['vector_score'] = result.get('score', 0) if chunk_id in [r.get('chunk_id') for r in vector_results] else None
            result['bm25_score'] = next((r.get('score') for r in bm25_results if r.get('chunk_id') == chunk_id), None)
            
            # 점수 임계값 필터링
            if score_threshold is not None and result['score'] < score_threshold:
                continue
            
            results.append(result)
            
            if len(results) >= limit:
                break
        
        self.logger.info(
            f"하이브리드 검색 완료: 벡터={len(vector_results)}개, "
            f"BM25={len(bm25_results)}개, 통합={len(results)}개"
        )
        
        return results


def create_vector_store_manager(config: Optional[Dict[str, Any]] = None) -> VectorStoreManager:
    """벡터 저장소 관리자 생성"""
    return VectorStoreManager(config)


def setup_vector_store(config: Optional[Dict[str, Any]] = None, force_recreate: bool = False) -> bool:
    """벡터 저장소 설정"""
    manager = create_vector_store_manager(config)
    return manager.setup_collection(force_recreate)
