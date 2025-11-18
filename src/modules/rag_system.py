"""
RAG 시스템 모듈
문서 처리, 임베딩, 벡터 검색, 답변 생성을 통합한 RAG 시스템
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time

from src.utils.logger import get_logger
from src.utils.config import get_config, get_rag_config
from src.utils.helpers import is_general_question
from src.modules.document_processor import DocumentProcessor, DocumentChunk
from src.modules.embedding_module import EmbeddingManager
from src.modules.vector_store import VectorStoreManager
from src.models.llm_models import OllamaLLMClient
from src.modules.reranker_module import CrossEncoderReranker


@dataclass
class RAGResponse:
    """RAG 응답"""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    processing_time: float
    query: str
    model_used: str
    is_general_answer: bool = False  # 일반 답변 여부
    is_rag_answer: bool = True  # RAG 답변 여부 (기본값 True)


class RAGSystem:
    """RAG 시스템"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = get_logger()
        
        if config is None:
            config = get_config()
        
        self.config = config
        self.rag_config = get_rag_config()
        
        print(f"  임베딩 모델: {self.config.model.get('embedding', {}).name if 'embedding' in config.model else 'N/A'}")
        print(f"  LLM 모델: {self.config.model.get('llm', {}).name if 'llm' in config.model else 'N/A'}")
        # 모듈 초기화
        self.document_processor = DocumentProcessor()
        # EmbeddingManager를 먼저 생성하고 VectorStoreManager에 전달하여 중복 로드 방지
        self.embedding_manager = EmbeddingManager()
        self.vector_store_manager = VectorStoreManager(config.qdrant, embedding_manager=self.embedding_manager)
        
        # 리랭커 초기화 (설정 기반)
        self.reranker: Optional[CrossEncoderReranker] = None
        try:
            reranker_cfg = getattr(self.config, 'reranker', None)
            if reranker_cfg is None:
                self.logger.info("리랭커 설정이 없습니다. 리랭커를 비활성화합니다.")
                self.reranker = None
            else:
                # Pydantic 모델 또는 dict 모두 지원
                enabled = (reranker_cfg.enabled if hasattr(reranker_cfg, 'enabled') 
                          else reranker_cfg.get('enabled', False)) if reranker_cfg else False
                
                if enabled:
                    # 설정 파싱 방어코드 (Pydantic 모델/dict 혼용)
                    model_path = (reranker_cfg.model_path if hasattr(reranker_cfg, 'model_path') 
                                 else reranker_cfg.get('model_path', '')) if reranker_cfg else ''
                    device = (reranker_cfg.device if hasattr(reranker_cfg, 'device') 
                             else reranker_cfg.get('device', 'cuda')) if reranker_cfg else 'cuda'
                    batch_size = (reranker_cfg.batch_size if hasattr(reranker_cfg, 'batch_size') 
                                 else reranker_cfg.get('batch_size', 32)) if reranker_cfg else 32
                    
                    self.logger.info(
                        f"리랭커 설정 확인: enabled={enabled}, model_path={model_path}, "
                        f"device={device}, batch_size={batch_size}"
                    )
                    
                    if model_path:
                        try:
                            self.reranker = CrossEncoderReranker(
                                model_path=model_path,
                                device=device,
                                batch_size=batch_size,
                            )
                            self.logger.info(
                                f"✅ 리랭커 초기화 완료: path={model_path}, device={self.reranker.device}, batch_size={batch_size}"
                            )
                        except Exception as reranker_error:
                            self.logger.error(f"리랭커 모델 로드 실패: {str(reranker_error)}")
                            self.reranker = None
                    else:
                        self.logger.warning("리랭커가 활성화되어 있지만 model_path가 비어 있습니다. 리랭커를 비활성화합니다.")
                        self.reranker = None
                else:
                    self.logger.info("리랭커 비활성화 상태 (설정 enabled=False)")
                    self.reranker = None
        except Exception as e:
            self.logger.error(f"리랭커 초기화 중 예외 발생: {str(e)}", exc_info=True)
            self.reranker = None
        
        # LLM 클라이언트 설정 가져오기
        llm_config = config.model.get('llm')
        if isinstance(llm_config, dict):
            self.llm_client = OllamaLLMClient(llm_config)
        else:
            self.llm_client = OllamaLLMClient(llm_config)
        
        # FAISS 및 BM25 인덱스 자동 로드 (VectorStoreManager에서 이미 시도했지만, 상태 확인 및 로깅)
        if hasattr(self.vector_store_manager, 'langchain_retrieval_manager') and \
           self.vector_store_manager.langchain_retrieval_manager:
            langchain_manager = self.vector_store_manager.langchain_retrieval_manager
            
            # 인덱스 상태 검증
            qdrant_stats = self.vector_store_manager.get_stats()
            qdrant_doc_count = qdrant_stats.get('points_count', None)
            
            validation_result = langchain_manager.validate_indexes(qdrant_doc_count)
            
            if validation_result['warnings']:
                for warning in validation_result['warnings']:
                    self.logger.warning(warning)
                self.logger.info("인덱스 재구축을 권장합니다. /rebuild-indexes API를 사용하세요.")
            else:
                if validation_result['faiss_available']:
                    self.logger.info(f"FAISS 인덱스 준비 완료 (문서 {validation_result['faiss_document_count']}개)")
                if validation_result['bm25_available']:
                    self.logger.info(f"BM25 인덱스 준비 완료 (문서 {validation_result['bm25_document_count']}개)")
        
        self.logger.info("RAG 시스템이 초기화되었습니다")
    
    def _release_gpu_memory(self):
        """GPU 메모리 해제 (모델 언로드)"""
        import torch
        import gc
        
        self.logger.info("GPU 메모리 해제 시작...")
        
        # 임베딩 모델 해제
        if hasattr(self, 'embedding_manager') and self.embedding_manager:
            if hasattr(self.embedding_manager, 'client') and hasattr(self.embedding_manager.client, 'model'):
                try:
                    del self.embedding_manager.client.model
                    self.logger.info("임베딩 모델 메모리 해제 완료")
                except Exception as e:
                    self.logger.warning(f"임베딩 모델 메모리 해제 실패: {str(e)}")
        
        # 리랭커 모델 해제
        if self.reranker and hasattr(self.reranker, 'model'):
            try:
                del self.reranker.model
                self.logger.info("리랭커 모델 메모리 해제 완료")
            except Exception as e:
                self.logger.warning(f"리랭커 모델 메모리 해제 실패: {str(e)}")
        
        # PyTorch 캐시 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            self.logger.info("PyTorch CUDA 캐시 정리 완료")
        
        # Python 가비지 컬렉션
        gc.collect()
        self.logger.info("GPU 메모리 해제 완료")
    
    def reload_embedding_model(self, config: Optional[Dict[str, Any]] = None):
        """임베딩 모델 동적 재로드"""
        self.logger.info("임베딩 모델 재로드 시작...")
        
        # 기존 모델 해제
        if hasattr(self, 'embedding_manager') and self.embedding_manager:
            if hasattr(self.embedding_manager, 'client') and hasattr(self.embedding_manager.client, 'model'):
                del self.embedding_manager.client.model
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # 새 모델 로드
        try:
            if config is None:
                from src.utils.config import get_embedding_config
                config = get_embedding_config()
            
            self.embedding_manager = EmbeddingManager(config)
            
            # VectorStoreManager도 업데이트
            self.vector_store_manager.embedding_manager = self.embedding_manager
            from src.modules.langchain_embedding_wrapper import EmbeddingManagerWrapper
            langchain_embeddings = EmbeddingManagerWrapper(self.embedding_manager)
            self.vector_store_manager.store.embeddings = langchain_embeddings
            
            self.logger.info("임베딩 모델 재로드 완료")
            return True
        except Exception as e:
            self.logger.error(f"임베딩 모델 재로드 실패: {str(e)}")
            return False
    
    def reload_reranker(self, config: Optional[Dict[str, Any]] = None):
        """리랭커 모델 동적 재로드"""
        self.logger.info("리랭커 모델 재로드 시작...")
        
        # 기존 리랭커 해제
        if self.reranker and hasattr(self.reranker, 'model'):
            del self.reranker.model
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.reranker = None
        
        # 새 리랭커 로드
        try:
            if config is None:
                reranker_cfg = getattr(self.config, 'reranker', None)
            else:
                reranker_cfg = config
            
            if reranker_cfg:
                enabled = (reranker_cfg.enabled if hasattr(reranker_cfg, 'enabled') 
                          else reranker_cfg.get('enabled', False)) if reranker_cfg else False
                
                if enabled:
                    model_path = (reranker_cfg.model_path if hasattr(reranker_cfg, 'model_path') 
                                 else reranker_cfg.get('model_path', '')) if reranker_cfg else ''
                    device = (reranker_cfg.device if hasattr(reranker_cfg, 'device') 
                             else reranker_cfg.get('device', 'cuda')) if reranker_cfg else 'cuda'
                    batch_size = (reranker_cfg.batch_size if hasattr(reranker_cfg, 'batch_size') 
                                 else reranker_cfg.get('batch_size', 32)) if reranker_cfg else 32
                    
                    if model_path:
                        self.reranker = CrossEncoderReranker(
                            model_path=model_path,
                            device=device,
                            batch_size=batch_size,
                        )
                        self.logger.info("리랭커 모델 재로드 완료")
                        return True
            
            self.logger.info("리랭커 비활성화됨")
            return True
        except Exception as e:
            self.logger.error(f"리랭커 모델 재로드 실패: {str(e)}")
            return False
    
    def delete_document(self, source_file: str) -> Dict[str, Any]:
        """
        특정 문서를 모든 저장소에서 삭제
        
        Args:
            source_file: 삭제할 문서의 소스 파일 경로
            
        Returns:
            삭제 결과 딕셔너리
        """
        try:
            self.logger.info(f"문서 삭제 시작: {source_file}")
            
            # VectorStoreManager를 통해 삭제 (Qdrant + BM25 + FAISS 처리)
            result = self.vector_store_manager.delete_document(source_file)
            
            # 임베딩 캐시는 메모리 기반이므로 서버 재시작 시 자동 초기화됨
            # 특정 문서의 캐시만 삭제하는 것은 복잡하므로 선택적 기능으로 제외
            
            qdrant_success = result.get('qdrant_success', result.get('qdrant_deleted', False))
            if qdrant_success:
                deleted_count = result.get('deleted_chunks_count', 0)
                self.logger.info(
                    f"문서 삭제 완료: {source_file} "
                    f"(청크 {deleted_count}개 삭제)"
                )
            else:
                self.logger.error(f"문서 삭제 실패: {source_file}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"문서 삭제 중 예외 발생: {source_file}, 오류: {str(e)}")
            import traceback
            self.logger.error(f"상세 오류: {traceback.format_exc()}")
            return {
                'qdrant_success': False,
                'bm25_success': False,
                'faiss_handled': False,
                'deleted_chunks_count': 0,
                'warnings': [f"삭제 중 예외 발생: {str(e)}"]
            }
    
    def process_and_store_documents(self, input_dir: str, force_update: bool = False, replace_mode: bool = False, build_bm25_index: bool = True) -> bool:
        """문서 처리 및 저장"""
        try:
            self.logger.info(f"문서 처리 시작: {input_dir}")
            
            # 1. 문서 처리
            chunks = self.document_processor.process_directory(input_dir, force_update)
            if not chunks:
                self.logger.error("처리할 문서가 없습니다")
                return False
            
            self.logger.info(f"문서 청킹 완료: {len(chunks)}개 청크")
            
            # 2. 벡터 저장소에 저장 (교체 모드 또는 일반 모드)
            # Sparse 벡터는 QdrantVectorStore.add_documents에서 자동으로 처리됨
            if replace_mode:
                # 교체 모드: 파일별로 완전 교체
                success = self._process_chunks_in_replace_mode(chunks)
            else:
                # 일반 모드: 기존 방식 (Qdrant + FAISS)
                # Qdrant에 저장 시 sparse_enabled이면 자동으로 dense+sparse 벡터 함께 저장
                success = self.vector_store_manager.add_chunks(chunks, force_update)
            
            if not success:
                self.logger.error("벡터 저장소 저장 실패")
                return False
            
            # FAISS 인덱스가 없으면 생성
            if hasattr(self.vector_store_manager, 'langchain_retrieval_manager') and \
               self.vector_store_manager.langchain_retrieval_manager:
                if self.vector_store_manager.langchain_retrieval_manager.faiss_store is None:
                    self.logger.info("FAISS 인덱스 생성 중...")
                    faiss_success = self.vector_store_manager.langchain_retrieval_manager.initialize_faiss_from_chunks(chunks)
                    if faiss_success:
                        self.logger.info("FAISS 인덱스 생성 완료")
                    else:
                        self.logger.warning("FAISS 인덱스 생성 실패 (Qdrant는 정상 동작)")
            
            # 3. BM25 인덱스 구축 (LangChain BM25Retriever 또는 레거시)
            if build_bm25_index:
                self.logger.info("BM25 인덱스 구축 시작...")
                bm25_success = self.vector_store_manager.build_bm25_index(chunks)
                if bm25_success:
                    self.logger.info("BM25 인덱스 구축 완료")
                else:
                    self.logger.warning("BM25 인덱스 구축 실패 (검색은 계속 진행됩니다)")
                
                # LangChain EnsembleRetriever 초기화 시도
                if hasattr(self.vector_store_manager, 'langchain_retrieval_manager') and \
                   self.vector_store_manager.langchain_retrieval_manager:
                    try:
                        from src.utils.config import get_qdrant_config
                        qdrant_config = get_qdrant_config()
                        faiss_weight = qdrant_config.hybrid_search_vector_weight if hasattr(qdrant_config, 'hybrid_search_vector_weight') else 0.7
                        bm25_weight = qdrant_config.hybrid_search_bm25_weight if hasattr(qdrant_config, 'hybrid_search_bm25_weight') else 0.3
                        rrf_c = qdrant_config.hybrid_search_rrf_k if hasattr(qdrant_config, 'hybrid_search_rrf_k') else 60
                        
                        self.vector_store_manager.langchain_retrieval_manager.create_ensemble_retriever(
                            faiss_weight=faiss_weight,
                            bm25_weight=bm25_weight,
                            c=rrf_c,
                            k=self.rag_config.default_max_sources
                        )
                        self.logger.info("EnsembleRetriever 초기화 완료")
                    except Exception as e:
                        self.logger.warning(f"EnsembleRetriever 초기화 실패: {str(e)}")
            
            self.logger.info("문서 처리 및 저장 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"문서 처리 및 저장 실패: {str(e)}")
            return False
    
    async def process_and_store_documents_async(self, input_dir: str, force_update: bool = False, replace_mode: bool = False, build_bm25_index: bool = True) -> bool:
        """문서 처리 및 저장 (비동기)"""
        import asyncio
        
        try:
            self.logger.info(f"비동기 문서 처리 시작: {input_dir}")
            
            # 1. 문서 처리 (I/O 작업 - 비동기화)
            chunks = await asyncio.to_thread(
                self.document_processor.process_directory,
                input_dir,
                force_update
            )
            if not chunks:
                self.logger.error("처리할 문서가 없습니다")
                return False
            
            self.logger.info(f"문서 청킹 완료: {len(chunks)}개 청크")
            
            # 2. 벡터 저장소에 저장 (I/O 작업 - 비동기화)
            if replace_mode:
                # 교체 모드: 파일별로 완전 교체
                success = await asyncio.to_thread(
                    self._process_chunks_in_replace_mode,
                    chunks
                )
            else:
                # 일반 모드: 기존 방식 (Qdrant + FAISS)
                success = await asyncio.to_thread(
                    self.vector_store_manager.add_chunks,
                    chunks,
                    force_update
                )
            
            if not success:
                self.logger.error("벡터 저장소 저장 실패")
                return False
            
            # FAISS 인덱스가 없으면 생성 (I/O 작업 - 비동기화)
            if hasattr(self.vector_store_manager, 'langchain_retrieval_manager') and \
               self.vector_store_manager.langchain_retrieval_manager:
                langchain_manager = self.vector_store_manager.langchain_retrieval_manager
                if langchain_manager.faiss_store is None:
                    self.logger.info("FAISS 인덱스 생성 중...")
                    faiss_success = await asyncio.to_thread(
                        langchain_manager.initialize_faiss_from_chunks,
                        chunks
                    )
                    if faiss_success:
                        self.logger.info("FAISS 인덱스 생성 완료")
                    else:
                        self.logger.warning("FAISS 인덱스 생성 실패 (Qdrant는 정상 동작)")
            
            # 3. BM25 인덱스 구축 (I/O 작업 - 비동기화)
            if build_bm25_index:
                self.logger.info("BM25 인덱스 구축 시작...")
                bm25_success = await asyncio.to_thread(
                    self.vector_store_manager.build_bm25_index,
                    chunks
                )
                if bm25_success:
                    self.logger.info("BM25 인덱스 구축 완료")
                else:
                    self.logger.warning("BM25 인덱스 구축 실패 (검색은 계속 진행됩니다)")
                
                # LangChain EnsembleRetriever 초기화 시도
                if hasattr(self.vector_store_manager, 'langchain_retrieval_manager') and \
                   self.vector_store_manager.langchain_retrieval_manager:
                    try:
                        from src.utils.config import get_qdrant_config
                        qdrant_config = get_qdrant_config()
                        faiss_weight = qdrant_config.hybrid_search_vector_weight if hasattr(qdrant_config, 'hybrid_search_vector_weight') else 0.7
                        bm25_weight = qdrant_config.hybrid_search_bm25_weight if hasattr(qdrant_config, 'hybrid_search_bm25_weight') else 0.3
                        rrf_c = qdrant_config.hybrid_search_rrf_k if hasattr(qdrant_config, 'hybrid_search_rrf_k') else 60
                        
                        await asyncio.to_thread(
                            self.vector_store_manager.langchain_retrieval_manager.create_ensemble_retriever,
                            faiss_weight=faiss_weight,
                            bm25_weight=bm25_weight,
                            c=rrf_c,
                            k=self.rag_config.default_max_sources
                        )
                        self.logger.info("EnsembleRetriever 초기화 완료")
                    except Exception as e:
                        self.logger.warning(f"EnsembleRetriever 초기화 실패: {str(e)}")
            
            self.logger.info("비동기 문서 처리 및 저장 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"비동기 문서 처리 및 저장 실패: {str(e)}")
            return False
    
    def _process_chunks_in_replace_mode(self, chunks: List[DocumentChunk]) -> bool:
        """교체 모드에서 청크 처리 (파일별로 완전 교체)"""
        try:
            import time
            total_start_time = time.time()
            
            # 파일별로 청크 그룹화
            file_chunks = {}
            for chunk in chunks:
                file_path = chunk.source_file
                if file_path not in file_chunks:
                    file_chunks[file_path] = []
                file_chunks[file_path].append(chunk)
            
            self.logger.info(f"교체 모드 처리 시작: 총 {len(file_chunks)}개 파일, {len(chunks)}개 청크")
            
            # 각 파일별로 완전 교체
            success_count = 0
            for idx, (file_path, file_chunk_list) in enumerate(file_chunks.items(), 1):
                filename = file_path.split('\\')[-1] if '\\' in file_path else file_path
                filename = filename.split('/')[-1] if '/' in filename else filename
                
                self.logger.info(
                    f"파일 처리 중: {idx}/{len(file_chunks)} | "
                    f"파일: {filename} | 청크 수: {len(file_chunk_list)}"
                )
                
                file_start_time = time.time()
                success = self.vector_store_manager.replace_document_vectors(file_path, file_chunk_list)
                file_time = time.time() - file_start_time
                
                if success:
                    success_count += 1
                    self.logger.info(
                        f"파일 교체 완료: {filename} | 청크 수: {len(file_chunk_list)} | "
                        f"처리 시간: {file_time:.2f}초"
                    )
                else:
                    self.logger.error(f"파일 교체 실패: {filename}")
            
            total_time = time.time() - total_start_time
            self.logger.info(
                f"교체 모드 처리 완료: {success_count}/{len(file_chunks)}개 파일 성공 | "
                f"총 처리 시간: {total_time:.2f}초"
            )
            
            return success_count == len(file_chunks)
            
        except Exception as e:
            self.logger.error(f"교체 모드 청크 처리 실패: {str(e)}")
            import traceback
            self.logger.error(f"상세 오류: {traceback.format_exc()}")
            return False
    
    async def query(self, question: str, max_sources: Optional[int] = None, score_threshold: Optional[float] = None, model_name: Optional[str] = None, retrievers: Optional[Dict[str, bool]] = None) -> RAGResponse:
        """
        질의응답 (비동기 - query_async() 호출)
        
        .. deprecated:: 1.0.0
            이 메서드는 query_async()의 래퍼입니다. 
            새로운 코드에서는 query_async()를 직접 사용하세요.
        """
        # 레거시 호환성을 위해 query_async()를 호출
        return await self.query_async(question, max_sources, score_threshold, model_name, retrievers)
    
    # ========== 비동기 메서드 (Phase 1: LLM 호출 비동기화) ==========
    
    async def query_async(self, question: str, max_sources: Optional[int] = None, score_threshold: Optional[float] = None, model_name: Optional[str] = None, retrievers: Optional[Dict[str, bool]] = None) -> RAGResponse:
        """비동기 질의응답 (LLM 호출, 검색, 리랭킹 모두 비동기 - Phase 3 완료)"""
        start_time = time.time()
        
        try:
            self.logger.info(f"비동기 질의 처리 시작: {question[:50]}...")
            
            # 기본값 적용
            max_sources = max_sources if max_sources is not None else self.rag_config.default_max_sources
            base_threshold = score_threshold if score_threshold is not None else self.rag_config.score_threshold
            
            # 동적 임계값 조정
            score_threshold = self._calculate_dynamic_threshold(
                question=question,
                base_threshold=base_threshold,
                max_sources=max_sources
            )
            
            self.logger.info(f"문서 검색 파라미터: max_sources={max_sources}, score_threshold={score_threshold:.3f} (기본값: {base_threshold:.3f})")
            
            # 모델 변경 처리
            if model_name and model_name != self.llm_client.model_name:
                self.logger.info(f"모델 변경: {self.llm_client.model_name} -> {model_name}")
                llm_config = self.config.model.get('llm')
                model_config = {
                    'name': model_name,
                    'base_url': llm_config.base_url if hasattr(llm_config, 'base_url') else 'http://localhost:11434',
                    'max_tokens': llm_config.max_tokens if hasattr(llm_config, 'max_tokens') else 1000,
                    'temperature': llm_config.temperature if hasattr(llm_config, 'temperature') else 0.1,
                    'top_p': llm_config.top_p if hasattr(llm_config, 'top_p') else 0.9
                }
                self.llm_client = OllamaLLMClient(model_config)
            
            # 일반적인 질문인지 확인
            is_general = is_general_question(question)
            self.logger.debug(f"질문 '{question}' 일반 질문 판별 결과: {is_general}")
            
            if is_general:
                # 일반 질문은 벡터 검색 없이 바로 LLM에 질문 (비동기)
                self.logger.info(f"일반 질문으로 판단: 벡터 검색 건너뛰기 (질문: '{question}')")
                llm_response = await self.llm_client.generate_answer_async(question, context="")
                answer = llm_response.text if llm_response else "답변을 생성할 수 없습니다."
                is_general_flag = llm_response.is_general if llm_response else True
                
                return RAGResponse(
                    answer=answer,
                    sources=[],
                    confidence=1.0,
                    processing_time=time.time() - start_time,
                    query=question,
                    model_used=self.llm_client.model_name,
                    is_general_answer=is_general_flag,
                    is_rag_answer=False
                )
            
            # 전문 질문이므로 검색 수행 (비동기 - Phase 2)
            self.logger.info(f"전문 질문으로 판단: 검색 수행 (질문: '{question}')")
            
            # 검색기 선택이 제공된 경우
            if retrievers is not None:
                self.logger.info(f"검색기 선택 사용: {retrievers}")
                
                selected_count = sum([
                    retrievers.get('use_qdrant', False),
                    retrievers.get('use_faiss', False),
                    retrievers.get('use_bm25', False)
                ])
                
                search_limit = max_sources if selected_count == 1 else max_sources * 2
                self.logger.debug(f"검색기 개수: {selected_count}, 검색 제한: {search_limit}")
                
                all_results = []
                
                # Qdrant 검색 (비동기)
                if retrievers.get('use_qdrant', False):
                    try:
                        # Dense/Sparse 가중치 추출
                        dense_weight = retrievers.get('dense_weight')
                        sparse_weight = retrievers.get('sparse_weight')
                        
                        qdrant_results = await self.vector_store_manager.store.search_similar_async(
                            query=question,
                            limit=search_limit,
                            score_threshold=score_threshold,
                            dense_weight=dense_weight,
                            sparse_weight=sparse_weight
                        )
                        if qdrant_results:
                            all_results.append(('qdrant', qdrant_results))
                            self.logger.debug(f"Qdrant 검색 결과: {len(qdrant_results)}개")
                    except Exception as e:
                        self.logger.warning(f"Qdrant 검색 실패: {str(e)}")
                
                # FAISS 검색 (비동기)
                if retrievers.get('use_faiss', False):
                    try:
                        langchain_manager = self.vector_store_manager.langchain_retrieval_manager
                        if langchain_manager is None:
                            self.logger.warning("LangChain Retrieval Manager가 초기화되지 않았습니다.")
                        elif langchain_manager.faiss_store is None:
                            self.logger.warning("FAISS 인덱스가 초기화되지 않았습니다.")
                        else:
                            import asyncio
                            faiss_results = await asyncio.to_thread(
                                langchain_manager.search_with_faiss_only,
                                query=question,
                                k=search_limit,
                                score_threshold=score_threshold
                            )
                            if faiss_results:
                                all_results.append(('faiss', faiss_results))
                                self.logger.info(f"FAISS 검색 성공: {len(faiss_results)}개 결과")
                    except Exception as e:
                        self.logger.error(f"FAISS 검색 실패: {str(e)}", exc_info=True)
                
                # BM25 검색 (비동기)
                if retrievers.get('use_bm25', False):
                    try:
                        langchain_manager = self.vector_store_manager.langchain_retrieval_manager
                        if langchain_manager and langchain_manager.bm25_retriever:
                            import asyncio
                            bm25_results = await asyncio.to_thread(
                                langchain_manager.search_with_bm25_only,
                                query=question,
                                k=search_limit,
                                score_threshold=score_threshold
                            )
                            if bm25_results:
                                all_results.append(('bm25', bm25_results))
                                self.logger.debug(f"BM25 검색 결과: {len(bm25_results)}개")
                        else:
                            self.logger.warning("BM25 인덱스가 초기화되지 않았습니다.")
                    except Exception as e:
                        self.logger.warning(f"BM25 검색 실패: {str(e)}")
                
                if not all_results:
                    self.logger.warning("선택된 검색기에서 결과를 찾을 수 없습니다.")
                    similar_docs = []
                elif selected_count == 1:
                    raw_results = all_results[0][1][:max_sources]
                    similar_docs = []
                    for item in raw_results:
                        if 'score' in item:
                            item['score'] = float(item['score'])
                        similar_docs.append(item)
                    self.logger.info(f"단일 검색기 사용: {all_results[0][0]}, 결과 {len(similar_docs)}개")
                else:
                    # 다중 검색기: RRF 통합 (동기 로직 재사용)
                    weights = retrievers.get('weights') or {'qdrant': 1.0, 'faiss': 0.0, 'bm25': 0.0}
                    name_to_weight = {
                        'qdrant': float(weights.get('qdrant', 0.0)),
                        'faiss': float(weights.get('faiss', 0.0)),
                        'bm25': float(weights.get('bm25', 0.0)),
                    }
                    results_list = [results for _, results in all_results]
                    retriever_names = [name for name, _ in all_results]
                    rrf_scores: Dict[str, float] = {}
                    data_map: Dict[str, Dict[str, Any]] = {}
                    K = 60
                    for idx, results in enumerate(results_list):
                        name = retriever_names[idx]
                        w = name_to_weight.get(name, 0.0)
                        if not results or w <= 0:
                            continue
                        for rank, res in enumerate(results, 1):
                            chunk_id = (
                                res.get('chunk_id') or
                                res.get('metadata', {}).get('chunk_id') or
                                res.get('id', '')
                            )
                            if not chunk_id:
                                content = res.get('content', res.get('page_content', ''))
                                import hashlib
                                chunk_id = hashlib.md5(content.encode()).hexdigest()
                            contrib = w * (1.0 / (K + rank))
                            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + contrib
                            if chunk_id not in data_map:
                                data_map[chunk_id] = res.copy()
                    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
                    merged = []
                    for cid in sorted_ids[:max_sources]:
                        item = data_map[cid].copy()
                        original_score = item.get('score', 0.0)
                        item['score'] = float(original_score)
                        item['rrf_score'] = float(rrf_scores[cid])
                        merged.append(item)
                    similar_docs = merged
                    self.logger.info(f"검색기 통합 완료: {len(similar_docs)}개 결과")
                
                # 리랭킹 적용 (동기)
                use_reranker = bool(retrievers.get('use_reranker', True)) if retrievers else True
                if similar_docs and use_reranker and self.reranker:
                    try:
                        reranker_cfg = getattr(self.config, 'reranker', {})
                        default_alpha = (getattr(reranker_cfg, 'alpha', 0.7) if not isinstance(reranker_cfg, dict) else reranker_cfg.get('alpha', 0.7))
                        default_top_k = (getattr(reranker_cfg, 'top_k', max_sources) if not isinstance(reranker_cfg, dict) else reranker_cfg.get('top_k', max_sources))
                        alpha = float(retrievers.get('reranker_alpha', default_alpha))
                        for d in similar_docs:
                            if not d.get('content') and d.get('page_content'):
                                d['content'] = d.get('page_content')
                        reranker_top_k_value = retrievers.get('reranker_top_k')
                        if reranker_top_k_value is None:
                            fallback_top_k = default_top_k if default_top_k is not None else (max_sources if max_sources is not None else 10)
                            requested_top_k = fallback_top_k
                        else:
                            requested_top_k = reranker_top_k_value
                        requested_top_k = int(requested_top_k)
                        safe_max_sources = max_sources if max_sources is not None else len(similar_docs)
                        top_k = max(1, min(requested_top_k, safe_max_sources, len(similar_docs)))
                        self.logger.info(f"리랭커 호출 시작: docs={len(similar_docs)}, top_k={top_k}, alpha={alpha}")
                        reranked_docs = await self.reranker.rerank_async(question, similar_docs, top_k=top_k)
                        if not reranked_docs:
                            self.logger.warning("리랭커가 결과를 반환하지 않았습니다.")
                            reranked_docs = similar_docs
                        similar_docs = reranked_docs
                        for d in similar_docs:
                            base_score = float(d.get('score', 0.0))
                            rr_score = float(d.get('reranker_score', 0.0))
                            d['score'] = alpha * rr_score + (1.0 - alpha) * base_score
                        similar_docs.sort(key=lambda x: x.get('score', 0.0), reverse=True)
                        similar_docs = similar_docs[:max_sources]
                    except Exception as e:
                        self.logger.warning(f"리랭킹 적용 실패: {str(e)}")
            else:
                # 기존 로직 (하이브리드 검색 또는 벡터 검색만) - 동기
                from src.utils.config import get_qdrant_config
                qdrant_config = get_qdrant_config()
                hybrid_enabled = qdrant_config.hybrid_search_enabled
                langchain_available = (
                    hasattr(self.vector_store_manager, 'langchain_retrieval_manager') and
                    self.vector_store_manager.langchain_retrieval_manager is not None and
                    self.vector_store_manager.langchain_retrieval_manager.faiss_store is not None and
                    self.vector_store_manager.langchain_retrieval_manager.bm25_retriever is not None
                )
                
                if hybrid_enabled and langchain_available:
                    self.logger.info("EnsembleRetriever 검색 사용 (FAISS + BM25) - 비동기")
                    similar_docs = await self.vector_store_manager.hybrid_search_async(
                        query=question,
                        limit=max_sources,
                        score_threshold=score_threshold
                    )
                else:
                    similar_docs = await self.vector_store_manager.search_similar_async(
                        query=question,
                        limit=max_sources,
                        score_threshold=score_threshold
                    )
                
                # 설정 기반 리랭킹 (비동기 - Phase 3)
                if similar_docs and self.reranker:
                    try:
                        reranker_cfg = getattr(self.config, 'reranker', {})
                        enabled = (getattr(reranker_cfg, 'enabled', False) if not isinstance(reranker_cfg, dict) else reranker_cfg.get('enabled', False))
                        if enabled:
                            alpha = (getattr(reranker_cfg, 'alpha', 0.7) if not isinstance(reranker_cfg, dict) else reranker_cfg.get('alpha', 0.7))
                            configured_top_k = (getattr(reranker_cfg, 'top_k', max_sources) if not isinstance(reranker_cfg, dict) else reranker_cfg.get('top_k', max_sources))
                            if configured_top_k is None:
                                fallback_top_k = max_sources if max_sources is not None else 10
                            else:
                                fallback_top_k = configured_top_k
                            safe_max_sources = max_sources if max_sources is not None else len(similar_docs)
                            top_k = max(1, min(int(fallback_top_k), safe_max_sources, len(similar_docs)))
                            self.logger.info(f"리랭커 호출 시작 (설정 기반): docs={len(similar_docs)}, top_k={top_k}, alpha={alpha}")
                            for d in similar_docs:
                                if not d.get('content') and d.get('page_content'):
                                    d['content'] = d.get('page_content')
                            reranked_docs = await self.reranker.rerank_async(question, similar_docs, top_k=top_k)
                            if not reranked_docs:
                                self.logger.warning("리랭커가 결과를 반환하지 않았습니다.")
                                reranked_docs = similar_docs
                            for d in similar_docs:
                                base_score = float(d.get('score', 0.0))
                                rr_score = float(d.get('reranker_score', 0.0))
                                d['score'] = float(alpha) * rr_score + (1.0 - float(alpha)) * base_score
                            similar_docs.sort(key=lambda x: x.get('score', 0.0), reverse=True)
                            similar_docs = similar_docs[:max_sources]
                    except Exception as e:
                        self.logger.warning(f"리랭킹 적용 실패: {str(e)}")
            
            # 검색 결과 정렬
            similar_docs.sort(key=lambda x: x.get('score', 0), reverse=True)
            
            # 검색 결과 점수가 낮으면 일반 질문으로 처리 (비동기 LLM 호출)
            if similar_docs:
                max_score = max(doc.get('score', 0) for doc in similar_docs)
                avg_score = sum(doc.get('score', 0) for doc in similar_docs) / len(similar_docs)
                low_score_threshold = self.rag_config.low_score_general_threshold
                
                if max_score < low_score_threshold and avg_score < low_score_threshold:
                    self.logger.info(f"검색 점수가 낮아 일반 질문으로 전환: 최고점수={max_score:.3f}, 평균점수={avg_score:.3f}")
                    llm_response = await self.llm_client.generate_answer_async(question, context="")
                    answer = llm_response.text if llm_response else "답변을 생성할 수 없습니다."
                    
                    return RAGResponse(
                        answer=answer,
                        sources=[],
                        confidence=0.3,
                        processing_time=time.time() - start_time,
                        query=question,
                        model_used=self.llm_client.model_name,
                        is_general_answer=True,
                        is_rag_answer=False
                    )
            elif not similar_docs:
                # 검색 결과가 없으면 일반 질문으로 처리 (비동기 LLM 호출)
                self.logger.info("검색 결과가 없어 일반 질문으로 처리")
                llm_response = await self.llm_client.generate_answer_async(question, context="")
                answer = llm_response.text if llm_response else "답변을 생성할 수 없습니다."
                
                return RAGResponse(
                    answer=answer,
                    sources=[],
                    confidence=0.2,
                    processing_time=time.time() - start_time,
                    query=question,
                    model_used=self.llm_client.model_name,
                    is_general_answer=True,
                    is_rag_answer=False
                )
            
            # 중복 청크 제거
            unique_docs = []
            seen_chunks = set()
            for doc in similar_docs:
                chunk_key = f"{doc.get('source_file', '')}:{doc.get('chunk_index', '')}"
                if chunk_key not in seen_chunks:
                    seen_chunks.add(chunk_key)
                    unique_docs.append(doc)
            similar_docs = unique_docs
            
            # 표 데이터 중복 제거
            if len(similar_docs) > 1:
                table_docs = [doc for doc in similar_docs if '표 데이터' in doc.get('content', '')]
                if len(table_docs) > 1:
                    table_docs.sort(key=lambda x: x.get('score', 0), reverse=True)
                    for table_doc in table_docs[1:]:
                        if table_doc in similar_docs:
                            similar_docs.remove(table_doc)
            
            if not similar_docs:
                return RAGResponse(
                    answer="관련 문서를 찾을 수 없습니다.",
                    sources=[],
                    confidence=0.0,
                    processing_time=time.time() - start_time,
                    query=question,
                    model_used=""
                )
            
            # 컨텍스트 구성 (토큰 제한 자동 조정)
            context = self._build_context(similar_docs, max_tokens=None)
            
            # LLM을 통한 답변 생성 (비동기)
            llm_response = await self.llm_client.generate_answer_async(question, context)
            answer = llm_response.text if llm_response else "답변을 생성할 수 없습니다."
            is_general = llm_response.is_general if llm_response else False
            has_rag_context = llm_response.has_rag_context if llm_response else True
            
            # 신뢰도 계산
            confidence = self._calculate_confidence(similar_docs, answer)
            
            # 소스 정보 정리
            sources = self._format_sources(similar_docs)
            
            processing_time = time.time() - start_time
            
            self.logger.info(f"비동기 질의 처리 완료: {processing_time:.2f}초, 신뢰도: {confidence:.2f}, 일반답변={is_general}, RAG답변={has_rag_context}")
            
            return RAGResponse(
                answer=answer,
                sources=sources,
                confidence=confidence,
                processing_time=processing_time,
                query=question,
                model_used=self.llm_client.model_name,
                is_general_answer=is_general,
                is_rag_answer=has_rag_context
            )
            
        except Exception as e:
            self.logger.error(f"비동기 질의 처리 실패: {str(e)}")
            return RAGResponse(
                answer="죄송합니다. 처리 중 오류가 발생했습니다.",
                sources=[],
                confidence=0.0,
                processing_time=time.time() - start_time,
                query=question,
                model_used=""
            )
    
    def _calculate_dynamic_threshold(self, question: str, base_threshold: float, max_sources: int) -> float:
        """
        동적 임계값 계산
        
        질문 유형과 요청된 문서 수에 따라 임계값을 조정합니다.
        
        Args:
            question: 사용자 질문
            base_threshold: 기본 임계값
            max_sources: 요청된 최대 소스 수
            
        Returns:
            조정된 임계값
        """
        threshold = base_threshold
        
        # 1. 질문 길이에 따른 조정 (짧은 질문은 더 엄격하게)
        question_length = len(question.strip())
        if question_length < 10:
            # 매우 짧은 질문: 임계값 증가 (더 관련성 높은 결과만)
            threshold += 0.1
        elif question_length > 50:
            # 긴 질문: 임계값 감소 (더 많은 결과 포함)
            threshold -= 0.05
        
        # 2. 요청된 문서 수에 따른 조정
        if max_sources <= 3:
            # 적은 수의 문서 요청: 임계값 증가 (고품질만)
            threshold += 0.1
        elif max_sources >= 10:
            # 많은 수의 문서 요청: 임계값 감소 (더 넓은 범위)
            threshold -= 0.1
        
        # 3. 질문 유형에 따른 조정
        question_lower = question.lower()
        
        # 키워드 기반 질문 (예: "변압기 진단 기준")
        if any(keyword in question_lower for keyword in ['기준', '방법', '절차', '과정', '원리']):
            # 구체적인 정보 요청: 임계값 약간 감소
            threshold -= 0.05
        
        # 비교/분석 질문 (예: "차이점", "비교")
        if any(keyword in question_lower for keyword in ['차이', '비교', '분석', '대비']):
            # 여러 문서 비교 필요: 임계값 감소
            threshold -= 0.1
        
        # 표/데이터 질문
        if any(keyword in question_lower for keyword in ['표', 'table', '데이터', '수치']):
            # 표 데이터는 정확한 매칭 필요: 임계값 유지 또는 약간 증가
            threshold += 0.05
        
        # 임계값 범위 제한 (0.0 ~ 1.0)
        threshold = max(0.0, min(1.0, threshold))
        
        return threshold
    
    def _estimate_tokens(self, text: str) -> int:
        """
        텍스트의 토큰 수 추정 (한국어 기준)
        
        한국어의 경우 일반적으로 1토큰 ≈ 2-3자 정도입니다.
        보수적으로 1토큰 = 2.5자로 계산합니다.
        
        Args:
            text: 토큰 수를 추정할 텍스트
            
        Returns:
            추정된 토큰 수
        """
        # 한국어 기준: 1토큰 ≈ 2.5자
        # 영어 기준: 1토큰 ≈ 4자
        # 혼합 텍스트를 고려하여 평균값 사용
        return int(len(text) / 2.5)
    
    def _build_context(self, similar_docs: List[Dict[str, Any]], max_tokens: Optional[int] = None) -> str:
        """
        컨텍스트 구성 (토큰 제한 자동 조정)
        
        Args:
            similar_docs: 검색된 문서 리스트
            max_tokens: 최대 토큰 수 (None이면 제한 없음)
            
        Returns:
            구성된 컨텍스트 문자열
        """
        context_parts = []
        total_tokens = 0
        
        # LLM의 max_tokens 가져오기 (기본값: 1000)
        if max_tokens is None:
            llm_max_tokens = getattr(self.llm_client, 'max_tokens', 1000)
            # 컨텍스트는 max_tokens의 70% 정도 사용 (나머지는 답변 생성용)
            max_tokens = int(llm_max_tokens * 0.7)
        
        for i, doc in enumerate(similar_docs, 1):
            # 파일명만 추출 (경로에서 파일명만)
            filename = doc['source_file'].split('\\')[-1] if '\\' in doc['source_file'] else doc['source_file']
            filename = filename.split('/')[-1] if '/' in filename else filename
            
            # 관련도 점수 포함
            relevance_score = doc.get('score', 0)
            source_info = f"[문서 {i}] 출처: {filename} (관련도: {relevance_score:.3f})"
            
            # 청크 인덱스 정보 추가
            chunk_info = f"청크 인덱스: {doc.get('chunk_index', 'N/A')}"
            
            content = doc['content']
            
            # 컨텍스트 부분 구성
            context_part = f"{source_info}\n{chunk_info}\n{content}"
            part_tokens = self._estimate_tokens(context_part)
            
            # 토큰 제한 확인
            if max_tokens and total_tokens + part_tokens > max_tokens:
                # 토큰 제한 초과 시 현재 문서의 내용을 자름
                remaining_tokens = max_tokens - total_tokens - self._estimate_tokens(f"{source_info}\n{chunk_info}\n")
                if remaining_tokens > 0:
                    # 남은 토큰 수에 맞춰 내용 자르기
                    max_chars = int(remaining_tokens * 2.5)  # 토큰 → 문자 변환
                    truncated_content = content[:max_chars] + "..."
                    context_part = f"{source_info}\n{chunk_info}\n{truncated_content}"
                    context_parts.append(context_part)
                    self.logger.warning(
                        f"토큰 제한으로 인해 문서 {i}의 내용이 잘렸습니다. "
                        f"(총 토큰: {total_tokens + self._estimate_tokens(context_part)}/{max_tokens})"
                    )
                break
            
            context_parts.append(context_part)
            total_tokens += part_tokens
        
        if max_tokens and total_tokens > 0:
            self.logger.debug(f"컨텍스트 구성 완료: {len(context_parts)}개 문서, {total_tokens}토큰 (제한: {max_tokens}토큰)")
        
        return "\n\n".join(context_parts)
    
    def _calculate_confidence(self, similar_docs: List[Dict[str, Any]], answer: str) -> float:
        """신뢰도 계산"""
        if not similar_docs:
            return 0.0
        
        # 평균 유사도 점수를 기반으로 신뢰도 계산
        avg_score = sum(doc['score'] for doc in similar_docs) / len(similar_docs)
        
        # 점수를 0-1 범위로 정규화
        confidence = min(avg_score, 1.0)
        
        return confidence
    
    def _format_sources(self, similar_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """소스 정보 포맷팅"""
        sources = []
        preview_length = self.rag_config.content_preview_length
        
        for doc in similar_docs:
            content = doc['content']
            preview = content[:preview_length] + "..." if len(content) > preview_length else content
            
            # 메타데이터에서 계층 정보 추출
            metadata = doc.get('metadata', {})
            source_parts = []
            
            # 파일명만 추출 (경로에서 파일명만)
            source_file = doc.get('source_file', '')
            filename = source_file.split('\\')[-1] if '\\' in source_file else source_file
            filename = filename.split('/')[-1] if '/' in filename else filename
            source_parts.append(filename)
            
            # heading, sub-heading, sub-sub-heading 순서로 추가
            if metadata.get('heading'):
                source_parts.append(metadata.get('heading'))
            if metadata.get('sub-heading'):
                source_parts.append(metadata.get('sub-heading'))
            if metadata.get('sub-sub-heading'):
                source_parts.append(metadata.get('sub-sub-heading'))

            # 표 데이터인 경우 표 제목을 출처 경로 마지막에 추가
            is_table_data = bool(metadata.get('is_table_data')) or ('표 데이터' in content)
            if is_table_data:
                table_title = metadata.get('table_title')
                if not table_title:
                    # 컨텐츠에서 표 제목 추출 시도 (예: "표 5-18, ...")
                    import re
                    m = re.search(r'(표\s*\d+[\-.]?\d*[,\.:]?\s*[^\n]+)', content)
                    if m:
                        table_title = m.group(1).strip()
                if table_title:
                    source_parts.append(table_title)
            
            # 출처 경로 생성 ("> "로 구분)
            source_path = " > ".join(source_parts) if source_parts else filename
            
            source = {
                'content': preview,
                'source_file': doc['source_file'],
                'source_path': source_path,  # 계층 형식 출처 경로 추가
                'relevance_score': doc['score'],
                'chunk_index': doc['chunk_index'],
                'metadata': metadata  # 메타데이터 전체도 포함
            }
            sources.append(source)
        
        return sources
    
    def get_system_stats(self) -> Dict[str, Any]:
        """시스템 통계 반환"""
        try:
            # 벡터 저장소 통계
            vector_stats = self.vector_store_manager.get_collection_info()
            
            # 임베딩 캐시 통계
            embedding_stats = {
                'cache_size': len(self.embedding_manager.cache) if hasattr(self.embedding_manager, 'cache') else 0,
                'model_name': self.config.model.get('embedding', {}).name if 'embedding' in self.config.model else 'unknown',
                'dimension': self.config.model.get('embedding', {}).dimension if 'embedding' in self.config.model else 1024
            }
            
            return {
                'vector_store_stats': vector_stats,
                'embedding_cache_stats': embedding_stats,
                'llm_model': self.config.model.get('llm', {}).name if 'llm' in self.config.model else 'unknown'
            }
            
        except Exception as e:
            self.logger.error(f"시스템 통계 조회 실패: {str(e)}")
            return {
                'vector_store_stats': {},
                'embedding_cache_stats': {'cache_size': 0, 'model_name': 'unknown', 'dimension': 1024},
                'llm_model': 'unknown'
            }
    
    def get_documents_info(self) -> List[Dict[str, Any]]:
        """저장된 문서들의 정보 반환"""
        try:
            return self.vector_store_manager.get_documents_info()
        except Exception as e:
            self.logger.error(f"문서 정보 조회 실패: {str(e)}")
            return []
    
    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """특정 문서의 청크 정보 반환"""
        try:
            return self.vector_store_manager.get_document_chunks(document_id)
        except Exception as e:
            self.logger.error(f"문서 청크 조회 실패: {str(e)}")
            return []
    
    def rebuild_faiss_and_bm25_indexes(self) -> bool:
        """
        Qdrant에서 모든 문서를 읽어서 FAISS 및 BM25 인덱스 재생성
        (문서 재업로드 없이 인덱스만 재구축)
        
        Returns:
            재구축 성공 여부
        """
        try:
            self.logger.info("FAISS 및 BM25 인덱스 재구축 시작...")
            
            # 1. Qdrant에서 모든 문서 정보 가져오기
            documents_info = self.vector_store_manager.get_documents_info()
            if not documents_info:
                self.logger.warning("Qdrant에 문서가 없습니다. 문서를 먼저 업로드해주세요.")
                return False
            
            self.logger.info(f"총 {len(documents_info)}개 문서 발견")
            
            # 2. 모든 문서의 청크 가져오기
            all_chunks_data = []
            for doc_info in documents_info:
                source_file = doc_info.get('source_file', '')
                if source_file:
                    chunks_data = self.vector_store_manager.get_document_chunks(source_file)
                    all_chunks_data.extend(chunks_data)
            
            if not all_chunks_data:
                self.logger.warning("청크 데이터를 가져올 수 없습니다.")
                return False
            
            self.logger.info(f"총 {len(all_chunks_data)}개 청크 발견")
            
            # 3. 딕셔너리 형식을 DocumentChunk로 변환
            from src.modules.document_processor import DocumentChunk
            chunks = []
            for chunk_data in all_chunks_data:
                # content_full이 있으면 사용, 없으면 content_preview 사용
                content = chunk_data.get('content_full', '')
                if not content:
                    content = chunk_data.get('content_preview', '')
                
                # metadata에서 source_file 추출 (여러 가능한 위치 확인)
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
            
            # 4. FAISS 인덱스 생성
            if hasattr(self.vector_store_manager, 'langchain_retrieval_manager') and \
               self.vector_store_manager.langchain_retrieval_manager:
                self.logger.info("FAISS 인덱스 생성 중...")
                faiss_success = self.vector_store_manager.langchain_retrieval_manager.initialize_faiss_from_chunks(chunks)
                if faiss_success:
                    self.logger.info("FAISS 인덱스 생성 완료")
                else:
                    self.logger.error("FAISS 인덱스 생성 실패")
                    return False
            else:
                self.logger.error("LangChain Retrieval Manager가 초기화되지 않았습니다.")
                return False
            
            # 5. BM25 인덱스 구축
            self.logger.info("BM25 인덱스 구축 중...")
            bm25_success = self.vector_store_manager.build_bm25_index(chunks)
            if bm25_success:
                self.logger.info("BM25 인덱스 구축 완료")
            else:
                self.logger.error("BM25 인덱스 구축 실패")
                return False
            
            # 6. EnsembleRetriever 초기화
            if hasattr(self.vector_store_manager, 'langchain_retrieval_manager') and \
               self.vector_store_manager.langchain_retrieval_manager:
                try:
                    from src.utils.config import get_qdrant_config
                    qdrant_config = get_qdrant_config()
                    faiss_weight = qdrant_config.hybrid_search_vector_weight if hasattr(qdrant_config, 'hybrid_search_vector_weight') else 0.7
                    bm25_weight = qdrant_config.hybrid_search_bm25_weight if hasattr(qdrant_config, 'hybrid_search_bm25_weight') else 0.3
                    rrf_c = qdrant_config.hybrid_search_rrf_k if hasattr(qdrant_config, 'hybrid_search_rrf_k') else 60
                    
                    self.vector_store_manager.langchain_retrieval_manager.create_ensemble_retriever(
                        faiss_weight=faiss_weight,
                        bm25_weight=bm25_weight,
                        c=rrf_c,
                        k=self.rag_config.default_max_sources
                    )
                    self.logger.info("EnsembleRetriever 초기화 완료")
                except Exception as e:
                    self.logger.warning(f"EnsembleRetriever 초기화 실패: {str(e)}")
            
            self.logger.info(f"인덱스 재구축 완료: {len(chunks)}개 청크")
            return True
            
        except Exception as e:
            self.logger.error(f"인덱스 재구축 실패: {str(e)}")
            import traceback
            self.logger.error(f"상세 오류: {traceback.format_exc()}")
            return False
    
    async def rebuild_faiss_and_bm25_indexes_async(self) -> bool:
        """
        Qdrant에서 모든 문서를 읽어서 FAISS 및 BM25 인덱스 재생성 (비동기)
        (문서 재업로드 없이 인덱스만 재구축)
        
        Returns:
            재구축 성공 여부
        """
        import asyncio
        
        try:
            self.logger.info("비동기 FAISS 및 BM25 인덱스 재구축 시작...")
            
            # 1. Qdrant에서 모든 문서 정보 가져오기 (I/O 작업 - 비동기화)
            documents_info = await asyncio.to_thread(
                self.vector_store_manager.get_documents_info
            )
            if not documents_info:
                self.logger.warning("Qdrant에 문서가 없습니다. 문서를 먼저 업로드해주세요.")
                return False
            
            self.logger.info(f"총 {len(documents_info)}개 문서 발견")
            
            # 2. 모든 문서의 청크 가져오기 (I/O 작업 - 비동기화)
            all_chunks_data = []
            for doc_info in documents_info:
                source_file = doc_info.get('source_file', '')
                if source_file:
                    chunks_data = await asyncio.to_thread(
                        self.vector_store_manager.get_document_chunks,
                        source_file
                    )
                    all_chunks_data.extend(chunks_data)
            
            if not all_chunks_data:
                self.logger.warning("청크 데이터를 가져올 수 없습니다.")
                return False
            
            self.logger.info(f"총 {len(all_chunks_data)}개 청크 발견")
            
            # 3. 딕셔너리 형식을 DocumentChunk로 변환 (CPU 작업 - 동기)
            from src.modules.document_processor import DocumentChunk
            chunks = []
            for chunk_data in all_chunks_data:
                # content_full이 있으면 사용, 없으면 content_preview 사용
                content = chunk_data.get('content_full', '')
                if not content:
                    content = chunk_data.get('content_preview', '')
                
                # metadata에서 source_file 추출 (여러 가능한 위치 확인)
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
            
            # 4. FAISS 인덱스 생성 (I/O 작업 - 비동기화)
            if hasattr(self.vector_store_manager, 'langchain_retrieval_manager') and \
               self.vector_store_manager.langchain_retrieval_manager:
                self.logger.info("FAISS 인덱스 생성 중...")
                langchain_manager = self.vector_store_manager.langchain_retrieval_manager
                faiss_success = await asyncio.to_thread(
                    langchain_manager.initialize_faiss_from_chunks,
                    chunks
                )
                if faiss_success:
                    self.logger.info("FAISS 인덱스 생성 완료")
                else:
                    self.logger.error("FAISS 인덱스 생성 실패")
                    return False
            else:
                self.logger.error("LangChain Retrieval Manager가 초기화되지 않았습니다.")
                return False
            
            # 5. BM25 인덱스 구축 (I/O 작업 - 비동기화)
            self.logger.info("BM25 인덱스 구축 중...")
            bm25_success = await asyncio.to_thread(
                self.vector_store_manager.build_bm25_index,
                chunks
            )
            if bm25_success:
                self.logger.info("BM25 인덱스 구축 완료")
            else:
                self.logger.error("BM25 인덱스 구축 실패")
                return False
            
            # 6. EnsembleRetriever 초기화 (I/O 작업 - 비동기화)
            if hasattr(self.vector_store_manager, 'langchain_retrieval_manager') and \
               self.vector_store_manager.langchain_retrieval_manager:
                try:
                    from src.utils.config import get_qdrant_config
                    qdrant_config = get_qdrant_config()
                    faiss_weight = qdrant_config.hybrid_search_vector_weight if hasattr(qdrant_config, 'hybrid_search_vector_weight') else 0.7
                    bm25_weight = qdrant_config.hybrid_search_bm25_weight if hasattr(qdrant_config, 'hybrid_search_bm25_weight') else 0.3
                    rrf_c = qdrant_config.hybrid_search_rrf_k if hasattr(qdrant_config, 'hybrid_search_rrf_k') else 60
                    
                    await asyncio.to_thread(
                        self.vector_store_manager.langchain_retrieval_manager.create_ensemble_retriever,
                        faiss_weight=faiss_weight,
                        bm25_weight=bm25_weight,
                        c=rrf_c,
                        k=self.rag_config.default_max_sources
                    )
                    self.logger.info("EnsembleRetriever 초기화 완료")
                except Exception as e:
                    self.logger.warning(f"EnsembleRetriever 초기화 실패: {str(e)}")
            
            self.logger.info(f"비동기 인덱스 재구축 완료: {len(chunks)}개 청크")
            return True
            
        except Exception as e:
            self.logger.error(f"비동기 인덱스 재구축 실패: {str(e)}")
            import traceback
            self.logger.error(f"상세 오류: {traceback.format_exc()}")
            return False
    
    def clear_cache(self):
        """캐시 초기화"""
        self.embedding_manager.clear_cache()
        self.logger.info("RAG 시스템 캐시가 초기화되었습니다")
    
    def query_by_table_title(self, 
                           table_title: str, 
                           question: str = "", 
                           max_sources: Optional[int] = None, 
                           score_threshold: Optional[float] = None, 
                           model_name: Optional[str] = None) -> RAGResponse:
        """표 제목으로 검색하여 질의응답"""
        start_time = time.time()
        
        try:
            self.logger.info(f"표 제목 검색 시작: {table_title}")
            
            # 기본값 적용 (통일된 임계값 사용)
            max_sources = max_sources if max_sources is not None else self.rag_config.default_max_sources_table
            score_threshold = score_threshold if score_threshold is not None else self.rag_config.score_threshold
            
            self.logger.info(f"표 제목 검색 파라미터: max_sources={max_sources}, score_threshold={score_threshold:.3f}")
            
            # 모델 변경 처리
            if model_name and model_name != self.llm_client.model_name:
                self.logger.info(f"모델 변경: {self.llm_client.model_name} -> {model_name}")
                # 설정에서 LLM 설정 가져오기
                llm_config = self.config.model.get('llm')
                model_config = {
                    'name': model_name,
                    'base_url': llm_config.base_url if hasattr(llm_config, 'base_url') else 'http://localhost:11434',
                    'max_tokens': llm_config.max_tokens if hasattr(llm_config, 'max_tokens') else 1000,
                    'temperature': llm_config.temperature if hasattr(llm_config, 'temperature') else 0.1,
                    'top_p': llm_config.top_p if hasattr(llm_config, 'top_p') else 0.9
                }
                self.llm_client = OllamaLLMClient(model_config)
            
            # 1. 표 제목으로 검색
            similar_docs = self.vector_store_manager.search_by_table_title(
                table_title=table_title,
                limit=max_sources,
                score_threshold=score_threshold
            )
            
            if not similar_docs:
                return RAGResponse(
                    answer=f"'{table_title}' 표를 찾을 수 없습니다.",
                    sources=[],
                    confidence=0.0,
                    processing_time=time.time() - start_time,
                    query=f"표 제목: {table_title}",
                    model_used=self.llm_client.model_name
                )
            
            # 2. 컨텍스트 구성 (토큰 제한 자동 조정)
            context = self._build_context(similar_docs, max_tokens=None)
            
            # 3. 질문이 있으면 답변 생성, 없으면 표 내용 요약
            if question.strip():
                llm_response = self.llm_client.generate_answer_with_metadata(question, context)
            else:
                llm_response = self.llm_client.generate_answer_with_metadata(
                    f"'{table_title}' 표의 내용을 요약해주세요.", 
                    context
                )
            
            answer = llm_response.text if llm_response else "답변을 생성할 수 없습니다."
            is_general = llm_response.is_general if llm_response else False
            has_rag_context = llm_response.has_rag_context if llm_response else True
            
            # 4. 신뢰도 계산
            confidence = self._calculate_confidence(similar_docs, answer)
            
            # 5. 소스 정보 정리
            sources = self._format_sources(similar_docs)
            
            processing_time = time.time() - start_time
            
            self.logger.info(f"표 제목 검색 완료: {processing_time:.2f}초, 신뢰도: {confidence:.2f}, 일반답변={is_general}, RAG답변={has_rag_context}")
            
            return RAGResponse(
                answer=answer,
                sources=sources,
                confidence=confidence,
                processing_time=processing_time,
                query=f"표 제목: {table_title}",
                model_used=self.llm_client.model_name,
                is_general_answer=is_general,
                is_rag_answer=has_rag_context
            )
            
        except Exception as e:
            self.logger.error(f"표 제목 검색 중 오류: {str(e)}")
            return RAGResponse(
                answer=f"표 제목 검색 중 오류가 발생했습니다: {str(e)}",
                sources=[],
                confidence=0.0,
                processing_time=time.time() - start_time,
                query=f"표 제목: {table_title}",
                model_used=""
            )
    
    def query_with_table_filter(self, 
                               question: str, 
                               table_title: Optional[str] = None,
                               is_table_data: Optional[bool] = None,
                               max_sources: Optional[int] = None, 
                               score_threshold: Optional[float] = None, 
                               model_name: Optional[str] = None) -> RAGResponse:
        """표 관련 필터와 함께 질의응답"""
        start_time = time.time()
        
        try:
            self.logger.info(f"필터 검색 시작: {question[:50]}...")
            
            # 기본값 적용 (통일된 임계값 사용)
            max_sources = max_sources if max_sources is not None else self.rag_config.default_max_sources_table
            score_threshold = score_threshold if score_threshold is not None else self.rag_config.score_threshold
            
            self.logger.info(f"필터 검색 파라미터: max_sources={max_sources}, score_threshold={score_threshold:.3f}")
            
            # 모델 변경 처리
            if model_name and model_name != self.llm_client.model_name:
                self.logger.info(f"모델 변경: {self.llm_client.model_name} -> {model_name}")
                # 설정에서 LLM 설정 가져오기
                llm_config = self.config.model.get('llm')
                model_config = {
                    'name': model_name,
                    'base_url': llm_config.base_url if hasattr(llm_config, 'base_url') else 'http://localhost:11434',
                    'max_tokens': llm_config.max_tokens if hasattr(llm_config, 'max_tokens') else 1000,
                    'temperature': llm_config.temperature if hasattr(llm_config, 'temperature') else 0.1,
                    'top_p': llm_config.top_p if hasattr(llm_config, 'top_p') else 0.9
                }
                self.llm_client = OllamaLLMClient(model_config)
            
            # 1. 필터와 함께 검색
            similar_docs = self.vector_store_manager.search_with_table_filter(
                query=question,
                table_title=table_title,
                is_table_data=is_table_data,
                limit=max_sources,
                score_threshold=score_threshold
            )
            
            if not similar_docs:
                filter_info = []
                if table_title:
                    filter_info.append(f"표 제목: {table_title}")
                if is_table_data is not None:
                    filter_info.append(f"표 데이터: {is_table_data}")
                
                filter_str = ", ".join(filter_info) if filter_info else "필터 없음"
                return RAGResponse(
                    answer=f"조건에 맞는 문서를 찾을 수 없습니다. (필터: {filter_str})",
                    sources=[],
                    confidence=0.0,
                    processing_time=time.time() - start_time,
                    query=question,
                    model_used=self.llm_client.model_name
                )
            
            # 2. 컨텍스트 구성 (토큰 제한 자동 조정)
            context = self._build_context(similar_docs, max_tokens=None)
            
            # 3. LLM을 통한 답변 생성 (메타데이터 포함)
            llm_response = self.llm_client.generate_answer_with_metadata(question, context)
            answer = llm_response.text if llm_response else "답변을 생성할 수 없습니다."
            is_general = llm_response.is_general if llm_response else False
            has_rag_context = llm_response.has_rag_context if llm_response else True
            
            # 4. 신뢰도 계산
            confidence = self._calculate_confidence(similar_docs, answer)
            
            # 5. 소스 정보 정리
            sources = self._format_sources(similar_docs)
            
            processing_time = time.time() - start_time
            
            self.logger.info(f"필터 검색 완료: {processing_time:.2f}초, 신뢰도: {confidence:.2f}, 일반답변={is_general}, RAG답변={has_rag_context}")
            
            return RAGResponse(
                answer=answer,
                sources=sources,
                confidence=confidence,
                processing_time=processing_time,
                query=question,
                model_used=self.llm_client.model_name,
                is_general_answer=is_general,
                is_rag_answer=has_rag_context
            )
            
        except Exception as e:
            self.logger.error(f"필터 검색 중 오류: {str(e)}")
            return RAGResponse(
                answer=f"필터 검색 중 오류가 발생했습니다: {str(e)}",
                sources=[],
                confidence=0.0,
                processing_time=time.time() - start_time,
                query=question,
                model_used=""
            )
    
    def _merge_search_results_with_rrf(
        self,
        results_list: List[List[Dict[str, Any]]],
        k: int = 60,
        rrf_k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        여러 검색기 결과를 RRF (Reciprocal Rank Fusion)로 통합
        
        Args:
            results_list: 검색기별 결과 리스트들의 리스트
            k: 반환할 최종 결과 수
            rrf_k: RRF 알고리즘 상수
            
        Returns:
            통합된 검색 결과 리스트
        """
        if not results_list:
            return []
        
        # RRF 점수 계산: RRF 점수 = Σ 1 / (k + rank)
        rrf_scores: Dict[str, float] = {}
        chunk_data_map: Dict[str, Dict[str, Any]] = {}
        
        # 각 검색기 결과에 대해 RRF 점수 계산
        for result_list in results_list:
            if not result_list:
                continue
            
            for rank, result in enumerate(result_list, 1):
                # 청크 ID 추출 (여러 가능한 위치 확인)
                chunk_id = (
                    result.get('chunk_id') or
                    result.get('metadata', {}).get('chunk_id') or
                    result.get('id', '')
                )
                
                if not chunk_id:
                    # chunk_id가 없으면 content 해시로 대체
                    content = result.get('content', result.get('page_content', ''))
                    import hashlib
                    chunk_id = hashlib.md5(content.encode()).hexdigest()
                
                # RRF 점수 누적
                rrf_score = 1.0 / (rrf_k + rank)
                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + rrf_score
                
                # 청크 데이터 저장 (가장 높은 점수의 결과 사용)
                if chunk_id not in chunk_data_map:
                    chunk_data_map[chunk_id] = result.copy()
                elif rrf_score > rrf_scores.get(chunk_id, 0.0):
                    # 같은 청크의 다른 결과가 더 높은 점수면 업데이트
                    chunk_data_map[chunk_id] = result.copy()
        
        # RRF 점수 기준으로 정렬
        sorted_chunk_ids = sorted(
            rrf_scores.keys(),
            key=lambda x: rrf_scores[x],
            reverse=True
        )
        
        # 최종 결과 생성
        merged_results = []
        for chunk_id in sorted_chunk_ids[:k]:
            result = chunk_data_map[chunk_id].copy()
            # RRF 점수를 최종 점수로 사용 (0~1 범위로 정규화)
            max_rrf_score = max(rrf_scores.values()) if rrf_scores else 1.0
            result['score'] = rrf_scores[chunk_id] / max_rrf_score if max_rrf_score > 0 else rrf_scores[chunk_id]
            result['rrf_score'] = rrf_scores[chunk_id]
            merged_results.append(result)
        
        return merged_results


def create_rag_system(config: Optional[Dict[str, Any]] = None) -> RAGSystem:
    """RAG 시스템 생성"""
    return RAGSystem(config)


def setup_rag_system(input_dir: str, config: Optional[Dict[str, Any]] = None) -> bool:
    """RAG 시스템 설정 및 문서 처리"""
    logger = get_logger()
    
    try:
        logger.info(f"RAG 시스템 설정 시작: {input_dir}")
        
        # RAG 시스템 생성
        rag_system = create_rag_system(config)
        logger.info("RAG 시스템 생성 완료")
        
        # 벡터 저장소 설정 (컬렉션이 없으면 새로 생성)
        logger.info("벡터 저장소 컬렉션 설정 중...")
        if not rag_system.vector_store_manager.setup_collection(force_recreate=False):
            logger.error("컬렉션 설정 실패")
            return False
        logger.info("벡터 저장소 컬렉션 설정 완료")
        
        # 문서 처리 및 저장
        logger.info(f"문서 처리 시작: {input_dir}")
        success = rag_system.process_and_store_documents(input_dir)
        
        if success:
            logger.info("RAG 시스템 설정 완료")
        else:
            logger.error("문서 처리 실패")
            
        return success
        
    except Exception as e:
        logger.error(f"RAG 시스템 설정 중 오류: {str(e)}")
        return False
