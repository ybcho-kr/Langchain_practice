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
from src.modules.vector_store import QdrantVectorStore
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
        # QdrantVectorStore 직접 사용
        self.vector_store = QdrantVectorStore(config.qdrant, bge_model=None)
        
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
        
        # Qdrant 벡터 저장소 상태 확인
        qdrant_stats = self.vector_store.get_collection_info()
        if qdrant_stats:
            points_count = qdrant_stats.get('points_count', 0)
            self.logger.info(f"Qdrant 컬렉션 준비 완료 (포인트 {points_count}개)")
        
        self.logger.info("RAG 시스템이 초기화되었습니다")
    
    def _release_gpu_memory(self):
        """GPU 메모리 해제 (모델 언로드)"""
        import torch
        import gc
        
        self.logger.info("GPU 메모리 해제 시작...")
        
        # BGE-m3 모델 해제
        if hasattr(self, 'vector_store') and self.vector_store:
            if hasattr(self.vector_store, 'bge_model') and self.vector_store.bge_model:
                try:
                    del self.vector_store.bge_model
                    self.logger.info("BGE-m3 모델 메모리 해제 완료")
                except Exception as e:
                    self.logger.warning(f"BGE-m3 모델 메모리 해제 실패: {str(e)}")
        
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
        """BGE-m3 모델 동적 재로드"""
        self.logger.info("BGE-m3 모델 재로드 시작...")
        
        # 기존 모델 해제
        if hasattr(self, 'vector_store') and self.vector_store:
            if hasattr(self.vector_store, 'bge_model') and self.vector_store.bge_model:
                del self.vector_store.bge_model
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # 새 모델 로드
        try:
            from FlagEmbedding import BGEM3FlagModel
            from src.utils.config import get_embedding_config
            
            if config is None:
                embedding_config = get_embedding_config()
            else:
                embedding_config = config
            
            model_path = embedding_config.model_path or embedding_config.name
            if not model_path:
                raise ValueError("BGE-m3 모델 경로가 설정되지 않았습니다.")
            
            # GPU 사용 가능 여부 확인
            use_fp16 = True
            try:
                import torch
                if not torch.cuda.is_available():
                    use_fp16 = False
            except ImportError:
                use_fp16 = False
            
            # 새 BGE-m3 모델 로드
            new_bge_model = BGEM3FlagModel(model_path, use_fp16=use_fp16)
            
            # QdrantVectorStore의 BGE-m3 모델 업데이트
            self.vector_store.bge_model = new_bge_model
            
            self.logger.info("BGE-m3 모델 재로드 완료")
            return True
        except Exception as e:
            self.logger.error(f"BGE-m3 모델 재로드 실패: {str(e)}")
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
        특정 문서를 Qdrant에서 삭제 (FAISS/BM25 제거됨)
        
        Args:
            source_file: 삭제할 문서의 소스 파일 경로
            
        Returns:
            삭제 결과 딕셔너리
        """
        try:
            self.logger.info(f"문서 삭제 시작: {source_file}")
            
            # Qdrant에서 삭제 (삭제된 청크 수 반환)
            qdrant_success, deleted_chunks_count = self.vector_store._delete_document_vectors(source_file)
            
            if qdrant_success:
                self.logger.info(f"문서 삭제 완료: {source_file}, {deleted_chunks_count}개 청크 삭제됨")
            else:
                self.logger.error(f"문서 삭제 실패: {source_file}")
            
            return {
                'success': qdrant_success,
                'qdrant_success': qdrant_success,
                'qdrant_deleted': qdrant_success,
                'deleted_chunks_count': deleted_chunks_count
            }
            
        except Exception as e:
            self.logger.error(f"문서 삭제 중 예외 발생: {source_file}, 오류: {str(e)}")
            import traceback
            self.logger.error(f"상세 오류: {traceback.format_exc()}")
            return {
                'success': False,
                'qdrant_success': False,
                'qdrant_deleted': False,
                'deleted_chunks_count': 0,
                'warnings': [f"삭제 중 예외 발생: {str(e)}"]
            }
    
    def process_and_store_documents(self, input_dir: str, force_update: bool = False, replace_mode: bool = False) -> bool:
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
                # 일반 모드: Qdrant에 저장 (sparse_enabled이면 자동으로 dense+sparse 벡터 함께 저장)
                success = self.vector_store.add_documents(chunks, force_update)
            
            if not success:
                self.logger.error("벡터 저장소 저장 실패")
                return False
            
            # FAISS/BM25 인덱스는 더 이상 사용하지 않음 (Qdrant Dense+Sparse만 사용)
            
            self.logger.info("문서 처리 및 저장 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"문서 처리 및 저장 실패: {str(e)}")
            return False
    
    async def process_and_store_documents_async(self, input_dir: str, force_update: bool = False, replace_mode: bool = False) -> bool:
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
                # 일반 모드: Qdrant에 저장 (sparse_enabled이면 자동으로 dense+sparse 벡터 함께 저장)
                success = await asyncio.to_thread(
                    self.vector_store.add_documents,
                    chunks,
                    force_update
                )
            
            if not success:
                self.logger.error("벡터 저장소 저장 실패")
                return False
            
            # FAISS/BM25 인덱스는 더 이상 사용하지 않음 (Qdrant Dense+Sparse만 사용)
            
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
                success = self.vector_store.replace_document_vectors(file_path, file_chunk_list)
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
    
    # ========== 비동기 메서드 (Phase 1: LLM 호출 비동기화) ==========
    
    async def query_async(self, question: str, max_sources: Optional[int] = None, score_threshold: Optional[float] = None, model_name: Optional[str] = None, retrievers: Optional[Dict[str, bool]] = None, session_id: Optional[str] = None, dense_weight: Optional[float] = None, sparse_weight: Optional[float] = None) -> RAGResponse:
        """
        비동기 질의응답 (LLM 호출, 검색, 리랭킹 모두 비동기 - Phase 3 완료)
        
        Args:
            question: 사용자 질문
            max_sources: 최대 소스 수
            score_threshold: 점수 임계값
            model_name: 사용할 LLM 모델명
            retrievers: 검색기 선택 정보
            session_id: 세션 ID (선택적, 기본 RAG에서는 사용하지 않지만 API 호환성을 위해 수락)
        """
        start_time = time.time()
        
        # session_id는 기본 RAG 시스템에서 사용하지 않지만, API 호환성을 위해 수락
        if session_id:
            self.logger.debug(f"세션 ID 수신: {session_id} (기본 RAG에서는 사용하지 않음)")
        
        try:
            self.logger.info(f"비동기 질의 처리 시작: {question[:50]}...")
            
            # 검색에 사용할 질문 (원본 그대로 사용)
            search_question = question
            
            # 기본값 적용
            max_sources = max_sources if max_sources is not None else self.rag_config.default_max_sources
            base_threshold = score_threshold if score_threshold is not None else self.rag_config.score_threshold
            
            # 동적 임계값 조정
            score_threshold = self._calculate_dynamic_threshold(
                question=search_question,
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
            is_general = is_general_question(search_question)
            self.logger.debug(f"질문 '{search_question}' 일반 질문 판별 결과: {is_general}")
            
            if is_general:
                # 일반 질문은 벡터 검색 없이 바로 LLM에 질문 (비동기)
                self.logger.info(f"일반 질문으로 판단: 벡터 검색 건너뛰기 (질문: '{search_question}')")
                llm_response = await self.llm_client.generate_answer_async(search_question, context="")
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
            self.logger.info(f"전문 질문으로 판단: 검색 수행 (질문: '{search_question}')")
            
            # 검색기 선택이 제공된 경우
            if retrievers is not None:
                self.logger.info(f"검색기 선택 사용: {retrievers}")
                
                # Qdrant만 사용 (FAISS/BM25 제거됨)
                selected_count = 1 if retrievers.get('use_qdrant', False) else 0
                
                search_limit = max_sources if selected_count == 1 else max_sources * 2
                self.logger.debug(f"검색기 개수: {selected_count}, 검색 제한: {search_limit}")
                
                all_results = []
                
                # Qdrant 검색 (비동기)
                if retrievers.get('use_qdrant', False):
                    try:
                        # Dense/Sparse 가중치 추출 (retrievers > 파라미터 > config 기본값)
                        from src.utils.config import get_qdrant_config
                        qdrant_config = get_qdrant_config()
                        config_dense_weight = getattr(qdrant_config, 'hybrid_search_dense_weight', 0.7)
                        config_sparse_weight = getattr(qdrant_config, 'hybrid_search_sparse_weight', 0.3)
                        
                        # 우선순위: retrievers > 파라미터 > config 기본값
                        effective_dense_weight = retrievers.get('dense_weight')
                        if effective_dense_weight is None:
                            effective_dense_weight = dense_weight
                        if effective_dense_weight is None:
                            effective_dense_weight = config_dense_weight
                        
                        effective_sparse_weight = retrievers.get('sparse_weight')
                        if effective_sparse_weight is None:
                            effective_sparse_weight = sparse_weight
                        if effective_sparse_weight is None:
                            effective_sparse_weight = config_sparse_weight
                        
                        self.logger.debug(f"Qdrant 검색 가중치: dense={effective_dense_weight:.2f}, sparse={effective_sparse_weight:.2f}")
                        
                        qdrant_results = await self.vector_store.search_similar_async(
                            query=search_question,
                            limit=search_limit,
                            score_threshold=score_threshold,
                            dense_weight=effective_dense_weight,
                            sparse_weight=effective_sparse_weight
                        )
                        if qdrant_results:
                            all_results.append(('qdrant', qdrant_results))
                            self.logger.debug(f"Qdrant 검색 결과: {len(qdrant_results)}개")
                    except Exception as e:
                        self.logger.warning(f"Qdrant 검색 실패: {str(e)}")
                
                # FAISS/BM25 검색은 더 이상 사용하지 않음 (Qdrant Dense+Sparse만 사용)
                
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
                    # 다중 검색기: RRF 통합 (Qdrant만 사용하므로 단일 검색기와 동일)
                    weights = retrievers.get('weights') or {'qdrant': 1.0}
                    name_to_weight = {
                        'qdrant': float(weights.get('qdrant', 1.0)),
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
                # retrievers가 None일 때: 기본 Qdrant 검색 사용 (dense/sparse 가중치 지원)
                from src.utils.config import get_qdrant_config
                qdrant_config = get_qdrant_config()
                
                # Dense/Sparse 가중치 결정 (파라미터 > retrievers > config 기본값)
                effective_dense_weight = dense_weight
                effective_sparse_weight = sparse_weight
                
                if effective_dense_weight is None or effective_sparse_weight is None:
                    # config에서 기본값 가져오기
                    config_dense_weight = getattr(qdrant_config, 'hybrid_search_dense_weight', 0.7)
                    config_sparse_weight = getattr(qdrant_config, 'hybrid_search_sparse_weight', 0.3)
                    
                    if effective_dense_weight is None:
                        effective_dense_weight = config_dense_weight
                    if effective_sparse_weight is None:
                        effective_sparse_weight = config_sparse_weight
                
                self.logger.info(f"기본 Qdrant 검색 사용: dense_weight={effective_dense_weight:.2f}, sparse_weight={effective_sparse_weight:.2f}")
                
                # Qdrant 검색 (dense/sparse 가중치 전달)
                similar_docs = await self.vector_store.search_similar_async(
                    query=search_question,
                    limit=max_sources,
                    score_threshold=score_threshold,
                    dense_weight=effective_dense_weight,
                    sparse_weight=effective_sparse_weight
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
            
            # 검색 결과 로깅
            self.logger.info(f"🔍 검색 결과 처리 시작: 총 {len(similar_docs)}개 문서 검색됨")
            for idx, doc in enumerate(similar_docs, 1):
                source_file = doc.get('source_file', 'N/A')
                chunk_index = doc.get('chunk_index', 'N/A')
                score = doc.get('score', 0.0)
                self.logger.info(f"  [{idx}] 파일: {source_file}, 청크: {chunk_index}, 점수: {score:.4f}")
            
            # 중복 청크 제거
            unique_docs = []
            seen_chunks = set()
            for doc in similar_docs:
                chunk_key = f"{doc.get('source_file', '')}:{doc.get('chunk_index', '')}"
                if chunk_key not in seen_chunks:
                    seen_chunks.add(chunk_key)
                    unique_docs.append(doc)
            removed_duplicates = len(similar_docs) - len(unique_docs)
            if removed_duplicates > 0:
                self.logger.info(f"🔄 중복 청크 제거: {removed_duplicates}개 제거됨 (남은 문서: {len(unique_docs)}개)")
            similar_docs = unique_docs
            
            # 표 데이터 중복 제거
            if len(similar_docs) > 1:
                table_docs = [doc for doc in similar_docs if '표 데이터' in doc.get('content', '')]
                if len(table_docs) > 1:
                    table_docs.sort(key=lambda x: x.get('score', 0), reverse=True)
                    removed_table_duplicates = 0
                    for table_doc in table_docs[1:]:
                        if table_doc in similar_docs:
                            similar_docs.remove(table_doc)
                            removed_table_duplicates += 1
                    if removed_table_duplicates > 0:
                        self.logger.info(f"🔄 표 데이터 중복 제거: {removed_table_duplicates}개 제거됨 (남은 문서: {len(similar_docs)}개)")
            
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
            
            # LLM을 통한 답변 생성 (비동기) - 정제된 질문 사용
            llm_response = await self.llm_client.generate_answer_async(search_question, context)
            answer = llm_response.text if llm_response else "답변을 생성할 수 없습니다."
            is_general = llm_response.is_general if llm_response else False
            has_rag_context = llm_response.has_rag_context if llm_response else True
            
            # 신뢰도 계산
            confidence = self._calculate_confidence(similar_docs, answer)
            
            # 소스 정보 정리
            sources = self._format_sources(similar_docs)
            self.logger.info(f"📚 최종 참조 문서: {len(sources)}개")
            for idx, source in enumerate(sources, 1):
                source_file = source.get('source_file', 'N/A')
                chunk_index = source.get('chunk_index', 'N/A')
                relevance_score = source.get('relevance_score', 0.0)
                self.logger.info(f"  [{idx}] 파일: {source_file}, 청크: {chunk_index}, 관련도: {relevance_score:.4f}")
            
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
        기본 임계값이 낮아졌으므로 조정 폭을 줄여 더 관대하게 처리합니다.
        
        Args:
            question: 사용자 질문
            base_threshold: 기본 임계값
            max_sources: 요청된 최대 소스 수
            
        Returns:
            조정된 임계값
        """
        threshold = base_threshold
        
        # 1. 질문 길이에 따른 조정 (더 관대하게)
        question_length = len(question.strip())
        if question_length < 10:
            # 매우 짧은 질문: 임계값 약간 증가
            threshold += 0.05
        elif question_length > 50:
            # 긴 질문: 임계값 감소 (더 많은 결과 포함)
            threshold -= 0.05
        
        # 2. 요청된 문서 수에 따른 조정 (더 관대하게)
        if max_sources <= 3:
            # 적은 수의 문서 요청: 임계값 약간 증가
            threshold += 0.05
        elif max_sources >= 10:
            # 많은 수의 문서 요청: 임계값 감소 (더 넓은 범위)
            threshold -= 0.05
        
        # 3. 질문 유형에 따른 조정
        question_lower = question.lower()
        
        # 키워드 기반 질문 (예: "변압기 진단 기준")
        if any(keyword in question_lower for keyword in ['기준', '방법', '절차', '과정', '원리']):
            # 구체적인 정보 요청: 임계값 약간 감소
            threshold -= 0.03
        
        # 비교/분석 질문 (예: "차이점", "비교")
        if any(keyword in question_lower for keyword in ['차이', '비교', '분석', '대비']):
            # 여러 문서 비교 필요: 임계값 감소
            threshold -= 0.05
        
        # 표/데이터 질문
        if any(keyword in question_lower for keyword in ['표', 'table', '데이터', '수치']):
            # 표 데이터는 정확한 매칭 필요: 임계값 약간 증가
            threshold += 0.03
        
        # 임계값 범위 제한 (0.0 ~ 1.0)
        # 최소 임계값을 0.2로 설정하여 너무 낮은 점수는 제외
        threshold = max(0.2, min(1.0, threshold))
        
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
        
        # 점수 정규화를 위한 최대/최소 점수 계산
        if similar_docs:
            scores = [doc.get('score', 0.0) for doc in similar_docs]
            if not scores:
                max_score = 1.0
                min_score = 0.0
                score_range = 1.0
            else:
                max_score = max(scores)
                min_score = min(scores)
                score_range = max_score - min_score if max_score > min_score else 1.0
                
                # 모든 점수가 같을 때 처리 (정규화 불가능)
                if score_range == 0.0:
                    # 모든 점수가 같으면 원본 점수를 그대로 사용 (정규화 없음)
                    self.logger.warning(
                        f"모든 점수가 동일함 ({max_score:.4f}). "
                        f"정규화 없이 원본 점수를 그대로 사용합니다. "
                        f"문서 수: {len(similar_docs)}개"
                    )
                    score_range = 1.0  # 나눗셈 오류 방지
                else:
                    # 정규화 가능한 경우 점수 범위 로그
                    self.logger.debug(
                        f"점수 정규화: 범위={min_score:.4f}~{max_score:.4f}, "
                        f"문서 수={len(similar_docs)}개"
                    )
        else:
            max_score = 1.0
            min_score = 0.0
            score_range = 1.0
            self.logger.warning("정규화할 문서가 없습니다. 기본값 사용")
        
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
            
            # 점수 정규화 (0-1 범위로)
            raw_score = doc.get('score', 0.0)
            
            # 원본 점수 검증
            if raw_score < 0.0 or raw_score > 1.0:
                self.logger.warning(
                    f"원본 점수가 0-1 범위를 벗어남: {raw_score:.4f}. "
                    f"자동 클리핑 적용"
                )
                raw_score = max(0.0, min(1.0, raw_score))
            
            # 모든 점수가 같을 때는 원본 점수를 그대로 사용
            if max_score == min_score:
                # 모든 점수가 같으면 원본 점수를 그대로 사용 (정규화 없음)
                normalized_score = raw_score
            else:
                # 개선된 정규화: 최소값을 0으로 만들지 않고, 최대값 기준으로 상대적 점수 계산
                # 이렇게 하면 최소값 문서도 0이 아닌 값을 가짐
                # 방법: score / max (최대값 기준 정규화)
                # 예: 점수 0.6491 / 최대값 0.7318 = 0.887 (0이 아닌 값)
                if max_score > 0:
                    # 최대값 기준 정규화 (원본 점수의 상대적 비율 유지)
                    # 모든 문서가 최대값 대비 상대적 점수를 가지므로, 최소값도 0이 아님
                    normalized_score = raw_score / max_score
                    self.logger.debug(
                        f"최대값 기준 정규화: {raw_score:.4f} / {max_score:.4f} = {normalized_score:.4f}"
                    )
                else:
                    # max_score가 0이면 원본 점수 사용
                    self.logger.warning(f"max_score가 0입니다. 원본 점수 사용: {raw_score:.4f}")
                    normalized_score = raw_score
            
            # 0-1 범위로 클리핑
            normalized_score = max(0.0, min(1.0, normalized_score))
            
            # 정규화 결과 검증
            if normalized_score < 0.0 or normalized_score > 1.0:
                self.logger.error(
                    f"정규화 후 점수가 0-1 범위를 벗어남: {normalized_score:.4f}. "
                    f"원본 점수: {raw_score:.4f}, 범위: {min_score:.4f}~{max_score:.4f}"
                )
                normalized_score = max(0.0, min(1.0, normalized_score))
            
            source = {
                'content': preview,
                'source_file': doc['source_file'],
                'source_path': source_path,  # 계층 형식 출처 경로 추가
                'relevance_score': normalized_score,  # 정규화된 점수 (0-1 범위)
                'raw_score': raw_score,  # 원본 점수도 보존 (디버깅용)
                'chunk_index': doc['chunk_index'],
                'metadata': metadata  # 메타데이터 전체도 포함
            }
            sources.append(source)
        
        return sources
    
    def get_system_stats(self) -> Dict[str, Any]:
        """시스템 통계 반환"""
        try:
            # 벡터 저장소 통계
            vector_stats = self.vector_store.get_collection_info()
            
            # 임베딩 캐시 통계
            embedding_stats = {
                'cache_size': 0,  # BGE-m3는 캐시를 사용하지 않음
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
            return self.vector_store.get_documents_info()
        except Exception as e:
            self.logger.error(f"문서 정보 조회 실패: {str(e)}")
            return []
    
    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """특정 문서의 청크 정보 반환"""
        try:
            return self.vector_store.get_document_chunks(document_id)
        except Exception as e:
            self.logger.error(f"문서 청크 조회 실패: {str(e)}")
            return []
    
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
            
            # 기본값 적용
            max_sources = max_sources if max_sources is not None else self.rag_config.default_max_sources_table
            score_threshold = score_threshold if score_threshold is not None else self.rag_config.score_threshold
            
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
            
            # 1. 표 제목으로 검색 (비동기 메서드 사용)
            import asyncio
            similar_docs = asyncio.run(
                self.vector_store.search_with_table_filter_async(
                    query=table_title,
                    table_title=table_title,
                    limit=max_sources,
                    score_threshold=score_threshold
                )
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
            
            # 1. 필터와 함께 검색 (비동기 메서드 사용)
            import asyncio
            similar_docs = asyncio.run(
                self.vector_store.search_with_table_filter_async(
                    query=question,
                    table_title=table_title,
                    is_table_data=is_table_data,
                    limit=max_sources,
                    score_threshold=score_threshold
                )
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
        if not rag_system.vector_store.create_collection(force_recreate=False):
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
