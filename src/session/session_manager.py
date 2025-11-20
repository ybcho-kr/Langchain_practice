"""
세션 관리자
세션 라이프사이클 관리, 컨텍스트 구성, 메모리 관리
"""

import uuid
import time
from typing import Optional, List, Dict, Any
from src.utils.logger import get_logger
from src.session.session_storage import SessionStorage
from src.session.models import SessionCache, ChatMessage, SessionStats
from src.utils.config import SessionConfig, get_session_config


class SessionManager:
    """세션 관리자"""
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        use_sqlite: Optional[bool] = None,
        max_context_length: Optional[int] = None,
        max_turns: Optional[int] = None,
        session_ttl: Optional[int] = None,
        max_memory_per_session_mb: Optional[int] = None,
        max_total_memory_mb: Optional[int] = None,
        cleanup_interval_seconds: Optional[int] = None,
        session_config: Optional[SessionConfig] = None
    ):
        """
        Args:
            storage_path: 저장소 경로
            use_sqlite: SQLite 사용 여부
            max_context_length: 최대 컨텍스트 길이 (문자)
            max_turns: 최대 턴 수
            session_ttl: 세션 TTL (초)
            max_memory_per_session_mb: 세션당 최대 메모리 (MB)
            max_total_memory_mb: 전체 최대 메모리 (MB)
        """
        self.logger = get_logger()
        self.session_config = session_config or self._load_session_config()
        
        def resolve(value, attr, default):
            if value is not None:
                return value
            if self.session_config:
                return getattr(self.session_config, attr, default)
            return default
        
        resolved_storage_path = resolve(storage_path, "storage_path", "data/sessions")
        resolved_use_sqlite = resolve(use_sqlite, "use_sqlite", True)
        resolved_max_context_length = resolve(max_context_length, "max_context_length", 4000)
        resolved_max_turns = resolve(max_turns, "max_turns", 50)
        resolved_session_ttl = resolve(session_ttl, "session_ttl", 3600)
        resolved_max_memory_per_session_mb = resolve(max_memory_per_session_mb, "max_memory_per_session_mb", 100)
        resolved_max_total_memory_mb = resolve(max_total_memory_mb, "max_total_memory_mb", 1024)
        resolved_cleanup_interval = resolve(cleanup_interval_seconds, "cleanup_interval_seconds", 300)
        
        self.max_context_length = resolved_max_context_length
        self.max_turns = resolved_max_turns
        self.max_memory_per_session_mb = resolved_max_memory_per_session_mb
        self.max_total_memory_mb = resolved_max_total_memory_mb
        self.session_ttl = resolved_session_ttl
        self.cleanup_interval_seconds = resolved_cleanup_interval
        
        # 세션 저장소 초기화
        self.storage = SessionStorage(
            storage_path=resolved_storage_path,
            use_sqlite=resolved_use_sqlite,
            session_ttl=resolved_session_ttl,
            cleanup_interval=resolved_cleanup_interval
        )
        self.logger.info(
            f"세션 저장소 초기화: path={resolved_storage_path}, "
            f"sqlite={resolved_use_sqlite}, ttl={resolved_session_ttl}, cleanup={resolved_cleanup_interval}s"
        )
    
    def _load_session_config(self) -> Optional[SessionConfig]:
        """설정 파일에서 세션 설정을 로드"""
        try:
            return get_session_config()
        except Exception as exc:
            self.logger.warning(f"세션 설정을 로드하지 못했습니다. 기본값을 사용합니다: {exc}")
            return None
    
    def create_session(self, session_id: Optional[str] = None) -> SessionCache:
        """
        세션 생성
        
        Args:
            session_id: 세션 ID (None이면 자동 생성)
            
        Returns:
            생성된 세션 캐시
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        session = self.storage.create_session(session_id)
        self.logger.info(f"세션 생성: {session_id}")
        return session
    
    def get_session(self, session_id: str) -> Optional[SessionCache]:
        """
        세션 조회
        
        Args:
            session_id: 세션 ID
            
        Returns:
            세션 캐시 (없으면 None)
        """
        return self.storage.get_session(session_id)
    
    def delete_session(self, session_id: str) -> bool:
        """
        세션 삭제
        
        Args:
            session_id: 세션 ID
            
        Returns:
            삭제 성공 여부
        """
        return self.storage.delete_session(session_id)
    
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        search_results: Optional[List[Dict]] = None,
        model_used: Optional[str] = None,
        confidence: Optional[float] = None,
        processing_time: Optional[float] = None
    ) -> bool:
        """
        메시지 추가
        
        Args:
            session_id: 세션 ID
            role: 메시지 역할 ('user' | 'assistant')
            content: 메시지 내용
            search_results: 검색 결과 (선택적)
            model_used: 사용된 모델 (선택적)
            confidence: 신뢰도 (선택적)
            processing_time: 처리 시간 (선택적)
            
        Returns:
            추가 성공 여부
        """
        session = self.get_session(session_id)
        if not session:
            session = self.create_session(session_id)
        
        message = ChatMessage(
            role=role,
            content=content,
            timestamp=time.time(),
            search_results=search_results,
            model_used=model_used,
            confidence=confidence,
            processing_time=processing_time
        )
        
        session.add_message(message)
        
        # 메모리 관리
        self._manage_memory()
        
        # 저장
        self.storage.save_session(session)
        
        return True
    
    def get_history(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        세션 히스토리 조회
        
        Args:
            session_id: 세션 ID
            limit: 최대 메시지 수 (None이면 전체)
            
        Returns:
            메시지 히스토리 리스트
        """
        session = self.get_session(session_id)
        if not session:
            return []
        
        messages = list(session.message_history)
        if limit:
            messages = messages[-limit:]
        
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp,
                "model_used": msg.model_used,
                "confidence": msg.confidence,
                "processing_time": msg.processing_time
            }
            for msg in messages
        ]
    
    def build_context(
        self,
        session_id: str,
        max_length: Optional[int] = None
    ) -> str:
        """
        컨텍스트 구성
        
        최근 메시지부터 역순으로 누적하여 최대 길이까지 구성합니다.
        
        Args:
            session_id: 세션 ID
            max_length: 최대 길이 (None이면 기본값 사용)
            
        Returns:
            구성된 컨텍스트 문자열
        """
        if max_length is None:
            max_length = self.max_context_length
        
        session = self.get_session(session_id)
        if not session:
            return ""
        
        # 최근 메시지부터 역순으로
        messages = list(session.message_history)
        messages.reverse()
        
        context_parts = []
        total_length = 0
        
        for msg in messages:
            # 표/검색 결과가 포함된 메시지는 우선 포함
            has_table = '표' in msg.content or 'table' in msg.content.lower()
            has_search_results = msg.search_results is not None and len(msg.search_results) > 0
            
            if has_table or has_search_results:
                # 우선 포함 (길이 제한 무시)
                context_parts.insert(0, f"[{msg.role}]: {msg.content}")
                total_length += len(msg.content)
            else:
                # 일반 메시지는 길이 제한 확인
                msg_length = len(msg.content)
                if total_length + msg_length <= max_length:
                    context_parts.insert(0, f"[{msg.role}]: {msg.content}")
                    total_length += msg_length
                else:
                    # 남은 공간만 사용
                    remaining = max_length - total_length
                    if remaining > 100:  # 최소 100자 이상만 포함
                        truncated = msg.content[:remaining] + "..."
                        context_parts.insert(0, f"[{msg.role}]: {truncated}")
                    break
        
        return "\n\n".join(context_parts)
    
    def get_cached_search_results(
        self,
        session_id: str,
        cache_key: str
    ) -> Optional[Dict[str, Any]]:
        """
        캐시된 검색 결과 조회
        
        Args:
            session_id: 세션 ID
            cache_key: 캐시 키
            
        Returns:
            캐시된 검색 결과 (없으면 None)
        """
        session = self.get_session(session_id)
        if not session:
            return None
        
        # TTL 확인 (간단한 구현, 실제로는 타임스탬프 확인 필요)
        cached = session.search_cache.get(cache_key)
        if cached:
            # 타임스탬프 확인 (1시간 TTL)
            if time.time() - cached.get('timestamp', 0) < 3600:
                return cached.get('data')
            else:
                # 만료된 캐시 삭제
                del session.search_cache[cache_key]
        
        return None
    
    def cache_search_results(
        self,
        session_id: str,
        cache_key: str,
        data: Dict[str, Any]
    ):
        """
        검색 결과 캐싱
        
        Args:
            session_id: 세션 ID
            cache_key: 캐시 키
            data: 캐시할 데이터
        """
        session = self.get_session(session_id)
        if not session:
            session = self.create_session(session_id)
        
        session.search_cache[cache_key] = {
            'data': data,
            'timestamp': time.time()
        }
        
        self.storage.save_session(session)
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        모든 세션 목록 조회 (제목, 마지막 메시지 등 포함)
        
        Returns:
            세션 정보 리스트 (최신 순으로 정렬)
        """
        sessions = self.storage.list_sessions()
        
        result = []
        for session in sessions:
            # 세션 제목 추출 (첫 사용자 메시지의 처음 50자)
            title = "새 대화"
            last_message = ""
            message_count = len(session.message_history)
            
            # 첫 사용자 메시지 찾기
            for msg in session.message_history:
                if msg.role == 'user':
                    title = msg.content[:50] if len(msg.content) > 50 else msg.content
                    title = title.replace('\n', ' ').strip()
                    if not title:
                        title = "새 대화"
                    break
            
            # 마지막 메시지 찾기
            if session.message_history:
                last_msg = session.message_history[-1]
                last_message = last_msg.content[:100] if len(last_msg.content) > 100 else last_msg.content
                last_message = last_message.replace('\n', ' ').strip()
            
            result.append({
                "session_id": session.session_id,
                "title": title,
                "last_message": last_message,
                "created_at": session.created_at,
                "last_accessed": session.last_accessed,
                "message_count": message_count
            })
        
        return result
    
    def get_stats(self) -> SessionStats:
        """
        세션 통계 조회
        
        Returns:
            세션 통계
        """
        return self.storage.get_stats()
    
    def _manage_memory(self):
        """메모리 관리 (전체 메모리 한도 초과 시 정리)"""
        stats = self.get_stats()
        
        if stats.total_memory_mb > self.max_total_memory_mb:
            self.logger.warning(
                f"전체 메모리 한도 초과: {stats.total_memory_mb:.2f}MB > {self.max_total_memory_mb}MB"
            )
            
            # 공격적 정리: 상위 10% 세션의 메시지 절반 축소
            # (간단한 구현, 실제로는 더 정교한 알고리즘 사용 가능)
            sessions = list(self.storage.memory_cache.values())
            if sessions:
                # 메모리 사용량 기준 정렬
                sessions.sort(key=lambda s: s.estimated_memory_usage, reverse=True)
                
                # 상위 10% 세션 처리
                top_10_percent = max(1, len(sessions) // 10)
                for session in sessions[:top_10_percent]:
                    # 메시지 절반 제거
                    target_size = len(session.message_history) // 2
                    while len(session.message_history) > target_size:
                        old_msg = session.message_history.popleft()
                        session.estimated_memory_usage -= len(old_msg.content.encode('utf-8'))
                    
                    self.storage.save_session(session)
                
                self.logger.info(f"메모리 정리 완료: {top_10_percent}개 세션 처리")

