"""
세션 데이터 모델
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Deque, Any
from collections import deque
import time


@dataclass
class ChatMessage:
    """채팅 메시지"""
    role: str  # 'user' | 'assistant'
    content: str
    timestamp: float
    search_results: Optional[List[Dict]] = None
    model_used: Optional[str] = None
    confidence: Optional[float] = None
    processing_time: Optional[float] = None


@dataclass
class SessionCache:
    """세션 캐시"""
    session_id: str
    created_at: float
    last_accessed: float
    message_history: Deque[ChatMessage] = field(default_factory=lambda: deque(maxlen=50))
    search_cache: Dict[str, Any] = field(default_factory=dict)
    search_cache_max_ttl_sec: int = 3600
    estimated_memory_usage: int = 0
    
    def add_message(self, message: ChatMessage):
        """메시지 추가"""
        self.message_history.append(message)
        self.last_accessed = time.time()
        # 메모리 사용량 추정 (간단한 계산)
        self.estimated_memory_usage += len(message.content.encode('utf-8'))
        if message.search_results:
            self.estimated_memory_usage += sum(
                len(str(r).encode('utf-8')) for r in message.search_results
            )
    
    def cleanup_old_messages(self, max_memory_mb: int = 100):
        """오래된 메시지 정리 (메모리 한도 초과 시)"""
        max_memory_bytes = max_memory_mb * 1024 * 1024
        
        while self.estimated_memory_usage > max_memory_bytes and len(self.message_history) > 1:
            # 가장 오래된 메시지 제거
            old_message = self.message_history.popleft()
            self.estimated_memory_usage -= len(old_message.content.encode('utf-8'))
            if old_message.search_results:
                self.estimated_memory_usage -= sum(
                    len(str(r).encode('utf-8')) for r in old_message.search_results
                )


@dataclass
class SessionStats:
    """세션 통계"""
    total_sessions: int
    total_memory_mb: float
    oldest_session_age: float
    newest_session_age: float
    active_sessions: int

