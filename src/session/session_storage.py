"""
세션 저장소
메모리 기반 세션 저장소 및 SQLite 영구 저장소
"""

import sqlite3
import json
import time
import threading
from typing import Dict, Optional, List
from pathlib import Path
from collections import deque
from cachetools import TTLCache

from src.utils.logger import get_logger
from src.session.models import SessionCache, ChatMessage, SessionStats


class SessionStorage:
    """세션 저장소"""
    
    def __init__(
        self,
        storage_path: str = "data/sessions",
        use_sqlite: bool = False,
        session_ttl: int = 3600,
        cleanup_interval: int = 300
    ):
        """
        Args:
            storage_path: 저장소 경로
            use_sqlite: SQLite 영구 저장소 사용 여부
            session_ttl: 세션 TTL (초)
            cleanup_interval: 정리 작업 주기 (초)
        """
        self.logger = get_logger()
        self.storage_path = Path(storage_path)
        self.use_sqlite = use_sqlite
        self.session_ttl = session_ttl
        self.cleanup_interval = cleanup_interval
        
        # 메모리 캐시 (TTL 기반)
        self.memory_cache: TTLCache[str, SessionCache] = TTLCache(
            maxsize=1000,
            ttl=session_ttl
        )
        
        # SQLite 연결 (옵션)
        self.db_path = self.storage_path / "sessions.db"
        self.db_conn: Optional[sqlite3.Connection] = None
        
        if self.use_sqlite:
            self._init_sqlite()
        
        # 백그라운드 정리 스레드 시작
        self._cleanup_thread = None
        self._stop_cleanup = False
        self._start_cleanup_thread()
    
    def _init_sqlite(self):
        """SQLite 데이터베이스 초기화"""
        try:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            
            self.db_conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False
            )
            self.db_conn.row_factory = sqlite3.Row
            
            # 테이블 생성
            cursor = self.db_conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at REAL,
                    last_accessed REAL,
                    message_history TEXT,
                    search_cache TEXT,
                    estimated_memory_usage INTEGER
                )
            """)
            self.db_conn.commit()
            
            self.logger.info(f"SQLite 세션 저장소 초기화 완료: {self.db_path}")
            self.logger.info(f"SQLite 사용 여부: {self.use_sqlite}, 파일 경로: {self.db_path.absolute()}")
        except Exception as e:
            self.logger.error(f"SQLite 초기화 실패: {str(e)}")
            import traceback
            self.logger.error(f"상세 오류: {traceback.format_exc()}")
            self.use_sqlite = False
    
    def _start_cleanup_thread(self):
        """백그라운드 정리 스레드 시작"""
        def cleanup_loop():
            while not self._stop_cleanup:
                try:
                    self.cleanup_expired_sessions()
                    time.sleep(self.cleanup_interval)
                except Exception as e:
                    self.logger.error(f"정리 작업 오류: {str(e)}")
        
        self._cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        self._cleanup_thread.start()
        self.logger.info(f"세션 정리 스레드 시작 (주기: {self.cleanup_interval}초)")
    
    def create_session(self, session_id: str) -> SessionCache:
        """
        세션 생성
        
        Args:
            session_id: 세션 ID
            
        Returns:
            생성된 세션 캐시
        """
        now = time.time()
        session = SessionCache(
            session_id=session_id,
            created_at=now,
            last_accessed=now
        )
        
        self.memory_cache[session_id] = session
        
        if self.use_sqlite and self.db_conn:
            self._save_to_sqlite(session)
        
        self.logger.debug(f"세션 생성: {session_id}")
        return session
    
    def get_session(self, session_id: str, extend_ttl: bool = True) -> Optional[SessionCache]:
        """
        세션 조회
        
        Args:
            session_id: 세션 ID
            extend_ttl: 만료된 세션도 TTL 연장하여 복원할지 여부 (기본값: True)
            
        Returns:
            세션 캐시 (없으면 None)
        """
        # 메모리 캐시에서 조회
        session = self.memory_cache.get(session_id)
        
        if session:
            session.last_accessed = time.time()
            return session
        
        # SQLite에서 조회 (옵션)
        if self.use_sqlite and self.db_conn:
            session = self._load_from_sqlite(session_id)
            if session:
                now = time.time()
                # 만료 여부 확인
                time_since_access = now - session.last_accessed
                
                if time_since_access < self.session_ttl:
                    # 만료되지 않았으면 메모리 캐시에 복원
                    session.last_accessed = now
                    self.memory_cache[session_id] = session
                    return session
                elif extend_ttl:
                    # 만료되었지만 extend_ttl=True이면 TTL 연장하여 복원
                    # (기존 대화를 유지하기 위해)
                    session.last_accessed = now
                    self.memory_cache[session_id] = session
                    self._save_to_sqlite(session)  # 갱신된 접근 시간 저장
                    self.logger.debug(f"만료된 세션 복원 및 TTL 연장: {session_id} (만료 후 {time_since_access:.0f}초 경과)")
                    return session
                else:
                    # 만료되었고 extend_ttl=False이면 None 반환
                    self.logger.debug(f"만료된 세션 조회 실패: {session_id} (만료 후 {time_since_access:.0f}초 경과)")
                    return None
        
        return None
    
    def save_session(self, session: SessionCache):
        """
        세션 저장
        
        Args:
            session: 세션 캐시
        """
        session.last_accessed = time.time()
        self.memory_cache[session.session_id] = session
        
        if self.use_sqlite and self.db_conn:
            self._save_to_sqlite(session)
    
    def delete_session(self, session_id: str) -> bool:
        """
        세션 삭제
        
        Args:
            session_id: 세션 ID
            
        Returns:
            삭제 성공 여부
        """
        # 메모리 캐시에서 삭제
        if session_id in self.memory_cache:
            del self.memory_cache[session_id]
        
        # SQLite에서 삭제 (옵션)
        if self.use_sqlite and self.db_conn:
            try:
                cursor = self.db_conn.cursor()
                cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
                self.db_conn.commit()
            except Exception as e:
                self.logger.error(f"SQLite 세션 삭제 실패: {str(e)}")
                return False
        
        self.logger.debug(f"세션 삭제: {session_id}")
        return True
    
    def cleanup_expired_sessions(self):
        """만료된 세션 정리"""
        now = time.time()
        expired_count = 0
        
        # 메모리 캐시는 TTLCache가 자동으로 정리
        # 여기서는 SQLite 정리만 수행
        if self.use_sqlite and self.db_conn:
            try:
                cursor = self.db_conn.cursor()
                cursor.execute(
                    "DELETE FROM sessions WHERE last_accessed < ?",
                    (now - self.session_ttl,)
                )
                expired_count = cursor.rowcount
                self.db_conn.commit()
                
                if expired_count > 0:
                    self.logger.info(f"만료된 세션 정리: {expired_count}개")
            except Exception as e:
                self.logger.error(f"만료 세션 정리 실패: {str(e)}")
    
    def list_sessions(self) -> List[SessionCache]:
        """
        모든 세션 목록 조회
        
        Returns:
            세션 캐시 리스트 (최신 순으로 정렬)
        """
        sessions = []
        
        # 메모리 캐시에서 모든 세션 가져오기
        for session_id in self.memory_cache.keys():
            session = self.memory_cache.get(session_id)
            if session:
                sessions.append(session)
        
        # SQLite에서도 조회 (메모리에 없는 세션 포함)
        if self.use_sqlite and self.db_conn:
            try:
                cursor = self.db_conn.cursor()
                cursor.execute("SELECT session_id FROM sessions")
                rows = cursor.fetchall()
                
                for row in rows:
                    session_id = row["session_id"]
                    # 메모리에 없으면 로드
                    if session_id not in self.memory_cache:
                        session = self._load_from_sqlite(session_id)
                        if session:
                            sessions.append(session)
            except Exception as e:
                self.logger.error(f"SQLite 세션 목록 조회 실패: {str(e)}")
        
        # last_accessed 기준으로 최신 순 정렬
        sessions.sort(key=lambda s: s.last_accessed, reverse=True)
        
        return sessions
    
    def get_stats(self) -> SessionStats:
        """
        세션 통계 조회
        
        Returns:
            세션 통계
        """
        sessions = list(self.memory_cache.values())
        
        if not sessions:
            return SessionStats(
                total_sessions=0,
                total_memory_mb=0.0,
                oldest_session_age=0.0,
                newest_session_age=0.0,
                active_sessions=0
            )
        
        now = time.time()
        total_memory = sum(s.estimated_memory_usage for s in sessions)
        ages = [now - s.created_at for s in sessions]
        
        return SessionStats(
            total_sessions=len(sessions),
            total_memory_mb=total_memory / (1024 * 1024),
            oldest_session_age=max(ages) if ages else 0.0,
            newest_session_age=min(ages) if ages else 0.0,
            active_sessions=len([s for s in sessions if now - s.last_accessed < 300])
        )
    
    def _save_to_sqlite(self, session: SessionCache):
        """SQLite에 세션 저장"""
        if not self.db_conn:
            self.logger.warning(f"SQLite 연결이 없어 세션 저장 실패: {session.session_id}")
            return
        
        try:
            # 메시지 히스토리 직렬화
            message_history = [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp,
                    "model_used": msg.model_used,
                    "confidence": msg.confidence,
                    "processing_time": msg.processing_time
                }
                for msg in session.message_history
            ]
            
            cursor = self.db_conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO sessions
                (session_id, created_at, last_accessed, message_history, search_cache, estimated_memory_usage)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                session.session_id,
                session.created_at,
                session.last_accessed,
                json.dumps(message_history, ensure_ascii=False),
                json.dumps(session.search_cache, ensure_ascii=False),
                session.estimated_memory_usage
            ))
            self.db_conn.commit()
            self.logger.debug(f"SQLite 세션 저장 완료: {session.session_id}")
        except Exception as e:
            self.logger.error(f"SQLite 세션 저장 실패: {session.session_id}, 오류: {str(e)}")
            import traceback
            self.logger.error(f"상세 오류: {traceback.format_exc()}")
    
    def _load_from_sqlite(self, session_id: str) -> Optional[SessionCache]:
        """SQLite에서 세션 로드"""
        if not self.db_conn:
            return None
        
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            # 메시지 히스토리 역직렬화
            message_history_data = json.loads(row["message_history"])
            message_history = deque(maxlen=50)
            for msg_data in message_history_data:
                message_history.append(ChatMessage(
                    role=msg_data["role"],
                    content=msg_data["content"],
                    timestamp=msg_data["timestamp"],
                    model_used=msg_data.get("model_used"),
                    confidence=msg_data.get("confidence"),
                    processing_time=msg_data.get("processing_time")
                ))
            
            # 검색 캐시 역직렬화
            search_cache = json.loads(row["search_cache"])
            
            session = SessionCache(
                session_id=row["session_id"],
                created_at=row["created_at"],
                last_accessed=row["last_accessed"],
                message_history=message_history,
                search_cache=search_cache,
                estimated_memory_usage=row["estimated_memory_usage"]
            )
            
            return session
        except Exception as e:
            self.logger.error(f"SQLite 세션 로드 실패: {str(e)}")
            return None
    
    def close(self):
        """저장소 종료"""
        self._stop_cleanup = True
        if self.db_conn:
            self.db_conn.close()

