import { useState, useEffect } from 'react';
import { useAppStore } from '../../stores/appStore';
import { apiClient } from '../../services/api';

const TABS = [
  { id: 'chat', label: '💬 채팅', icon: '💬' },
  { id: 'documents', label: '📚 문서 관리', icon: '📚' },
  { id: 'settings', label: '⚙️ 설정', icon: '⚙️' },
  { id: 'evidence', label: '🔍 답변 근거', icon: '🔍' },
];

/**
 * ChatGPT 스타일 사이드바 컴포넌트
 * 탭 네비게이션 및 세션 목록 표시 및 관리
 */
export default function Sidebar({ activeTab, onTabChange }) {
  const {
    sessions,
    setSessions,
    selectedSessionId,
    setSelectedSessionId,
    currentSessionId,
    setCurrentSessionId,
  } = useAppStore();

  const [loading, setLoading] = useState(false);
  const [hoveredSessionId, setHoveredSessionId] = useState(null);

  // 세션 목록 로드
  const loadSessions = async () => {
    setLoading(true);
    try {
      const response = await apiClient.getSessions();
      if (response && response.sessions) {
        const newSessions = response.sessions;
        setSessions(newSessions);
        
        // 세션 목록이 비어있거나, 현재 선택된 세션이 목록에 없으면 선택 해제
        if (newSessions.length === 0) {
          if (selectedSessionId) {
            console.log('[Sidebar] 세션 목록이 비어있음, 선택 해제');
            setSelectedSessionId(null);
            setCurrentSessionId(null);
            // 채팅 화면 초기화 이벤트 발생
            window.dispatchEvent(new CustomEvent('sessionSelected', { detail: null }));
          }
        } else {
          // 현재 선택된 세션이 목록에 있는지 확인
          const sessionExists = newSessions.some(s => s.session_id === selectedSessionId);
          if (selectedSessionId && !sessionExists) {
            console.log('[Sidebar] 선택된 세션이 목록에 없음, 선택 해제');
            setSelectedSessionId(null);
            setCurrentSessionId(null);
            // 채팅 화면 초기화 이벤트 발생
            window.dispatchEvent(new CustomEvent('sessionSelected', { detail: null }));
          }
        }
      } else {
        setSessions([]);
        // 세션 목록이 비어있으면 선택 해제
        if (selectedSessionId) {
          setSelectedSessionId(null);
          setCurrentSessionId(null);
          window.dispatchEvent(new CustomEvent('sessionSelected', { detail: null }));
        }
      }
    } catch (error) {
      console.error('세션 목록 로드 실패:', error);
      setSessions([]);
      if (selectedSessionId) {
        setSelectedSessionId(null);
        setCurrentSessionId(null);
        window.dispatchEvent(new CustomEvent('sessionSelected', { detail: null }));
      }
    } finally {
      setLoading(false);
    }
  };

  // 초기 로드 및 주기적 새로고침
  useEffect(() => {
    loadSessions();
    // 30초마다 세션 목록 새로고침 (너무 자주 호출하지 않도록)
    const interval = setInterval(loadSessions, 30000);
    
    // 세션 생성 이벤트 리스너 (새 세션이 생성되면 목록 새로고침)
    const handleSessionCreated = () => {
      loadSessions();
    };
    window.addEventListener('sessionCreated', handleSessionCreated);
    
    return () => {
      clearInterval(interval);
      window.removeEventListener('sessionCreated', handleSessionCreated);
    };
  }, []);

  // 새 대화 생성
  const handleNewChat = async () => {
    try {
      const response = await apiClient.createSession();
      if (response && response.session_id) {
        setCurrentSessionId(response.session_id);
        setSelectedSessionId(response.session_id);
        // 세션 목록 새로고침
        await loadSessions();
        if (window.showToast) {
          window.showToast('새 대화가 생성되었습니다.', 'success');
        }
      }
    } catch (error) {
      console.error('새 대화 생성 실패:', error);
      if (window.showToast) {
        window.showToast(`새 대화 생성 실패: ${error.message}`, 'error');
      }
    }
  };

  // 세션 선택
  const handleSelectSession = (sessionId) => {
    console.log('[Sidebar] 세션 선택:', sessionId);
    setSelectedSessionId(sessionId);
    setCurrentSessionId(sessionId);
    
    // 채팅 탭으로 자동 전환
    if (onTabChange) {
      console.log('[Sidebar] 채팅 탭으로 전환');
      onTabChange('chat');
    }
    
    // 세션 선택 이벤트 발생 (Chat 컴포넌트에서 히스토리 로드)
    window.dispatchEvent(new CustomEvent('sessionSelected', { detail: sessionId }));
  };

  // 세션 삭제
  const handleDeleteSession = async (sessionId, e) => {
    e.stopPropagation(); // 부모 클릭 이벤트 방지

    if (!window.confirm('이 대화를 삭제하시겠습니까?')) {
      return;
    }

    try {
      await apiClient.deleteSession(sessionId);
      // 현재 선택된 세션이 삭제되면 선택 해제
      if (selectedSessionId === sessionId || currentSessionId === sessionId) {
        console.log('[Sidebar] 선택된 세션 삭제됨, 선택 해제');
        setSelectedSessionId(null);
        setCurrentSessionId(null);
        // 채팅 화면 초기화 이벤트 발생 (약간의 지연을 두어 상태 업데이트 보장)
        setTimeout(() => {
          window.dispatchEvent(new CustomEvent('sessionSelected', { detail: null }));
        }, 100);
      }
      // 세션 목록 새로고침
      await loadSessions();
      if (window.showToast) {
        window.showToast('대화가 삭제되었습니다.', 'success');
      }
    } catch (error) {
      console.error('세션 삭제 실패:', error);
      if (window.showToast) {
        window.showToast(`세션 삭제 실패: ${error.message}`, 'error');
      }
    }
  };

  // 시간 포맷팅
  const formatTime = (timestamp) => {
    if (!timestamp) return '';
    const date = new Date(timestamp * 1000);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return '방금 전';
    if (diffMins < 60) return `${diffMins}분 전`;
    if (diffHours < 24) return `${diffHours}시간 전`;
    if (diffDays < 7) return `${diffDays}일 전`;
    return date.toLocaleDateString('ko-KR');
  };

  return (
    <div className="sidebar" style={{
      width: '260px',
      height: '100vh',
      background: '#202123',
      color: '#ececf1',
      display: 'flex',
      flexDirection: 'column',
      borderRight: '1px solid #565869',
    }}>
      {/* 탭 네비게이션 메뉴 */}
      <div style={{ padding: '8px', borderBottom: '1px solid #565869' }}>
        {TABS.map((tab) => {
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => onTabChange && onTabChange(tab.id)}
              style={{
                width: '100%',
                padding: '10px 12px',
                marginBottom: '4px',
                background: isActive ? '#343541' : 'transparent',
                border: 'none',
                borderRadius: '6px',
                color: isActive ? '#ececf1' : '#8e8ea0',
                cursor: 'pointer',
                fontSize: '14px',
                textAlign: 'left',
                display: 'flex',
                alignItems: 'center',
                gap: '10px',
                transition: 'all 0.2s',
                fontWeight: isActive ? '500' : '400',
              }}
              onMouseEnter={(e) => {
                if (!isActive) {
                  e.target.style.background = '#2d2d3a';
                  e.target.style.color = '#ececf1';
                }
              }}
              onMouseLeave={(e) => {
                if (!isActive) {
                  e.target.style.background = 'transparent';
                  e.target.style.color = '#8e8ea0';
                }
              }}
            >
              <span>{tab.icon}</span>
              <span>{tab.label.replace(/^[^\s]+\s/, '')}</span>
            </button>
          );
        })}
      </div>

      {/* 새 대화 버튼 (채팅 탭에서만 표시) */}
      {activeTab === 'chat' && (
        <div style={{ padding: '12px', borderBottom: '1px solid #565869' }}>
          <button
            onClick={handleNewChat}
            style={{
              width: '100%',
              padding: '12px',
              background: 'transparent',
              border: '1px solid #565869',
              borderRadius: '6px',
              color: '#ececf1',
              cursor: 'pointer',
              fontSize: '14px',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              transition: 'background 0.2s',
            }}
            onMouseEnter={(e) => e.target.style.background = '#343541'}
            onMouseLeave={(e) => e.target.style.background = 'transparent'}
          >
            <span>➕</span>
            <span>새 대화</span>
          </button>
        </div>
      )}

      {/* 세션 목록 (채팅 탭에서만 표시) */}
      {activeTab === 'chat' && (
        <div style={{
          flex: 1,
          overflowY: 'auto',
          padding: '8px',
        }}>
        {loading ? (
          <div style={{ padding: '20px', textAlign: 'center', color: '#8e8ea0' }}>
            로딩 중...
          </div>
        ) : sessions.length === 0 ? (
          <div style={{ padding: '20px', textAlign: 'center', color: '#8e8ea0', fontSize: '14px' }}>
            대화가 없습니다.<br />
            새 대화를 시작해보세요.
          </div>
        ) : (
          sessions.map((session) => {
            const isSelected = selectedSessionId === session.session_id;
            const isHovered = hoveredSessionId === session.session_id;

            return (
              <div
                key={session.session_id}
                onClick={() => handleSelectSession(session.session_id)}
                onMouseEnter={() => setHoveredSessionId(session.session_id)}
                onMouseLeave={() => setHoveredSessionId(null)}
                style={{
                  padding: '12px',
                  marginBottom: '4px',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  background: isSelected ? '#343541' : isHovered ? '#2d2d3a' : 'transparent',
                  position: 'relative',
                  transition: 'background 0.2s',
                }}
              >
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  gap: '8px',
                }}>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{
                      fontSize: '14px',
                      fontWeight: isSelected ? '500' : '400',
                      color: '#ececf1',
                      marginBottom: '4px',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                    }}>
                      {session.title || '새 대화'}
                    </div>
                    {session.last_message && (
                      <div style={{
                        fontSize: '12px',
                        color: '#8e8ea0',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                      }}>
                        {session.last_message}
                      </div>
                    )}
                    <div style={{
                      fontSize: '11px',
                      color: '#6e6e80',
                      marginTop: '4px',
                    }}>
                      {formatTime(session.last_accessed)}
                    </div>
                  </div>
                  {(isHovered || isSelected) && (
                    <button
                      onClick={(e) => handleDeleteSession(session.session_id, e)}
                      style={{
                        padding: '4px 8px',
                        background: 'transparent',
                        border: 'none',
                        color: '#8e8ea0',
                        cursor: 'pointer',
                        fontSize: '16px',
                        borderRadius: '4px',
                        transition: 'background 0.2s',
                      }}
                      onMouseEnter={(e) => {
                        e.target.style.background = '#565869';
                        e.target.style.color = '#ececf1';
                      }}
                      onMouseLeave={(e) => {
                        e.target.style.background = 'transparent';
                        e.target.style.color = '#8e8ea0';
                      }}
                      title="삭제"
                    >
                      🗑️
                    </button>
                  )}
                </div>
              </div>
            );
          })
        )}
        </div>
      )}
    </div>
  );
}

