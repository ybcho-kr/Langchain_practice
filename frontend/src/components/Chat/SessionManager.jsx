import { useState, useEffect } from 'react';
import { useAppStore } from '../../stores/appStore';
import { apiClient } from '../../services/api';
import Modal from '../common/Modal';

/**
 * 세션 관리 컴포넌트
 */
export default function SessionManager() {
  const { currentSessionId, setCurrentSessionId, sessions, setSessions } = useAppStore();
  const [historyModalOpen, setHistoryModalOpen] = useState(false);
  const [history, setHistory] = useState([]);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [stats, setStats] = useState(null);

  // 세션 생성
  const handleCreateSession = async () => {
    try {
      const response = await apiClient.createSession();
      if (response && response.session_id) {
        setCurrentSessionId(response.session_id);
        if (window.showToast) {
          window.showToast('새 세션이 생성되었습니다.', 'success');
        }
        // 세션 목록 새로고침
        loadSessionStats();
      }
    } catch (error) {
      console.error('세션 생성 실패:', error);
      if (window.showToast) {
        window.showToast(`세션 생성 실패: ${error.message}`, 'error');
      }
    }
  };

  // 세션 삭제
  const handleDeleteSession = async () => {
    if (!currentSessionId) {
      return;
    }

    if (!window.confirm('현재 세션을 삭제하시겠습니까?')) {
      return;
    }

    try {
      await apiClient.deleteSession(currentSessionId);
      setCurrentSessionId(null);
      if (window.showToast) {
        window.showToast('세션이 삭제되었습니다.', 'success');
      }
      // 세션 목록 새로고침
      loadSessionStats();
    } catch (error) {
      console.error('세션 삭제 실패:', error);
      if (window.showToast) {
        window.showToast(`세션 삭제 실패: ${error.message}`, 'error');
      }
    }
  };

  // 세션 히스토리 조회
  const loadHistory = async () => {
    if (!currentSessionId) {
      setHistory([]);
      return;
    }

    setLoadingHistory(true);
    try {
      const response = await apiClient.getSessionHistory(currentSessionId);
      if (response && response.history) {
        setHistory(response.history);
      } else {
        setHistory([]);
      }
    } catch (error) {
      console.error('히스토리 조회 실패:', error);
      setHistory([]);
      if (window.showToast) {
        window.showToast(`히스토리 조회 실패: ${error.message}`, 'error');
      }
    } finally {
      setLoadingHistory(false);
    }
  };

  // 세션 통계 조회
  const loadSessionStats = async () => {
    try {
      const response = await apiClient.getSessionStats();
      if (response) {
        setStats(response);
      }
    } catch (error) {
      console.error('세션 통계 조회 실패:', error);
    }
  };

  // 히스토리 모달 열기
  const handleOpenHistory = () => {
    setHistoryModalOpen(true);
    loadHistory();
  };

  // 컴포넌트 마운트 시 통계 로드
  useEffect(() => {
    loadSessionStats();
  }, []);

  return (
    <div className="session-manager" style={{
      display: 'flex',
      gap: '8px',
      alignItems: 'center',
      padding: '8px',
      background: '#f8f9fa',
      borderRadius: '6px',
      marginBottom: '10px',
    }}>
      <div style={{ flex: 1, fontSize: '0.9em' }}>
        {currentSessionId ? (
          <span style={{ color: '#28a745' }}>
            세션: <code style={{ fontSize: '0.85em' }}>{currentSessionId.substring(0, 8)}...</code>
          </span>
        ) : (
          <span style={{ color: '#6c757d' }}>세션 없음</span>
        )}
        {stats && (
          <span style={{ marginLeft: '12px', color: '#6c757d', fontSize: '0.85em' }}>
            (전체: {stats.total_sessions || 0}개)
          </span>
        )}
      </div>
      <button
        className="btn btn-sm"
        onClick={handleCreateSession}
        style={{
          padding: '4px 12px',
          fontSize: '0.85em',
          background: '#28a745',
          color: 'white',
          border: 'none',
          borderRadius: '4px',
          cursor: 'pointer',
        }}
        title="새 세션 생성"
      >
        ➕ 새 세션
      </button>
      {currentSessionId && (
        <>
          <button
            className="btn btn-sm"
            onClick={handleOpenHistory}
            style={{
              padding: '4px 12px',
              fontSize: '0.85em',
              background: '#17a2b8',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
            }}
            title="세션 히스토리 보기"
          >
            📜 히스토리
          </button>
          <button
            className="btn btn-sm"
            onClick={handleDeleteSession}
            style={{
              padding: '4px 12px',
              fontSize: '0.85em',
              background: '#dc3545',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
            }}
            title="세션 삭제"
          >
            🗑️ 삭제
          </button>
        </>
      )}

      {/* 히스토리 모달 */}
      <Modal
        isOpen={historyModalOpen}
        onClose={() => setHistoryModalOpen(false)}
        title="세션 히스토리"
      >
        {loadingHistory ? (
          <div className="loading">
            <div className="spinner"></div>
            <span>히스토리를 불러오는 중...</span>
          </div>
        ) : history.length === 0 ? (
          <div style={{ padding: '20px', textAlign: 'center', color: '#6c757d' }}>
            히스토리가 없습니다.
          </div>
        ) : (
          <div style={{ maxHeight: '400px', overflowY: 'auto' }}>
            {history.map((msg, index) => (
              <div
                key={index}
                style={{
                  padding: '12px',
                  marginBottom: '8px',
                  background: msg.role === 'user' ? '#e7f3ff' : '#f8f9fa',
                  borderRadius: '6px',
                  borderLeft: `4px solid ${msg.role === 'user' ? '#007bff' : '#28a745'}`,
                }}
              >
                <div style={{ fontWeight: '500', marginBottom: '4px', fontSize: '0.9em' }}>
                  {msg.role === 'user' ? '👤 사용자' : '🤖 어시스턴트'}
                  {msg.timestamp && (
                    <span style={{ marginLeft: '8px', fontSize: '0.85em', color: '#6c757d' }}>
                      {new Date(msg.timestamp * 1000).toLocaleString('ko-KR')}
                    </span>
                  )}
                </div>
                <div style={{ fontSize: '0.9em', whiteSpace: 'pre-wrap' }}>
                  {msg.content}
                </div>
                {msg.search_results && msg.search_results.length > 0 && (
                  <div style={{ marginTop: '8px', fontSize: '0.85em', color: '#6c757d' }}>
                    참조 문서: {msg.search_results.length}개
                  </div>
                )}
                {msg.confidence !== undefined && (
                  <div style={{ marginTop: '4px', fontSize: '0.85em', color: '#6c757d' }}>
                    신뢰도: {(msg.confidence * 100).toFixed(1)}%
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </Modal>
    </div>
  );
}

