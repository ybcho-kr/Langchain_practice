import { useState, useCallback, useEffect } from 'react';
import { apiClient } from '../services/api';
import { useAppStore } from '../stores/appStore';

/**
 * 세션 관리 훅
 * @returns {{sessionId: string|null, createSession: Function, deleteSession: Function, loadHistory: Function, history: Array, loading: boolean}}
 */
export function useSession() {
  const { currentSessionId, setCurrentSessionId } = useAppStore();
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);

  // 세션 생성
  const createSession = useCallback(async () => {
    setLoading(true);
    try {
      const response = await apiClient.createSession();
      if (response && response.session_id) {
        setCurrentSessionId(response.session_id);
        return response.session_id;
      }
      return null;
    } catch (error) {
      console.error('세션 생성 실패:', error);
      throw error;
    } finally {
      setLoading(false);
    }
  }, [setCurrentSessionId]);

  // 세션 삭제
  const deleteSession = useCallback(async (sessionId) => {
    setLoading(true);
    try {
      await apiClient.deleteSession(sessionId);
      if (sessionId === currentSessionId) {
        setCurrentSessionId(null);
      }
    } catch (error) {
      console.error('세션 삭제 실패:', error);
      throw error;
    } finally {
      setLoading(false);
    }
  }, [currentSessionId, setCurrentSessionId]);

  // 히스토리 로드
  const loadHistory = useCallback(async (sessionId = currentSessionId) => {
    if (!sessionId) {
      setHistory([]);
      return;
    }

    setLoading(true);
    try {
      const response = await apiClient.getSessionHistory(sessionId);
      if (response && response.history) {
        setHistory(response.history);
      } else {
        setHistory([]);
      }
    } catch (error) {
      console.error('히스토리 로드 실패:', error);
      setHistory([]);
    } finally {
      setLoading(false);
    }
  }, [currentSessionId]);

  // 현재 세션 ID가 변경되면 히스토리 자동 로드
  useEffect(() => {
    if (currentSessionId) {
      loadHistory(currentSessionId);
    } else {
      setHistory([]);
    }
  }, [currentSessionId, loadHistory]);

  return {
    sessionId: currentSessionId,
    createSession,
    deleteSession,
    loadHistory,
    history,
    loading,
  };
}

