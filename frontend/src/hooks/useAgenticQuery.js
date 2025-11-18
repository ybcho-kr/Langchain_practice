import { useState, useCallback } from 'react';
import { apiClient } from '../services/api';

/**
 * Agentic API 호출 훅
 * @returns {{data: any, loading: boolean, error: Error|null, execute: Function, reset: Function}}
 */
export function useAgenticQuery() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const execute = useCallback(
    async (question, sessionId = null, graph = 'basic', parameters = {}) => {
      setLoading(true);
      setError(null);

      try {
        const result = await apiClient.agenticExecute({
          question,
          session_id: sessionId,
          graph,
          parameters,
        });
        setData(result);
        return result;
      } catch (err) {
        setError(err);
        throw err;
      } finally {
        setLoading(false);
      }
    },
    []
  );

  const reset = useCallback(() => {
    setData(null);
    setError(null);
    setLoading(false);
  }, []);

  return { data, loading, error, execute, reset };
}

