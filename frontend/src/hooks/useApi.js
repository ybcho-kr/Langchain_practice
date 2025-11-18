import { useState, useCallback } from 'react';
import { apiClient } from '../services/api';

/**
 * API 호출 공통 로직을 제공하는 커스텀 훅
 * @param {Function} apiCall - API 호출 함수
 * @returns {{data: any, loading: boolean, error: Error|null, execute: Function, reset: Function}}
 */
export function useApi(apiCall) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const execute = useCallback(
    async (...args) => {
      setLoading(true);
      setError(null);

      try {
        const result = await apiCall(...args);
        setData(result);
        return result;
      } catch (err) {
        setError(err);
        throw err;
      } finally {
        setLoading(false);
      }
    },
    [apiCall]
  );

  const reset = useCallback(() => {
    setData(null);
    setError(null);
    setLoading(false);
  }, []);

  return { data, loading, error, execute, reset };
}

/**
 * 질의 요청 훅
 */
export function useQuery() {
  return useApi(apiClient.query);
}

/**
 * 문서 목록 조회 훅
 */
export function useDocuments() {
  return useApi(apiClient.getDocuments);
}

/**
 * 문서 업로드 훅
 */
export function useUploadDocuments() {
  return useApi(apiClient.uploadDocuments);
}

/**
 * 문서 삭제 훅
 */
export function useDeleteDocument() {
  return useApi(apiClient.deleteDocument);
}

/**
 * 모델 목록 조회 훅
 */
export function useModels() {
  return useApi(apiClient.getModels);
}

/**
 * 설정 조회 훅
 */
export function useConfig() {
  return useApi(apiClient.getConfig);
}

/**
 * Sparse 벡터 Vocabulary 조회 훅
 */
export function useSparseVocabulary() {
  return useApi((params) => apiClient.getSparseVocabulary(params));
}

