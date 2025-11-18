import { useCallback } from 'react';
import { useAppStore } from '../stores/appStore';

/**
 * 진행 상황 관리 훅
 * @returns {{setStage: Function, updateProgress: Function, clearProgress: Function}}
 */
export function useProgress() {
  const { setProgress, clearProgress } = useAppStore();

  // 단계 설정
  const setStage = useCallback((stage, message = '', progress = 0) => {
    setProgress({
      stage,
      message,
      progress,
    });
  }, [setProgress]);

  // 진행률 업데이트
  const updateProgress = useCallback((progress, message = null) => {
    setProgress((prev) => ({
      ...prev,
      progress: Math.min(100, Math.max(0, progress)),
      message: message || prev.message,
    }));
  }, [setProgress]);

  // 진행 상황 초기화
  const resetProgress = useCallback(() => {
    clearProgress();
  }, [clearProgress]);

  return {
    setStage,
    updateProgress,
    clearProgress: resetProgress,
  };
}

