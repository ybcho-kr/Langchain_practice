import { useState, useEffect } from 'react';

/**
 * localStorage와 동기화되는 상태 훅
 * @param {string} key - localStorage 키
 * @param {any} initialValue - 초기값
 * @returns {[any, Function]} [값, 설정 함수]
 */
export function useLocalStorage(key, initialValue) {
  // 초기값 로드
  const [storedValue, setStoredValue] = useState(() => {
    try {
      const item = window.localStorage.getItem(key);
      return item ? JSON.parse(item) : initialValue;
    } catch (error) {
      console.error(`localStorage 읽기 오류 (${key}):`, error);
      return initialValue;
    }
  });

  // 값 설정 함수
  const setValue = (value) => {
    try {
      // 함수도 지원
      const valueToStore = value instanceof Function ? value(storedValue) : value;
      setStoredValue(valueToStore);
      window.localStorage.setItem(key, JSON.stringify(valueToStore));
    } catch (error) {
      console.error(`localStorage 쓰기 오류 (${key}):`, error);
    }
  };

  return [storedValue, setValue];
}

/**
 * localStorage 변경 감지 훅
 * @param {string} key - 감지할 키
 * @param {Function} callback - 변경 시 호출할 콜백
 */
export function useLocalStorageListener(key, callback) {
  useEffect(() => {
    const handleStorageChange = (e) => {
      if (e.key === key && e.newValue !== null) {
        try {
          const newValue = JSON.parse(e.newValue);
          callback(newValue);
        } catch (error) {
          console.error('localStorage 변경 감지 오류:', error);
        }
      }
    };

    window.addEventListener('storage', handleStorageChange);
    return () => {
      window.removeEventListener('storage', handleStorageChange);
    };
  }, [key, callback]);
}

