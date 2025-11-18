/**
 * localStorage 관리 서비스
 */

const SETTINGS_KEY = 'search_settings_v1';

/**
 * 검색 설정 저장
 * @param {Object} settings - 저장할 설정 객체
 * @returns {boolean} 성공 여부
 */
export function saveSearchSettings(settings) {
  try {
    const payload = {
      ...settings,
      saved_at: new Date().toISOString(),
    };
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(payload));
    return true;
  } catch (error) {
    console.error('설정 저장 실패:', error);
    return false;
  }
}

/**
 * 저장된 검색 설정 로드
 * @returns {Object|null} 설정 객체 또는 null
 */
export function loadSearchSettings() {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (!raw) return null;
    return JSON.parse(raw);
  } catch (error) {
    console.error('설정 로드 실패:', error);
    return null;
  }
}

/**
 * 저장된 검색 설정 삭제
 * @returns {boolean} 성공 여부
 */
export function clearSearchSettings() {
  try {
    localStorage.removeItem(SETTINGS_KEY);
    return true;
  } catch (error) {
    console.error('설정 삭제 실패:', error);
    return false;
  }
}

/**
 * localStorage에 값 저장
 * @param {string} key - 키
 * @param {any} value - 값
 */
export function setItem(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch (error) {
    console.error(`localStorage 저장 실패 (${key}):`, error);
  }
}

/**
 * localStorage에서 값 로드
 * @param {string} key - 키
 * @param {any} defaultValue - 기본값
 * @returns {any}
 */
export function getItem(key, defaultValue = null) {
  try {
    const item = localStorage.getItem(key);
    return item ? JSON.parse(item) : defaultValue;
  } catch (error) {
    console.error(`localStorage 로드 실패 (${key}):`, error);
    return defaultValue;
  }
}

/**
 * localStorage에서 값 삭제
 * @param {string} key - 키
 */
export function removeItem(key) {
  try {
    localStorage.removeItem(key);
  } catch (error) {
    console.error(`localStorage 삭제 실패 (${key}):`, error);
  }
}

