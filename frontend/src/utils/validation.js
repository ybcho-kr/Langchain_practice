/**
 * 입력 검증 유틸리티
 */

/**
 * 질문 입력 검증
 * @param {string} message - 검증할 메시지
 * @returns {{valid: boolean, error?: string}}
 */
export function validateQueryInput(message) {
  if (!message || message.trim().length === 0) {
    return { valid: false, error: '메시지를 입력해주세요.' };
  }
  if (message.length > 1000) {
    return { valid: false, error: '메시지는 1000자 이하여야 합니다.' };
  }
  return { valid: true };
}

/**
 * 검색 설정 검증
 * @param {Object} settings - 검색 설정 객체
 * @returns {{valid: boolean, error?: string}}
 */
export function validateSearchSettings(settings) {
  const { useQdrant, useFaiss, useBm25, maxSources, scoreThreshold } = settings;

  // 최소 1개 이상의 검색기 선택 확인
  if (!useQdrant && !useFaiss && !useBm25) {
    return { valid: false, error: '최소 1개 이상의 검색기를 선택해야 합니다.' };
  }

  // Qdrant와 FAISS는 배타적 선택
  if (useQdrant && useFaiss) {
    return { valid: false, error: 'Qdrant와 FAISS는 동시에 선택할 수 없습니다.' };
  }

  // maxSources 검증
  if (maxSources < 1 || maxSources > 20) {
    return { valid: false, error: '최대 소스 수는 1-20 사이여야 합니다.' };
  }

  // scoreThreshold 검증
  if (scoreThreshold < 0 || scoreThreshold > 1) {
    return { valid: false, error: '유사도 임계값은 0-1 사이여야 합니다.' };
  }

  return { valid: true };
}

/**
 * 숫자 범위 검증
 * @param {number} value - 검증할 값
 * @param {number} min - 최소값
 * @param {number} max - 최대값
 * @returns {boolean}
 */
export function validateRange(value, min, max) {
  return value >= min && value <= max;
}

/**
 * 파일 확장자 검증
 * @param {string} filename - 파일명
 * @param {string[]} allowedExtensions - 허용된 확장자 배열
 * @returns {boolean}
 */
export function validateFileExtension(filename, allowedExtensions = ['.md', '.docx', '.txt', '.pdf']) {
  const ext = filename.toLowerCase().substring(filename.lastIndexOf('.'));
  return allowedExtensions.includes(ext);
}

