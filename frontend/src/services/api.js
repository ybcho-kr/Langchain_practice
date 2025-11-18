import axios from 'axios';

// 환경 변수가 설정되어 있으면 사용, 없으면 기본값 사용
// 개발 환경: Vite 프록시를 통해 /api로 요청
// 프로덕션: 직접 API 서버로 요청
const getApiBaseUrl = () => {
  // 환경 변수가 명시적으로 설정되어 있으면 사용
  if (import.meta.env.VITE_API_BASE_URL) {
    return import.meta.env.VITE_API_BASE_URL;
  }
  
  // 개발 환경 (Vite dev server)
  if (import.meta.env.DEV) {
    // 프록시를 사용하므로 상대 경로 사용
    return '/api';
  }
  
  // 프로덕션 환경: 현재 호스트의 8000 포트 사용
  const protocol = window.location.protocol;
  const hostname = window.location.hostname;
  return `${protocol}//${hostname}:8000`;
};

const API_BASE_URL = getApiBaseUrl();

// Axios 인스턴스 생성
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// 요청 인터셉터
api.interceptors.request.use(
  (config) => {
    // 요청 전 처리 (필요시)
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// 응답 인터셉터
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    // 에러 처리
    if (error.code === 'ECONNABORTED') {
      error.message = '요청 시간이 초과되었습니다.';
    } else if (error.response) {
      // 서버 응답이 있는 경우
      error.message = error.response.data?.message || `HTTP ${error.response.status}`;
    } else if (error.request) {
      // 요청은 보냈지만 응답을 받지 못한 경우
      error.message = '서버에 연결할 수 없습니다.';
    }
    return Promise.reject(error);
  }
);

// API 클라이언트
export const apiClient = {
  /**
   * 질의 요청
   * @param {Object} data - 질의 요청 데이터
   * @returns {Promise}
   */
  async query(data) {
    const response = await api.post('/query', data);
    return response.data;
  },

  /**
   * 문서 목록 조회
   * @returns {Promise}
   */
  async getDocuments() {
    const response = await api.get('/documents');
    return response.data;
  },

  /**
   * 문서 업로드
   * @param {FormData} formData - 파일 및 옵션 데이터
   * @returns {Promise}
   */
  async uploadDocuments(formData) {
    const response = await api.post('/upload-documents', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 300000, // 5분
    });
    return response.data;
  },

  /**
   * 문서 삭제
   * @param {string} sourceFile - 문서 파일 경로
   * @returns {Promise}
   */
  async deleteDocument(sourceFile) {
    const response = await api.delete(`/documents/${encodeURIComponent(sourceFile)}`);
    return response.data;
  },

  /**
   * 문서 청크 조회
   * @param {string} documentId - 문서 ID
   * @returns {Promise}
   */
  async getDocumentChunks(documentId) {
    const response = await api.get(`/documents/${encodeURIComponent(documentId)}/chunks`);
    return response.data;
  },

  /**
   * 설정 조회
   * @returns {Promise}
   */
  async getConfig() {
    const response = await api.get('/config');
    return response.data;
  },

  /**
   * 사용 가능한 모델 목록 조회
   * @returns {Promise}
   */
  async getModels() {
    const response = await api.get('/models');
    return response.data;
  },

  /**
   * 헬스 체크
   * @returns {Promise}
   */
  async healthCheck() {
    const response = await api.get('/health');
    return response.data;
  },

  /**
   * Sparse 벡터 Vocabulary 조회
   * @param {Object} params - 쿼리 파라미터
   * @param {number} params.limit - 반환할 vocabulary 항목 수
   * @param {string} params.search_token - 특정 토큰 검색
   * @returns {Promise}
   */
  async getSparseVocabulary(params = {}) {
    const response = await api.get('/sparse-vocabulary', { params });
    return response.data;
  },

  /**
   * Agentic RAG 실행 (LangGraph)
   * @param {Object} data - Agentic 실행 요청 데이터
   * @param {string} data.question - 사용자 질문
   * @param {string} [data.session_id] - 세션 ID (선택적)
   * @param {string} [data.graph] - 그래프 타입 ('basic' | 'advanced', 기본값: 'basic')
   * @param {Object} [data.parameters] - 추가 파라미터
   * @returns {Promise}
   */
  async agenticExecute(data) {
    const response = await api.post('/agentic/execute', data);
    return response.data;
  },

  /**
   * 세션 생성
   * @param {string} [sessionId] - 세션 ID (선택적, 없으면 자동 생성)
   * @returns {Promise}
   */
  async createSession(sessionId = null) {
    const params = sessionId ? { session_id: sessionId } : {};
    const response = await api.post('/sessions', {}, { params });
    return response.data;
  },

  /**
   * 세션 히스토리 조회
   * @param {string} sessionId - 세션 ID
   * @param {number} [limit] - 반환할 최대 메시지 수
   * @returns {Promise}
   */
  async getSessionHistory(sessionId, limit = null) {
    const params = limit ? { limit } : {};
    const response = await api.get(`/sessions/${encodeURIComponent(sessionId)}/history`, { params });
    return response.data;
  },

  /**
   * 세션 삭제
   * @param {string} sessionId - 세션 ID
   * @returns {Promise}
   */
  async deleteSession(sessionId) {
    const response = await api.delete(`/sessions/${encodeURIComponent(sessionId)}`);
    return response.data;
  },

  /**
   * 세션 목록 조회
   * @returns {Promise}
   */
  async getSessions() {
    const response = await api.get('/sessions');
    return response.data;
  },

  /**
   * 세션 통계 조회
   * @returns {Promise}
   */
  async getSessionStats() {
    const response = await api.get('/sessions/stats');
    return response.data;
  },

  /**
   * LangGraph 진행 상황 조회 (폴링용, 향후 백엔드 API 추가 시 사용)
   * @param {string} graphRunId - 그래프 실행 ID
   * @returns {Promise}
   */
  async getGraphProgress(graphRunId) {
    // 현재는 백엔드에 진행 상황 조회 API가 없으므로 null 반환
    // 향후 백엔드에 API가 추가되면 구현
    return null;
  },
};

export default apiClient;

