import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';

/**
 * 전역 상태 관리 스토어
 */
export const useAppStore = create(
  persist(
    (set, get) => ({
      // Evidence 상태
      evidence: null,

      // 모델 설정
      model: 'gemma3:4b',

      // API 모드 선택 ('query' | 'agentic')
      apiMode: 'query',

      // 현재 세션 ID
      currentSessionId: null,

      // 진행 상황 상태
      progress: {
        stage: null, // 'analyzing', 'planning', 'searching', 'evaluating', 'generating', 'reranking', 'rewriting'
        progress: 0, // 0-100
        message: '',
        graphRunId: null,
        reretrieveCount: 0,
      },

      // 세션 목록
      sessions: [],

      // 선택된 세션 ID
      selectedSessionId: null,

      // 세션 제목 캐시 (session_id -> title)
      sessionTitles: {},

      // 검색 설정
      settings: {
        maxSources: 5,
        scoreThreshold: 0.85,
        useQdrant: true,
        useFaiss: false,
        useBm25: true,
        useReranker: true,
        rerankerAlpha: 0.7,
        rerankerTopK: 3,
        weights: {
          qdrant: 0.7,
          faiss: 0.0,
          bm25: 0.3,
        },
        sliderWeights: {
          qdrant: 0.7,
          faiss: 0.0,
          bm25: 0.3,
        },
        denseWeight: 0.7,
        sparseWeight: 0.3,
      },

      // 문서 목록
      documents: [],

      // Actions
      setEvidence: (evidence) => set({ evidence }),

      updateModel: (model) => set({ model }),

      updateSettings: (newSettings) =>
        set((state) => ({
          settings: {
            ...state.settings,
            ...newSettings,
          },
        })),

      setDocuments: (documents) => set({ documents }),

      // API 모드 업데이트
      setApiMode: (mode) => set({ apiMode: mode }),

      // 세션 ID 설정
      setCurrentSessionId: (sessionId) => set({ currentSessionId: sessionId }),

      // 진행 상황 업데이트
      setProgress: (progress) => set((state) => ({
        progress: {
          ...state.progress,
          ...progress,
        },
      })),

      // 진행 상황 초기화
      clearProgress: () => set({
        progress: {
          stage: null,
          progress: 0,
          message: '',
          graphRunId: null,
          reretrieveCount: 0,
        },
      }),

      // 세션 목록 업데이트
      setSessions: (sessions) => {
        // 세션 제목 캐시도 업데이트
        const titles = {};
        if (Array.isArray(sessions)) {
          sessions.forEach((session) => {
            if (session.session_id) {
              titles[session.session_id] = session.title || '새 대화';
            }
          });
        }
        set({ sessions, sessionTitles: titles });
      },

      // 선택된 세션 ID 설정
      setSelectedSessionId: (sessionId) => set({ selectedSessionId: sessionId }),

      // 세션 제목 업데이트
      updateSessionTitle: (sessionId, title) => set((state) => ({
        sessionTitles: {
          ...state.sessionTitles,
          [sessionId]: title,
        },
      })),

      // 가중치 정규화
      normalizeWeights: () => {
        const { settings } = get();
        const { useQdrant, useFaiss, useBm25, sliderWeights } = settings;
        const rawQ = parseFloat(sliderWeights.qdrant || 0);
        const rawF = parseFloat(sliderWeights.faiss || 0);
        const rawB = parseFloat(sliderWeights.bm25 || 0);

        let weights = { qdrant: 0, faiss: 0, bm25: 0 };

        if (useQdrant && !useFaiss) {
          if (useBm25) {
            const v = Math.max(0, rawQ);
            const b = Math.max(0, rawB);
            const s = v + b > 0 ? v + b : 1;
            weights.qdrant = v / s;
            weights.bm25 = b / s;
          } else {
            weights.qdrant = 1.0;
          }
        } else if (useFaiss && !useQdrant) {
          if (useBm25) {
            const v = Math.max(0, rawF);
            const b = Math.max(0, rawB);
            const s = v + b > 0 ? v + b : 1;
            weights.faiss = v / s;
            weights.bm25 = b / s;
          } else {
            weights.faiss = 1.0;
          }
        } else if (useBm25 && !useQdrant && !useFaiss) {
          weights.bm25 = 1.0;
        } else {
          // 안전장치
          if (useQdrant) weights.qdrant = 1.0;
          else if (useFaiss) weights.faiss = 1.0;
          else weights.bm25 = 1.0;
        }

        set((state) => ({
          settings: {
            ...state.settings,
            weights,
          },
        }));
      },
    }),
    {
      name: 'rag-app-storage',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        model: state.model,
        settings: state.settings,
        apiMode: state.apiMode,
        currentSessionId: state.currentSessionId,
        selectedSessionId: state.selectedSessionId,
      }),
    }
  )
);

