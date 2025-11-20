import { useState, useEffect, useRef, useCallback } from 'react';
import { useAppStore } from '../../stores/appStore';
import { useQuery } from '../../hooks/useApi';
import { useAgenticQuery } from '../../hooks/useAgenticQuery';
import { useProgress } from '../../hooks/useProgress';
import { validateQueryInput, validateSearchSettings } from '../../utils/validation';
import { getItem, setItem, removeItem } from '../../services/storage';
import ChatMessage from './ChatMessage';
import ChatInput from './ChatInput';
import ProgressIndicator from './ProgressIndicator';
import '../../styles/components/chat.css';

const CHAT_MESSAGES_KEY = 'chat_messages_v1';

/**
 * 채팅 컴포넌트
 */
export default function Chat() {
  const { 
    model, 
    settings, 
    evidence, 
    setEvidence, 
    normalizeWeights,
    apiMode,
    setApiMode,
    currentSessionId,
    setCurrentSessionId,
    selectedSessionId,
    setSelectedSessionId,
    progress,
    setProgress,
    sessions,
  } = useAppStore();
  
  const { setStage, clearProgress } = useProgress();
  const { execute: sendQuery, loading: queryLoading } = useQuery();
  const { execute: sendAgenticQuery, loading: agenticLoading } = useAgenticQuery();
  const loading = queryLoading || agenticLoading;
  
  // 초기 메시지 생성 (세션이 활성화된 경우 localStorage 무시)
  const getInitialMessages = useCallback(() => {
    // 세션이 활성화된 경우 localStorage 메시지 무시
    if (selectedSessionId || currentSessionId) {
      return [
        {
          id: Date.now(),
          type: 'assistant',
          content: `안녕하세요! 전기설비 진단 RAG 시스템입니다.<br>DGA, 부분방전(PD), 변압기 진단 등에 대해 질문해주세요.<br><strong>현재 모델:</strong> ${model}`,
          sources: [],
          metadata: null,
        },
      ];
    }
    
    // 세션이 없는 경우에만 localStorage에서 복원
    const savedMessages = getItem(CHAT_MESSAGES_KEY, null);
    if (savedMessages && Array.isArray(savedMessages) && savedMessages.length > 0) {
      return savedMessages;
    }
    
    // 세션이 없고 localStorage에도 메시지가 없는 경우 안내 메시지
    return [
      {
        id: Date.now(),
        type: 'assistant',
        content: `안녕하세요! 전기설비 진단 RAG 시스템입니다.<br><br>💡 <strong>새 대화를 시작하려면</strong><br>사이드바의 "➕ 새 대화" 버튼을 클릭하거나, 질문을 입력하면 자동으로 세션이 생성됩니다.<br><br>DGA, 부분방전(PD), 변압기 진단 등에 대해 질문해주세요.<br><strong>현재 모델:</strong> ${model}`,
        sources: [],
        metadata: null,
      },
    ];
  }, [model, selectedSessionId, currentSessionId]);

  const [messages, setMessages] = useState(getInitialMessages);

  const messagesContainerRef = useRef(null);

  // 채팅 기록 초기화
  const handleClearChat = useCallback(() => {
    // localStorage에서 채팅 메시지 삭제
    removeItem(CHAT_MESSAGES_KEY);
    
    // 메시지를 초기 상태로 되돌리기
    const initialMessage = [
      {
        id: Date.now(),
        type: 'assistant',
        content: `안녕하세요! 전기설비 진단 RAG 시스템입니다.<br>DGA, 부분방전(PD), 변압기 진단 등에 대해 질문해주세요.<br><strong>현재 모델:</strong> ${model}`,
        sources: [],
        metadata: null,
      },
    ];
    setMessages(initialMessage);
    
    // 근거 정보도 초기화
    setEvidence(null);
    
    if (window.showToast) {
      window.showToast('채팅 기록이 초기화되었습니다.', 'success');
    }
  }, [model, setEvidence]);

  // 메시지 변경 시 localStorage에 저장 (세션이 활성화된 경우 저장하지 않음)
  useEffect(() => {
    // 세션이 활성화된 경우 localStorage에 저장하지 않음 (세션 히스토리와 충돌 방지)
    if (selectedSessionId || currentSessionId) {
      return;
    }
    
    // 세션이 없는 경우에만 localStorage에 저장
    if (messages && messages.length > 0) {
      setItem(CHAT_MESSAGES_KEY, messages);
    }
  }, [messages, selectedSessionId, currentSessionId]);

  // 메시지 추가 시 스크롤
  useEffect(() => {
    if (messagesContainerRef.current) {
      messagesContainerRef.current.scrollTop = messagesContainerRef.current.scrollHeight;
    }
  }, [messages]);

  // 모델 변경 시 시스템 메시지 업데이트
  useEffect(() => {
    if (messages.length > 0 && messages[0].type === 'assistant') {
      setMessages((prev) => [
        {
          ...prev[0],
          content: `안녕하세요! 전기설비 진단 RAG 시스템입니다.<br>DGA, 부분방전(PD), 변압기 진단 등에 대해 질문해주세요.<br><strong>현재 모델:</strong> ${model}`,
        },
        ...prev.slice(1),
      ]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model]);

  // 세션 선택 시 히스토리 로드
  useEffect(() => {
    const handleSessionSelected = async (e) => {
      const sessionId = e.detail;
      console.log('[Chat] 세션 선택 이벤트 수신:', sessionId);
      
      if (!sessionId) {
        // 세션이 없으면 초기 메시지만 표시
        console.log('[Chat] 세션 ID가 없음, 초기 메시지 표시');
        setCurrentSessionId(null);
        setSelectedSessionId(null);
        // 초기 메시지 강제 설정 (의존성 문제 해결)
        const initialMessage = [
          {
            id: Date.now(),
            type: 'assistant',
            content: `안녕하세요! 전기설비 진단 RAG 시스템입니다.<br><br>💡 <strong>새 대화를 시작하려면</strong><br>사이드바의 "➕ 새 대화" 버튼을 클릭하거나, 질문을 입력하면 자동으로 세션이 생성됩니다.<br><br>DGA, 부분방전(PD), 변압기 진단 등에 대해 질문해주세요.<br><strong>현재 모델:</strong> ${model}`,
            sources: [],
            metadata: null,
          },
        ];
        setMessages(initialMessage);
        return;
      }

      console.log('[Chat] 세션 히스토리 로드 시작:', sessionId);
      setCurrentSessionId(sessionId);
      setSelectedSessionId(sessionId);

      try {
        const { apiClient } = await import('../../services/api');
        const response = await apiClient.getSessionHistory(sessionId);
        console.log('[Chat] 세션 히스토리 API 응답:', response);
        
        if (response && response.history && Array.isArray(response.history)) {
          // 히스토리를 메시지 형식으로 변환
          const historyMessages = response.history.map((msg, index) => ({
            id: `session-${sessionId}-${index}-${Date.now()}`,
            type: msg.role === 'user' ? 'user' : 'assistant',
            content: msg.content,
            sources: msg.search_results || [],
            metadata: {
              confidence: msg.confidence,
              processing_time: msg.processing_time,
              model_used: msg.model_used,
              timestamp: msg.timestamp,
            },
          }));

          console.log('[Chat] 변환된 히스토리 메시지:', historyMessages.length, '개');

          // 히스토리가 있으면 표시, 없으면 초기 메시지
          if (historyMessages.length > 0) {
            console.log('[Chat] 히스토리 메시지 표시');
            setMessages(historyMessages);
          } else {
            console.log('[Chat] 히스토리가 비어있음, 초기 메시지 표시');
            const initialMessage = [
              {
                id: Date.now(),
                type: 'assistant',
                content: `안녕하세요! 전기설비 진단 RAG 시스템입니다.<br>DGA, 부분방전(PD), 변압기 진단 등에 대해 질문해주세요.<br><strong>현재 모델:</strong> ${model}`,
                sources: [],
                metadata: null,
              },
            ];
            setMessages(initialMessage);
          }
        } else {
          console.warn('[Chat] 히스토리 응답 형식이 올바르지 않음:', response);
          const initialMessage = [
            {
              id: Date.now(),
              type: 'assistant',
              content: `안녕하세요! 전기설비 진단 RAG 시스템입니다.<br>DGA, 부분방전(PD), 변압기 진단 등에 대해 질문해주세요.<br><strong>현재 모델:</strong> ${model}`,
              sources: [],
              metadata: null,
            },
          ];
          setMessages(initialMessage);
        }
      } catch (error) {
        console.error('[Chat] 세션 히스토리 로드 실패:', error);
        const initialMessage = [
          {
            id: Date.now(),
            type: 'assistant',
            content: `안녕하세요! 전기설비 진단 RAG 시스템입니다.<br>DGA, 부분방전(PD), 변압기 진단 등에 대해 질문해주세요.<br><strong>현재 모델:</strong> ${model}`,
            sources: [],
            metadata: null,
          },
        ];
        setMessages(initialMessage);
      }
    };

    window.addEventListener('sessionSelected', handleSessionSelected);
    return () => {
      window.removeEventListener('sessionSelected', handleSessionSelected);
    };
  }, [model, setCurrentSessionId, setSelectedSessionId]);

  // selectedSessionId 변경 시 히스토리 로드
  useEffect(() => {
    if (selectedSessionId) {
      console.log('[Chat] selectedSessionId 변경됨:', selectedSessionId);
      // 세션 선택 이벤트 발생 (항상 로드)
      window.dispatchEvent(new CustomEvent('sessionSelected', { detail: selectedSessionId }));
    } else {
      console.log('[Chat] selectedSessionId가 null, 초기 메시지로 리셋');
      // 세션이 없으면 초기 메시지로 리셋
      setMessages(getInitialMessages());
      setCurrentSessionId(null);
    }
  }, [selectedSessionId, getInitialMessages, setCurrentSessionId]);

  // 세션 자동 생성 (필요 시)
  const ensureSession = useCallback(async () => {
    // 세션이 없으면 자동 생성 (기본 RAG와 Agentic 모두)
    if (!currentSessionId && !selectedSessionId) {
      try {
        console.log('[Chat] 세션이 없음, 자동 생성 시작');
        const { apiClient } = await import('../../services/api');
        const response = await apiClient.createSession();
        if (response && response.session_id) {
          console.log('[Chat] 세션 자동 생성 완료:', response.session_id);
          setCurrentSessionId(response.session_id);
          setSelectedSessionId(response.session_id);
          // 사이드바 세션 목록 새로고침을 위해 이벤트 발생
          window.dispatchEvent(new CustomEvent('sessionCreated', { detail: response.session_id }));
          return response.session_id;
        }
      } catch (error) {
        console.error('[Chat] 세션 생성 실패:', error);
        if (window.showToast) {
          window.showToast('세션 생성에 실패했습니다. 다시 시도해주세요.', 'error');
        }
      }
    }
    return currentSessionId || selectedSessionId;
  }, [apiMode, currentSessionId, selectedSessionId, setCurrentSessionId, setSelectedSessionId]);

  // 기본 RAG 진행 상황 시뮬레이션
  const simulateBasicRAGProgress = useCallback(() => {
    setStage('searching', '검색 중...', 20);
    setTimeout(() => setStage('searching', '검색 중...', 40), 300);
    setTimeout(() => {
      if (settings.useReranker) {
        setStage('reranking', '리랭킹 중...', 60);
      } else {
        setStage('generating', '답변 생성 중...', 60);
      }
    }, 600);
    setTimeout(() => setStage('generating', '답변 생성 중...', 80), 900);
  }, [settings.useReranker, setStage]);

  // LangGraph 진행 상황 시뮬레이션
  const simulateAgenticProgress = useCallback((graphType = 'basic') => {
    if (graphType === 'basic') {
      // 기본 그래프: analyze → search → generate
      setStage('analyzing', '질문 분석 중...', 15);
      setTimeout(() => setStage('searching', '검색 실행 중...', 40), 400);
      setTimeout(() => setStage('generating', '답변 생성 중...', 70), 800);
    } else {
      // 고급 그래프: analyze → plan → search → evaluate → (rewrite → search → evaluate)* → generate
      setStage('analyzing', '질문 분석 중...', 10);
      setTimeout(() => setStage('planning', '계획 수립 중...', 20), 300);
      setTimeout(() => setStage('searching', '검색 실행 중...', 40), 600);
      setTimeout(() => setStage('evaluating', '결과 평가 중...', 60), 900);
      setTimeout(() => setStage('generating', '답변 생성 중...', 80), 1200);
    }
  }, [setStage]);

  const handleSend = useCallback(async (message) => {
    // 입력 검증
    const validation = validateQueryInput(message);
    if (!validation.valid) {
      if (window.showToast) {
        window.showToast(validation.error, 'warning');
      }
      return;
    }

    // 검색 설정 검증 (기본 RAG만)
    if (apiMode === 'query') {
      const settingsValidation = validateSearchSettings(settings);
      if (!settingsValidation.valid) {
        if (window.showToast) {
          window.showToast(settingsValidation.error, 'warning');
        }
        return;
      }
    }

    // 사용자 메시지 추가
    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: message,
      sources: [],
      metadata: null,
    };
    setMessages((prev) => [...prev, userMessage]);

    // 진행 상황 초기화
    clearProgress();

    try {
      let response;

      if (apiMode === 'agentic') {
        // Agentic API 호출
        const sessionId = await ensureSession();
        
        // 진행 상황 시뮬레이션 시작
        simulateAgenticProgress('basic'); // 기본 그래프 사용
        
        // 가중치 정규화
        normalizeWeights();

        response = await sendAgenticQuery(
          message,
          sessionId,
          'basic', // 기본 그래프 사용
          {
            max_sources: settings.maxSources,
            score_threshold: settings.scoreThreshold,
            use_reranker: settings.useReranker,
          }
        );

        // 진행 상황 업데이트
        setProgress({
          stage: 'generating',
          progress: 90,
          message: '답변 생성 완료',
          graphRunId: response.graph_run_id,
        });
      } else {
        // 기본 RAG API 호출
        // 세션이 없으면 자동 생성
        let sessionId = currentSessionId || selectedSessionId;
        if (!sessionId) {
          sessionId = await ensureSession();
        }
        
        // 가중치 정규화
        normalizeWeights();

        // 진행 상황 시뮬레이션 시작
        simulateBasicRAGProgress();

        // 요청 데이터 준비
        const requestData = {
          question: message,
          max_sources: settings.maxSources,
          score_threshold: settings.scoreThreshold,
          model: model,
          use_qdrant: settings.useQdrant,
          use_faiss: settings.useFaiss,
          use_bm25: settings.useBm25,
          use_reranker: settings.useReranker,
          reranker_alpha: settings.rerankerAlpha,
          reranker_top_k: settings.rerankerTopK,
          weights: settings.weights,
          dense_weight: settings.denseWeight !== undefined ? settings.denseWeight : null,
          sparse_weight: settings.sparseWeight !== undefined ? settings.sparseWeight : null,
          session_id: sessionId,
        };

        response = await sendQuery(requestData);

        // 진행 상황 업데이트
        setProgress({
          stage: 'generating',
          progress: 100,
          message: '완료',
        });
      }

      // 경고 메시지 처리
      if (response.warnings && response.warnings.length > 0) {
        response.warnings.forEach((warning) => {
          if (window.showToast) {
            window.showToast(warning, 'warning', 10000);
          }
        });
      }

      // 어시스턴트 응답 추가
      const assistantMessage = {
        id: Date.now() + 1,
        type: 'assistant',
        content: response.answer,
        sources: response.sources || [],
        metadata: {
          confidence: response.confidence,
          processing_time: response.processing_time,
          model_used: response.model_used,
          max_sources: settings.maxSources,
          score_threshold: settings.scoreThreshold,
          graph_run_id: response.graph_run_id,
          api_mode: apiMode,
        },
      };
      setMessages((prev) => [...prev, assistantMessage]);

      // 근거 정보 저장
      setEvidence({
        question: message,
        answer: response.answer,
        sources: response.sources || [],
        confidence: response.confidence,
        processing_time: response.processing_time,
        model_used: response.model_used,
        graph_run_id: response.graph_run_id,
      });

      // 세션에 메시지 저장은 백엔드에서 자동으로 처리됨
      // /query 또는 /agentic/execute API 호출 시 session_id가 포함되면 자동 저장

      // 진행 상황 완료 후 초기화
      setTimeout(() => clearProgress(), 1000);
    } catch (error) {
      console.error('Error:', error);
      const errorMessage =
        error.message || '오류가 발생했습니다. 다시 시도해주세요.';

      // 에러 메시지 추가
      const errorMsg = {
        id: Date.now() + 1,
        type: 'assistant',
        content: errorMessage,
        sources: [],
        metadata: null,
      };
      setMessages((prev) => [...prev, errorMsg]);

      // 진행 상황 초기화
      clearProgress();

      if (window.showToast) {
        window.showToast(errorMessage, 'error');
      }
    }
  }, [
    model, 
    settings, 
    normalizeWeights, 
    sendQuery, 
    sendAgenticQuery,
    apiMode,
    currentSessionId,
    ensureSession,
    simulateBasicRAGProgress,
    simulateAgenticProgress,
    setProgress,
    clearProgress,
    setEvidence,
  ]);

  // API 모드 선택 모달 상태
  const [showApiModeModal, setShowApiModeModal] = useState(false);

  // ESC 키로 모달 닫기
  useEffect(() => {
    if (!showApiModeModal) return;

    const handleEscape = (e) => {
      if (e.key === 'Escape') {
        setShowApiModeModal(false);
      }
    };

    document.addEventListener('keydown', handleEscape);
    return () => {
      document.removeEventListener('keydown', handleEscape);
    };
  }, [showApiModeModal]);

  return (
    <div className="chat-container">
      {/* 상단 컨트롤 영역 */}
      <div className="chat-controls">
        <div className="chat-controls-left">
          {/* API 모드 표시 (현재 선택된 모드만 표시) */}
          <div className="api-mode-display">
            <span className="api-mode-label">모드:</span>
            <span className="api-mode-value">{apiMode === 'query' ? '기본 RAG' : 'Agentic'}</span>
          </div>
        </div>
        <div className="chat-controls-right">
          <button
            className="chat-refresh-button"
            onClick={handleClearChat}
            title="채팅 기록 초기화"
          >
            <span>🔄</span>
            <span>새로고침</span>
          </button>
        </div>
      </div>

      {/* 진행 상황 표시 */}
      {progress.stage && (
        <ProgressIndicator
          stage={progress.stage}
          progress={progress.progress}
          message={progress.message}
          apiMode={apiMode}
          reretrieveCount={progress.reretrieveCount}
        />
      )}

      {/* 메시지 영역 */}
      <div className="chat-messages" ref={messagesContainerRef} id="chatMessages">
        {messages.map((msg) => (
          <ChatMessage
            key={msg.id}
            type={msg.type}
            content={msg.content}
            sources={msg.sources}
            metadata={msg.metadata}
          />
        ))}
      </div>

      {/* 입력 영역 */}
      <div className="chat-input-wrapper">
        <button
          className="api-mode-toggle-button"
          onClick={() => setShowApiModeModal(true)}
          title="API 모드 선택"
          aria-label="API 모드 선택"
        >
          <span>➕</span>
        </button>
        <ChatInput onSend={handleSend} disabled={loading} />
      </div>

      {/* API 모드 선택 모달 */}
      {showApiModeModal && (
        <div className="api-mode-modal-overlay" onClick={() => setShowApiModeModal(false)}>
          <div className="api-mode-modal" onClick={(e) => e.stopPropagation()}>
            <div className="api-mode-modal-header">
              <h3>API 모드 선택</h3>
              <button
                className="api-mode-modal-close"
                onClick={() => setShowApiModeModal(false)}
                aria-label="닫기"
              >
                &times;
              </button>
            </div>
            <div className="api-mode-modal-body">
              <div className="api-mode-option">
                <label>
                  <input
                    type="radio"
                    name="apiMode"
                    value="query"
                    checked={apiMode === 'query'}
                    onChange={(e) => {
                      setApiMode(e.target.value);
                      setShowApiModeModal(false);
                    }}
                  />
                  <div className="api-mode-option-content">
                    <strong>기본 RAG</strong>
                    <p>전통적인 RAG 방식으로 빠르고 안정적인 답변을 제공합니다.</p>
                  </div>
                </label>
              </div>
              <div className="api-mode-option">
                <label>
                  <input
                    type="radio"
                    name="apiMode"
                    value="agentic"
                    checked={apiMode === 'agentic'}
                    onChange={(e) => {
                      setApiMode(e.target.value);
                      setShowApiModeModal(false);
                    }}
                  />
                  <div className="api-mode-option-content">
                    <strong>Agentic[진행중]</strong>
                    <p>LangGraph 기반 에이전트로 더 지능적이고 맥락을 고려한 답변을 제공합니다.</p>
                  </div>
                </label>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

