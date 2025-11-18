import { useState, useEffect } from 'react';
import { useAppStore } from './stores/appStore';
import { useConfig } from './hooks/useApi';
import { loadSearchSettings, getItem, setItem } from './services/storage';
import { ToastContainer } from './components/common/Toast';
import Chat from './components/Chat/Chat';
import Documents from './components/Documents/Documents';
import Settings from './components/Settings/Settings';
import Evidence from './components/Evidence/Evidence';
import Sidebar from './components/common/Sidebar';
import { apiClient } from './services/api';
import './styles/components/progress.css';
import './styles/components/session.css';
import './styles/components/sidebar.css';
import './styles/components/modals.css';
import './styles/main.css';

const ACTIVE_TAB_KEY = 'active_tab';

/**
 * 메인 App 컴포넌트
 */
function App() {
  // localStorage에서 저장된 탭 복원 (없으면 'chat' 기본값)
  const [activeTab, setActiveTab] = useState(() => {
    const savedTab = getItem(ACTIVE_TAB_KEY, 'chat');
    return savedTab || 'chat';
  });
  const { settings, updateSettings, updateModel } = useAppStore();
  const { execute: loadConfig } = useConfig();

  // 초기 설정 로드
  useEffect(() => {
    initializeApp();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // 탭 전환 이벤트 리스너
  useEffect(() => {
    const handleSwitchTab = (e) => {
      setActiveTab(e.detail);
    };

    const handleShowEvidence = () => {
      setActiveTab('evidence');
    };

    window.addEventListener('switchTab', handleSwitchTab);
    window.addEventListener('showEvidence', handleShowEvidence);
    return () => {
      window.removeEventListener('switchTab', handleSwitchTab);
      window.removeEventListener('showEvidence', handleShowEvidence);
    };
  }, []);

  const initializeApp = async () => {
    try {
      // 서버 헬스 체크
      await apiClient.healthCheck();

      // 서버 설정 로드
      const config = await loadConfig();
      if (config) {
        // 모델 정보 업데이트
        if (config.llm_model) {
          updateModel(config.llm_model);
        }

        // 저장된 설정이 없으면 서버 기본값 적용
        const savedSettings = loadSearchSettings();
        if (!savedSettings) {
          if (config.max_sources) {
            updateSettings({ maxSources: config.max_sources });
          }
          if (config.score_threshold) {
            updateSettings({ scoreThreshold: config.score_threshold });
          }
          if (config.reranker_enabled && config.reranker_alpha !== undefined) {
            updateSettings({ rerankerAlpha: config.reranker_alpha });
          }
        } else {
          // 저장된 설정 복원
          const updates = {};
          if (savedSettings.max_sources) updates.maxSources = savedSettings.max_sources;
          if (savedSettings.score_threshold) updates.scoreThreshold = savedSettings.score_threshold;
          if (savedSettings.use_qdrant !== undefined) updates.useQdrant = savedSettings.use_qdrant;
          if (savedSettings.use_faiss !== undefined) updates.useFaiss = savedSettings.use_faiss;
          if (savedSettings.use_bm25 !== undefined) updates.useBm25 = savedSettings.use_bm25;
          if (savedSettings.use_reranker !== undefined) updates.useReranker = savedSettings.use_reranker;
          if (savedSettings.reranker_alpha !== undefined) updates.rerankerAlpha = savedSettings.reranker_alpha;
          if (savedSettings.reranker_top_k !== undefined) updates.rerankerTopK = savedSettings.reranker_top_k;
          
          if (savedSettings.slider_wq !== undefined || savedSettings.slider_wf !== undefined || savedSettings.slider_wb !== undefined) {
            updates.sliderWeights = {
              ...settings.sliderWeights,
            };
            if (savedSettings.slider_wq !== undefined) updates.sliderWeights.qdrant = savedSettings.slider_wq;
            if (savedSettings.slider_wf !== undefined) updates.sliderWeights.faiss = savedSettings.slider_wf;
            if (savedSettings.slider_wb !== undefined) updates.sliderWeights.bm25 = savedSettings.slider_wb;
          }
          
          if (Object.keys(updates).length > 0) {
            updateSettings(updates);
          }
        }
      }
    } catch (error) {
      console.error('앱 초기화 실패:', error);
      if (window.showToast) {
        window.showToast('서버 연결에 실패했습니다. API 서버가 실행 중인지 확인해주세요.', 'error');
      }
    }
  };

  const handleTabChange = (tabId) => {
    setActiveTab(tabId);
    // localStorage에 현재 탭 저장
    setItem(ACTIVE_TAB_KEY, tabId);
  };

  return (
    <div className="container">
      <div className="main-layout" style={{
        display: 'flex',
        height: '100vh',
      }}>
        {/* 사이드바 (모든 탭에서 표시) */}
        <div className="sidebar-container">
          <Sidebar activeTab={activeTab} onTabChange={handleTabChange} />
        </div>

        {/* 메인 컨텐츠 영역 */}
        <div className="main-content" style={{
          flex: 1,
          overflow: 'auto',
          background: '#343541',
        }}>
          <div className={`tab-content ${activeTab === 'chat' ? 'active' : ''}`} id="chat" role="tabpanel" aria-labelledby="tab-chat">
            {activeTab === 'chat' && <Chat />}
          </div>
          <div className={`tab-content ${activeTab === 'documents' ? 'active' : ''}`} id="documents" role="tabpanel" aria-labelledby="tab-documents">
            {activeTab === 'documents' && <Documents />}
          </div>
          <div className={`tab-content ${activeTab === 'settings' ? 'active' : ''}`} id="settings" role="tabpanel" aria-labelledby="tab-settings">
            {activeTab === 'settings' && <Settings />}
          </div>
          <div className={`tab-content ${activeTab === 'evidence' ? 'active' : ''}`} id="evidence" role="tabpanel" aria-labelledby="tab-evidence">
            {activeTab === 'evidence' && <Evidence />}
          </div>
        </div>
      </div>

      <ToastContainer />
    </div>
  );
}

export default App;

