import { useEffect, useState } from 'react';
import { useAppStore } from '../../stores/appStore';
import { useConfig } from '../../hooks/useApi';
import { saveSearchSettings, loadSearchSettings, clearSearchSettings } from '../../services/storage';
import RetrieverSettings from './RetrieverSettings';
import RerankerSettings from './RerankerSettings';
import SearchSettings from './SearchSettings';
import ModelSelector from '../Chat/ModelSelector';
import '../../styles/components/settings.css';

/**
 * 검색 설정 컴포넌트
 */
export default function Settings() {
  const { settings, model, updateSettings, normalizeWeights } = useAppStore();
  const [config, setConfig] = useState(null);
  const { execute: loadConfig } = useConfig();

  // 설정 로드
  useEffect(() => {
    loadInitialConfig();
  }, []);

  const loadInitialConfig = async () => {
    try {
      // 서버 설정 로드
      const serverConfig = await loadConfig();
      if (serverConfig) {
        setConfig(serverConfig);
        // 저장된 설정이 없으면 서버 기본값 적용
        const savedSettings = loadSearchSettings();
        if (!savedSettings) {
          if (serverConfig.max_sources) {
            updateSettings({ maxSources: serverConfig.max_sources });
          }
          if (serverConfig.score_threshold) {
            updateSettings({ scoreThreshold: serverConfig.score_threshold });
          }
          if (serverConfig.reranker_enabled && serverConfig.reranker_alpha !== undefined) {
            updateSettings({ rerankerAlpha: serverConfig.reranker_alpha });
          }
        }
      }
    } catch (error) {
      console.error('설정 로드 실패:', error);
    }
  };

  // 가중치 미리보기 계산
  const weightsSum = (settings.weights.qdrant + settings.weights.faiss + settings.weights.bm25).toFixed(2);

  // 현재 적용된 검색기 목록
  const retrieversList = [];
  if (settings.useQdrant) retrieversList.push('Qdrant');
  if (settings.useFaiss) retrieversList.push('FAISS');
  if (settings.useBm25) retrieversList.push('BM25');
  const retrieversText = retrieversList.length > 0 ? retrieversList.join(', ') : '없음';

  // 프리셋 적용
  const applyPreset = (preset) => {
    switch (preset) {
      case 'strict':
        updateSettings({ maxSources: 5, scoreThreshold: 0.8 });
        break;
      case 'normal':
        updateSettings({ maxSources: 5, scoreThreshold: 0.6 });
        break;
      case 'loose':
        updateSettings({ maxSources: 10, scoreThreshold: 0.4 });
        break;
    }
  };

  // 설정 저장
  const handleSaveSettings = () => {
    const success = saveSearchSettings({
      max_sources: settings.maxSources,
      score_threshold: settings.scoreThreshold,
      use_qdrant: settings.useQdrant,
      use_faiss: settings.useFaiss,
      use_bm25: settings.useBm25,
      use_reranker: settings.useReranker,
      reranker_alpha: settings.rerankerAlpha,
      reranker_top_k: settings.rerankerTopK,
      weights: settings.weights,
      slider_wq: settings.sliderWeights.qdrant,
      slider_wf: settings.sliderWeights.faiss,
      slider_wb: settings.sliderWeights.bm25,
    });

    if (success && window.showToast) {
      window.showToast('검색 설정이 저장되었습니다. 새로고침 후에도 유지됩니다.', 'success');
    }
  };

  // 설정 초기화
  const handleResetSettings = async () => {
    clearSearchSettings();
    await loadInitialConfig();
    if (window.showToast) {
      window.showToast('설정 파일의 기본값으로 재설정되었습니다.', 'success');
    }
  };

  // 설정 초기화 (저장값 삭제)
  const handleClearSavedSettings = () => {
    clearSearchSettings();
    if (window.showToast) {
      window.showToast('저장된 검색 설정이 삭제되었습니다. 서버 기본값이 적용됩니다.', 'info');
    }
    handleResetSettings();
  };

  return (
    <div className="settings-container">
        <div className="evidence-header">
          <h2>⚙️ 검색 설정</h2>
          <p>RAG 시스템의 검색 파라미터를 조정할 수 있습니다.</p>
        </div>

        <div style={{ padding: '20px' }}>
          {/* 언어 모델 선택 */}
          <ModelSelector />

          {/* 시스템 설정 정보 */}
          <div className="model-selection-section">
            <h3>ℹ️ 시스템 설정 정보</h3>
            <div className="settings-info">
              <div style={{ marginBottom: '8px' }}>
                <strong>LLM 모델:</strong> <span id="configLlmModel">{config?.llm_model || model || '로딩 중...'}</span>
              </div>
              <div style={{ marginBottom: '8px' }}>
                <strong>임베딩 모델:</strong> <span id="configEmbeddingModel">{config?.embedding_model || '로딩 중...'}</span>
              </div>
              <div style={{ marginBottom: '8px' }}>
                <strong>리랭커 모델:</strong>{' '}
                <span id="configRerankerModel">
                  {config?.reranker_enabled && config?.reranker_model ? config.reranker_model : '비활성화됨'}
                </span>
              </div>
              <div style={{ marginTop: '10px', paddingTop: '10px', borderTop: '1px solid #565869' }}>
                <small>💡 이 값들은 기본 설정값입니다.</small>
              </div>
            </div>
          </div>

          {/* 검색기 설정 */}
          <RetrieverSettings />

          {/* 리랭커 설정 */}
          <RerankerSettings />

          {/* 검색 설정 */}
          <SearchSettings />

          {/* 가중치 합산 미리보기 */}
          <div className="model-selection-section" style={{ marginTop: '30px' }}>
            <h3>🧮 가중치 합산 미리보기</h3>
            <div className="settings-preview" id="weightsPreview">
              <div>
                검색기 가중치 합: <span id="weightsSum">{weightsSum}</span> (Qdrant=
                <span id="pvQ">{settings.weights.qdrant.toFixed(2)}</span>, FAISS=
                <span id="pvF">{settings.weights.faiss.toFixed(2)}</span>, BM25=
                <span id="pvB">{settings.weights.bm25.toFixed(2)}</span>)
              </div>
              <div style={{ marginTop: '6px' }}>
                최종 점수 식: score = α·reranker + (1-α)·base (α=
                <span id="pvA">{settings.rerankerAlpha.toFixed(2)}</span>)
              </div>
            </div>
          </div>

          {/* 현재 적용된 설정 */}
          <div className="model-selection-section" style={{ marginTop: '30px' }}>
            <h3>📋 현재 적용된 설정</h3>
            <div className="settings-preview" style={{ marginTop: '15px' }}>
              <div style={{ marginBottom: '10px' }}>
                <strong>검색기:</strong> <span id="currentRetrievers">{retrieversText}</span>
              </div>
              <div style={{ marginBottom: '10px' }}>
                <strong>최대 소스 수:</strong> <span id="currentMaxSources">{settings.maxSources}</span>
              </div>
              <div style={{ marginBottom: '10px' }}>
                <strong>유사도 임계값:</strong> <span id="currentScoreThreshold">{settings.scoreThreshold}</span>
              </div>
              <div style={{ marginTop: '15px', paddingTop: '15px', borderTop: '1px solid #565869' }}>
                <small>⚠️ 설정을 변경한 후 채팅 탭에서 질문을 하면 변경된 값이 적용됩니다.</small>
              </div>
            </div>
          </div>

          {/* 빠른 설정 프리셋 */}
          <div className="model-selection-section" style={{ marginTop: '30px' }}>
            <h3>⚡ 빠른 설정 프리셋</h3>
            <div style={{ display: 'flex', gap: '10px', marginTop: '15px', flexWrap: 'wrap' }}>
              <button className="btn btn-primary" onClick={() => applyPreset('strict')} aria-label="높은 신뢰도 수집 프리셋 적용">
                높은 신뢰도 수집 (임계값: 0.8)
              </button>
              <button className="btn btn-primary" onClick={() => applyPreset('normal')} aria-label="기본 수집 프리셋 적용">
                기본 수집 (임계값: 0.6)
              </button>
              <button className="btn btn-primary" onClick={() => applyPreset('loose')} aria-label="많은 신뢰도 수집 프리셋 적용">
                많은 신뢰도 수집 (임계값: 0.4)
              </button>
              <button className="btn btn-secondary" onClick={handleResetSettings} aria-label="설정 파일 기본값으로 재설정">
                설정 파일 기본값으로 재설정
              </button>
              <button className="btn btn-primary" onClick={handleSaveSettings} aria-label="검색 설정 저장">
                설정 저장
              </button>
              <button className="btn btn-secondary" onClick={handleClearSavedSettings} aria-label="저장된 설정 초기화">
                저장값 초기화
              </button>
            </div>
            <div className="info-text" style={{ marginTop: '12px' }}>
              <small>
                💾 설정 저장: 브라우저에 저장되어 새로고침 후에도 적용됩니다. (서버 기본값 위에 덮어쓰기)
                <br />
                ♻️ 저장값 초기화: 저장된 사용자 설정을 삭제하고 서버 기본값으로 되돌립니다.
              </small>
            </div>
          </div>
        </div>
      </div>
  );
}

