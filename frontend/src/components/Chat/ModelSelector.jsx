import { useState, useEffect } from 'react';
import { useAppStore } from '../../stores/appStore';
import { useModels } from '../../hooks/useApi';
import { apiClient } from '../../services/api';

/**
 * 모델 선택 컴포넌트
 */
export default function ModelSelector() {
  const { model, updateModel } = useAppStore();
  const [models, setModels] = useState([]);

  const { execute: loadModels, loading } = useModels();

  // 초기 모델 목록 로드
  useEffect(() => {
    refreshModels();
  }, []);

  const refreshModels = async () => {
    try {
      const data = await loadModels();
      if (data?.available_models) {
        setModels(data.available_models);
        if (data.current_model && data.current_model !== model) {
          updateModel(data.current_model);
        }
      }
    } catch (error) {
      console.error('모델 목록 로드 실패:', error);
      if (window.showToast) {
        window.showToast(`모델 목록 로드 실패: ${error.message}`, 'error');
      }
    }
  };

  const handleModelChange = (e) => {
    const newModel = e.target.value;
    updateModel(newModel);
  };

  return (
    <div className="model-selection-section">
      <h3>🤖 언어 모델 선택</h3>
      <div className="model-controls">
        <select
          id="modelSelect"
          aria-label="언어 모델 선택"
          value={model}
          onChange={handleModelChange}
          disabled={loading}
        >
          {models.length > 0 ? (
            models.map((m) => (
              <option key={m.name} value={m.name}>
                {m.name} ({m.size || 'N/A'})
              </option>
            ))
          ) : (
            <option value={model}>{model} (기본)</option>
          )}
        </select>
        <button
          onClick={refreshModels}
          className="btn btn-secondary btn-sm"
          aria-label="모델 목록 새로고침"
          disabled={loading}
        >
          {loading ? '로딩 중...' : '모델 새로고침'}
        </button>
        <div className="model-info">
          <span id="modelStatus">현재 모델: {model}</span>
        </div>
      </div>
    </div>
  );
}

