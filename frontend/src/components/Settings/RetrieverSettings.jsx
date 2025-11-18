import { useEffect } from 'react';
import { useAppStore } from '../../stores/appStore';

/**
 * 검색기 설정 컴포넌트
 */
export default function RetrieverSettings() {
  const { settings, updateSettings, normalizeWeights } = useAppStore();
  const { useQdrant, useFaiss, useBm25, sliderWeights, weights, denseWeight, sparseWeight } = settings;
  
  // 기본값 설정 (undefined 방지)
  const denseWeightValue = denseWeight ?? 0.7;
  const sparseWeightValue = sparseWeight ?? 0.3;

  // 가중치 정규화
  useEffect(() => {
    normalizeWeights();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [useQdrant, useFaiss, useBm25, sliderWeights.qdrant, sliderWeights.faiss, sliderWeights.bm25]);

  const handleRetrieverChange = (type, checked) => {
    if (type === 'qdrant' && checked) {
      // Qdrant 선택 시 FAISS 해제
      updateSettings({ useQdrant: true, useFaiss: false });
    } else if (type === 'faiss' && checked) {
      // FAISS 선택 시 Qdrant 해제
      updateSettings({ useQdrant: false, useFaiss: true });
    } else if (type === 'qdrant') {
      updateSettings({ useQdrant: checked });
    } else if (type === 'faiss') {
      updateSettings({ useFaiss: checked });
    } else if (type === 'bm25') {
      updateSettings({ useBm25: checked });
    }

    // Qdrant만 사용 (FAISS, BM25는 숨김 처리됨)
    if (type === 'qdrant' && !checked) {
      // Qdrant 해제 시 강제로 다시 체크
      updateSettings({ useQdrant: true });
      if (window.showToast) {
        window.showToast('Qdrant 검색기는 필수입니다.', 'warning');
      }
    }
  };

  const handleWeightChange = (type, value) => {
    const numValue = parseFloat(value) || 0;
    const clampedValue = Math.max(0, Math.min(1, numValue));
    updateSettings({
      sliderWeights: {
        ...sliderWeights,
        [type]: clampedValue,
      },
    });
  };

  return (
    <div className="model-selection-section" style={{ marginTop: '30px' }}>
      <h3>🔍 하이브리드 검색 가중치</h3>
      <div className="settings-preview" style={{ marginTop: '15px' }}>
        {/* Dense/Sparse 가중치 조절 */}
        <div>
          <small style={{ display: 'block', marginBottom: '8px', color: '#8e8ea0', fontWeight: 500 }}>
            하이브리드 검색 가중치 (Dense + Sparse)
          </small>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
            <label htmlFor="denseWeight" className="settings-label" style={{ minWidth: '100px' }}>Dense 가중치</label>
            <input
              type="range"
              id="denseWeight"
              min="0"
              max="1"
              step="0.01"
              value={denseWeightValue}
              onChange={(e) => {
                const val = parseFloat(e.target.value);
                const sparseVal = Math.max(0, Math.min(1, 1.0 - val));
                updateSettings({ denseWeight: val, sparseWeight: sparseVal });
              }}
              aria-label="Dense 가중치 슬라이더"
              style={{ flex: 1 }}
            />
            <input
              type="number"
              id="denseWeightNum"
              min="0"
              max="1"
              step="0.01"
              value={denseWeightValue.toFixed(2)}
              onChange={(e) => {
                const val = Math.max(0, Math.min(1, parseFloat(e.target.value) || 0));
                const sparseVal = Math.max(0, Math.min(1, 1.0 - val));
                updateSettings({ denseWeight: val, sparseWeight: sparseVal });
              }}
              aria-label="Dense 가중치 수치 입력"
              className="settings-input"
              style={{ width: '80px', padding: '6px' }}
            />
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <label htmlFor="sparseWeight" className="settings-label" style={{ minWidth: '100px' }}>Sparse 가중치</label>
            <input
              type="range"
              id="sparseWeight"
              min="0"
              max="1"
              step="0.01"
              value={sparseWeightValue}
              onChange={(e) => {
                const val = parseFloat(e.target.value);
                const denseVal = Math.max(0, Math.min(1, 1.0 - val));
                updateSettings({ denseWeight: denseVal, sparseWeight: val });
              }}
              aria-label="Sparse 가중치 슬라이더"
              style={{ flex: 1 }}
            />
            <input
              type="number"
              id="sparseWeightNum"
              min="0"
              max="1"
              step="0.01"
              value={sparseWeightValue.toFixed(2)}
              onChange={(e) => {
                const val = Math.max(0, Math.min(1, parseFloat(e.target.value) || 0));
                const denseVal = Math.max(0, Math.min(1, 1.0 - val));
                updateSettings({ denseWeight: denseVal, sparseWeight: val });
              }}
              aria-label="Sparse 가중치 수치 입력"
              className="settings-input"
              style={{ width: '80px', padding: '6px' }}
            />
          </div>
          <small style={{ display: 'block', marginTop: '8px', color: '#8e8ea0', fontSize: '0.85em' }}>
            💡 Dense와 Sparse 가중치의 합이 1.0이 되도록 자동 조절됩니다.
          </small>
        </div>

        {/* FAISS - 숨김 처리 */}
        {false && (
          <div style={{ marginBottom: '15px', display: 'none' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '10px', cursor: 'pointer', padding: '8px' }}>
              <input
                type="checkbox"
                id="useFaiss"
                checked={useFaiss}
                onChange={(e) => handleRetrieverChange('faiss', e.target.checked)}
                aria-label="FAISS 벡터 검색 사용"
                style={{ width: '18px', height: '18px', cursor: 'pointer' }}
              />
              <span style={{ fontWeight: 500 }}>FAISS 벡터 검색</span>
            </label>
            <small id="faiss-desc" style={{ display: 'block', marginLeft: '28px', color: '#6c757d', fontSize: '0.9em' }}>
              FAISS 벡터 검색 (CPU 사용) 추후 GPU 가능성 여부 판단 후 개발
            </small>
            {useFaiss && (
              <div style={{ marginLeft: '28px', marginTop: '8px', display: 'flex', alignItems: 'center', gap: '10px' }}>
                <label htmlFor="wFaiss" style={{ minWidth: '90px' }}>가중치(FAISS)</label>
                <input
                  type="range"
                  id="wFaiss"
                  min="0"
                  max="1"
                  step="0.01"
                  value={sliderWeights.faiss}
                  onChange={(e) => handleWeightChange('faiss', e.target.value)}
                  aria-label="FAISS 가중치 슬라이더"
                  style={{ flex: 1 }}
                />
                <input
                  type="number"
                  id="wFaissNum"
                  min="0"
                  max="1"
                  step="0.01"
                  value={weights.faiss.toFixed(2)}
                  disabled
                  aria-label="FAISS 가중치 수치"
                  style={{ width: '80px', padding: '6px', border: '1px solid #ccc', borderRadius: '6px', background: '#f8f9fa' }}
                />
              </div>
            )}
          </div>
        )}

        {/* BM25 - 숨김 처리 */}
        {false && (
          <div style={{ marginBottom: '15px', display: 'none' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '10px', cursor: 'pointer', padding: '8px' }}>
              <input
                type="checkbox"
                id="useBm25"
                checked={useBm25}
                onChange={(e) => handleRetrieverChange('bm25', e.target.checked)}
                aria-label="BM25 키워드 검색 사용"
                style={{ width: '18px', height: '18px', cursor: 'pointer' }}
              />
              <span style={{ fontWeight: 500 }}>BM25 키워드 검색</span>
            </label>
            <small id="bm25-desc" style={{ display: 'block', marginLeft: '28px', color: '#6c757d', fontSize: '0.9em' }}>
              키워드 기반 통계 검색 (용어 빈도 기반)
            </small>
            {useBm25 && (
              <div style={{ marginLeft: '28px', marginTop: '8px', display: 'flex', alignItems: 'center', gap: '10px' }}>
                <label htmlFor="wBm25" style={{ minWidth: '90px' }}>가중치(BM25)</label>
                <input
                  type="range"
                  id="wBm25"
                  min="0"
                  max="1"
                  step="0.01"
                  value={sliderWeights.bm25}
                  onChange={(e) => handleWeightChange('bm25', e.target.value)}
                  aria-label="BM25 가중치 슬라이더"
                  style={{ flex: 1 }}
                />
                <input
                  type="number"
                  id="wBm25Num"
                  min="0"
                  max="1"
                  step="0.01"
                  value={weights.bm25.toFixed(2)}
                  disabled
                  aria-label="BM25 가중치 수치"
                  style={{ width: '80px', padding: '6px', border: '1px solid #ccc', borderRadius: '6px', background: '#f8f9fa' }}
                />
              </div>
            )}
          </div>
        )}

        <div className="info-text" style={{ marginTop: '15px' }}>
          <small>
            💡 Qdrant 하이브리드 검색 (Dense + Sparse 벡터)을 사용합니다.<br />
            💡 설정은 자동으로 저장되어 새로고침 후에도 유지됩니다.
          </small>
        </div>
      </div>
    </div>
  );
}

