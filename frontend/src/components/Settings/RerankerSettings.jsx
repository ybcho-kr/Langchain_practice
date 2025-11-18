import { useAppStore } from '../../stores/appStore';

/**
 * 리랭커 설정 컴포넌트
 */
export default function RerankerSettings() {
  const { settings, updateSettings } = useAppStore();
  const { useReranker, rerankerAlpha, rerankerTopK } = settings;

  const handleRerankerToggle = (checked) => {
    updateSettings({ useReranker: checked });
  };

  const handleAlphaChange = (value) => {
    const numValue = parseFloat(value) || 0;
    const clampedValue = Math.max(0, Math.min(1, numValue));
    updateSettings({ rerankerAlpha: clampedValue });
  };

  const handleTopKChange = (value) => {
    const numValue = parseInt(value) || 3;
    const clampedValue = Math.max(1, Math.min(50, numValue));
    updateSettings({ rerankerTopK: clampedValue });
  };

  return (
    <div className="model-selection-section" style={{ marginTop: '30px' }}>
      <h3>🏷️ 리랭커 설정</h3>
      <div className="settings-preview" style={{ marginTop: '15px' }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: '10px', cursor: 'pointer', padding: '8px' }}>
          <input
            type="checkbox"
            id="useReranker"
            checked={useReranker}
            onChange={(e) => handleRerankerToggle(e.target.checked)}
            aria-label="리랭커 사용"
            style={{ width: '18px', height: '18px', cursor: 'pointer', accentColor: '#10a37f' }}
          />
          <span style={{ fontWeight: 500, color: '#ececf1' }}>리랭커 사용</span>
        </label>
        {useReranker && (
          <>
            <div style={{ marginLeft: '28px', marginTop: '8px', display: 'flex', alignItems: 'center', gap: '10px' }}>
              <label htmlFor="alpha" className="settings-label" style={{ minWidth: '120px' }}>리랭커 비율(α)</label>
              <input
                type="range"
                id="alpha"
                min="0"
                max="1"
                step="0.05"
                value={rerankerAlpha}
                onChange={(e) => handleAlphaChange(e.target.value)}
                aria-label="리랭커 비율 슬라이더"
                style={{ flex: 1 }}
              />
              <span id="alphaVal" aria-live="polite" style={{ color: '#ececf1', minWidth: '50px' }}>
                {rerankerAlpha.toFixed(2)}
              </span>
            </div>
            <div style={{ marginLeft: '28px', marginTop: '8px', display: 'flex', alignItems: 'center', gap: '10px' }}>
              <label htmlFor="rerankerTopK" className="settings-label" style={{ minWidth: '120px' }}>리랭커 K(top_k)</label>
              <input
                type="number"
                id="rerankerTopK"
                min="1"
                max="50"
                step="1"
                value={rerankerTopK}
                onChange={(e) => handleTopKChange(e.target.value)}
                aria-label="리랭커 top_k 값 입력"
                className="settings-input"
                style={{ width: '100px', padding: '6px' }}
              />
              <small className="settings-hint">(1-50)</small>
            </div>
            <div className="info-text" style={{ marginTop: '15px' }}>
              <small>
                최종 점수 = α·리랭커점수 + (1-α)·기본점수. α가 클수록 리랭커 영향이 커집니다.
              </small>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

