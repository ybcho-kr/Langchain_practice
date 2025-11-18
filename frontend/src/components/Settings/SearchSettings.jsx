import { useAppStore } from '../../stores/appStore';

/**
 * 문서 검색 설정 컴포넌트
 */
export default function SearchSettings() {
  const { settings, updateSettings } = useAppStore();
  const { maxSources, scoreThreshold } = settings;

  const handleMaxSourcesChange = (value) => {
    const numValue = parseInt(value) || 5;
    const clampedValue = Math.max(1, Math.min(20, numValue));
    updateSettings({ maxSources: clampedValue });
  };

  const handleScoreThresholdChange = (value) => {
    const numValue = parseFloat(value);
    // 빈 문자열이거나 유효하지 않은 값이면 업데이트하지 않음
    if (value === '' || isNaN(numValue)) {
      return;
    }
    const clampedValue = Math.max(0, Math.min(1, numValue));
    updateSettings({ scoreThreshold: clampedValue });
  };

  const handleScoreThresholdBlur = (value) => {
    const numValue = parseFloat(value);
    // 빈 문자열이거나 유효하지 않은 값이면 기본값으로 설정
    if (value === '' || isNaN(numValue)) {
      updateSettings({ scoreThreshold: 0.85 });
      return;
    }
    const clampedValue = Math.max(0, Math.min(1, numValue));
    updateSettings({ scoreThreshold: clampedValue });
  };

  return (
    <div className="model-selection-section" style={{ marginTop: '30px' }}>
      <h3>📊 문서 검색 설정</h3>
      <div className="model-controls" style={{ flexWrap: 'wrap', gap: '20px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '15px' }}>
          <label htmlFor="maxSources" className="settings-label">
            최대 소스 수:
          </label>
          <input
            type="number"
            id="maxSources"
            min="1"
            max="20"
            value={maxSources}
            onChange={(e) => handleMaxSourcesChange(e.target.value)}
            aria-label="최대 소스 수 입력"
            className="settings-input"
            style={{
              width: '100px',
            }}
          />
          <span className="settings-hint">(1-20)</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '15px' }}>
          <label htmlFor="scoreThreshold" className="settings-label">
            유사도 임계값:
          </label>
          <input
            type="number"
            id="scoreThreshold"
            min="0"
            max="1"
            step="0.01"
            value={scoreThreshold}
            onChange={(e) => {
              // 입력 중에는 값만 업데이트 (검증은 onBlur에서)
              const numValue = parseFloat(e.target.value);
              if (!isNaN(numValue) && numValue >= 0 && numValue <= 1) {
                updateSettings({ scoreThreshold: numValue });
              }
            }}
            onBlur={(e) => handleScoreThresholdBlur(e.target.value)}
            aria-label="유사도 임계값 입력"
            className="settings-input"
            style={{
              width: '120px',
            }}
          />
          <span className="settings-hint">(0.0-1.0)</span>
        </div>
        <div className="info-text" style={{ marginTop: '15px', width: '100%' }}>
          <small>
            💡 <strong>최대 소스 수</strong>: 검색 결과로 반환할 최대 문서 수입니다. (기본값: 설정 파일에서 로드)
            <br />
            💡 <strong>유사도 임계값</strong>: 이 값 이상인 문서만 검색 결과에 포함됩니다. 높을수록 더 관련성 높은 문서만 검색됩니다. 임계값
            이상의 문서가 없으면 일반답변으로 대체합니다.)
          </small>
        </div>
      </div>
    </div>
  );
}

