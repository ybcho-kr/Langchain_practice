import { memo } from 'react';

/**
 * 진행 상황 표시 컴포넌트
 * @param {Object} props
 * @param {string} props.stage - 현재 단계 ('analyzing', 'planning', 'searching', 'evaluating', 'generating', 'reranking', 'rewriting')
 * @param {number} props.progress - 진행률 (0-100)
 * @param {string} props.message - 진행 상황 메시지
 * @param {string} props.apiMode - API 모드 ('query' | 'agentic')
 * @param {number} props.reretrieveCount - 재검색 횟수 (고급 그래프용)
 */
function ProgressIndicator({ stage, progress, message, apiMode = 'query', reretrieveCount = 0 }) {
  if (!stage) {
    return null;
  }

  // 단계별 아이콘 및 메시지 매핑
  const stageConfig = {
    // 기본 RAG 단계
    searching: { icon: '🔍', label: '검색 중', color: '#007bff' },
    reranking: { icon: '🔄', label: '리랭킹 중', color: '#6c757d' },
    generating: { icon: '✍️', label: '답변 생성 중', color: '#28a745' },
    
    // LangGraph 단계
    analyzing: { icon: '🧠', label: '질문 분석 중', color: '#007bff' },
    planning: { icon: '📋', label: '계획 수립 중', color: '#17a2b8' },
    evaluating: { icon: '✅', label: '결과 평가 중', color: '#ffc107' },
    rewriting: { icon: '✏️', label: '쿼리 재작성 중', color: '#fd7e14' },
  };

  const config = stageConfig[stage] || { icon: '⏳', label: '처리 중', color: '#6c757d' };
  const displayMessage = message || config.label;

  // 진행률 바 스타일
  const progressBarStyle = {
    width: `${Math.min(100, Math.max(0, progress))}%`,
    backgroundColor: config.color,
    height: '4px',
    transition: 'width 0.3s ease',
    borderRadius: '2px',
  };

  // 재검색 표시 (고급 그래프)
  const reretrieveInfo = reretrieveCount > 0 && apiMode === 'agentic' ? (
    <span style={{ fontSize: '0.85em', color: '#6c757d', marginLeft: '10px' }}>
      (재검색 {reretrieveCount}회)
    </span>
  ) : null;

  return (
    <div className="progress-indicator" style={{
      padding: '12px 16px',
      background: '#f8f9fa',
      borderRadius: '8px',
      marginBottom: '12px',
      border: `1px solid ${config.color}20`,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', marginBottom: '8px' }}>
        <span style={{ fontSize: '1.2em', marginRight: '8px' }}>{config.icon}</span>
        <span style={{ fontWeight: '500', color: '#333' }}>{displayMessage}</span>
        {reretrieveInfo}
      </div>
      <div style={{
        width: '100%',
        height: '4px',
        backgroundColor: '#e9ecef',
        borderRadius: '2px',
        overflow: 'hidden',
      }}>
        <div style={progressBarStyle} />
      </div>
      {progress > 0 && progress < 100 && (
        <div style={{
          fontSize: '0.85em',
          color: '#6c757d',
          marginTop: '4px',
          textAlign: 'right',
        }}>
          {Math.round(progress)}%
        </div>
      )}
    </div>
  );
}

export default memo(ProgressIndicator);

