import { memo, useState } from 'react';
import { sanitizeHTML } from '../../utils/sanitize';

/**
 * 소스 카드 컴포넌트
 * @param {Object} props
 * @param {Object} props.source - 소스 객체
 * @param {number} props.index - 인덱스
 * @param {Function} props.onViewFull - 전체 내용 보기 핸들러
 */
function SourceCard({ source, index, onViewFull }) {
  const [expanded, setExpanded] = useState(false);
  
  // source_file이 null이거나 없을 때 처리
  const getFilename = (sourceFile) => {
    if (!sourceFile || sourceFile === null || sourceFile === 'N/A') {
      return null;
    }
    return sourceFile.split('\\').pop()?.split('/').pop() || sourceFile;
  };
  
  const filename = getFilename(source.source_file);
  const displayPath = source.source_path || filename || '문서 정보 없음';
  const safeDisplayPath = sanitizeHTML(displayPath);
  const safeFilename = filename ? sanitizeHTML(filename) : null;
  const safeContent = sanitizeHTML(source.content || '');
  const safeScore = ((source.relevance_score || source.score || 0) * 100).toFixed(1);
  const scoreValue = parseFloat(safeScore);

  // 관련도에 따른 색상 결정 (다크 테마에 맞게 밝은 색상 사용)
  const getScoreColor = (score) => {
    if (score >= 80) return '#4ade80'; // 밝은 녹색
    if (score >= 60) return '#fbbf24'; // 밝은 노란색
    if (score >= 40) return '#fb923c'; // 밝은 주황색
    return '#f87171'; // 밝은 빨간색
  };

  const scoreColor = getScoreColor(scoreValue);

  // 미리보기 길이 제한 (200자)
  const previewLength = 200;
  const shouldTruncate = safeContent.length > previewLength;
  const displayContent = expanded || !shouldTruncate 
    ? safeContent 
    : safeContent.substring(0, previewLength) + '...';

  return (
    <div className="source-card" style={{
      background: '#40414f',
      border: `1px solid ${scoreColor}40`,
      borderRadius: '8px',
      marginBottom: '16px',
      overflow: 'hidden',
      transition: 'box-shadow 0.3s',
    }}
    onMouseEnter={(e) => e.currentTarget.style.boxShadow = `0 4px 8px ${scoreColor}20`}
    onMouseLeave={(e) => e.currentTarget.style.boxShadow = 'none'}
    >
      <div className="source-header" style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        padding: '12px 16px',
        background: `${scoreColor}15`,
        borderBottom: `1px solid ${scoreColor}30`,
      }}>
        <div className="source-file" style={{ fontWeight: '500', fontSize: '0.95em', color: '#ececf1' }}>
          {index + 1}. {safeDisplayPath}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          {/* 관련도 시각화 */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <div style={{
              width: '120px',
              height: '8px',
              backgroundColor: '#2d2d3a',
              borderRadius: '4px',
              overflow: 'hidden',
            }}>
              <div style={{
                width: `${scoreValue}%`,
                height: '100%',
                backgroundColor: scoreColor,
                transition: 'width 0.3s ease',
              }} />
            </div>
            <div className="source-score" style={{
              fontWeight: '600',
              color: scoreColor,
              minWidth: '50px',
              fontSize: '0.9em',
            }}>
              {safeScore}%
            </div>
          </div>
        </div>
      </div>
      
      <div className="source-content" 
        style={{
          padding: '16px',
          fontSize: '0.9em',
          lineHeight: '1.6',
          color: '#ececf1',
        }}
        dangerouslySetInnerHTML={{ __html: displayContent }} 
      />
      
      {shouldTruncate && (
        <div style={{ padding: '0 16px 8px', textAlign: 'right' }}>
          <button
            onClick={() => setExpanded(!expanded)}
            style={{
              background: 'none',
              border: 'none',
              color: '#10a37f',
              cursor: 'pointer',
              fontSize: '0.85em',
              textDecoration: 'underline',
            }}
          >
            {expanded ? '접기' : '더 보기'}
          </button>
        </div>
      )}
      
      <div className="source-metadata" style={{
        padding: '10px 16px',
        background: '#2d2d3a',
        borderTop: '1px solid #565869',
        fontSize: '0.85em',
        color: '#8e8ea0',
        display: 'flex',
        flexWrap: 'wrap',
        gap: '12px',
      }}>
        <span>청크 인덱스: {source.chunk_index ?? 'N/A'}</span>
        {filename && <span>파일: {safeFilename}</span>}
        {source.metadata?.chunk_id && (
          <span>청크 ID: {source.metadata.chunk_id}</span>
        )}
        {source.score && (
          <span>점수: {source.score.toFixed(4)}</span>
        )}
      </div>
      
      <div className="source-actions" style={{
        padding: '10px 16px',
        background: '#2d2d3a',
        borderTop: '1px solid #565869',
        textAlign: 'right',
      }}>
        <button
          className="btn btn-primary btn-sm"
          onClick={onViewFull}
          style={{
            padding: '6px 12px',
            fontSize: '0.875em',
            background: '#10a37f',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            transition: 'background 0.2s',
          }}
          onMouseEnter={(e) => e.currentTarget.style.background = '#0d8f6e'}
          onMouseLeave={(e) => e.currentTarget.style.background = '#10a37f'}
        >
          📖 자세히 보기
        </button>
      </div>
    </div>
  );
}

export default memo(SourceCard);

