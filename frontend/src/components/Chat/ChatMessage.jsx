import { memo } from 'react';
import { markdownToHtml } from '../../utils/markdown';
import { sanitizeHTML } from '../../utils/sanitize';
import { useAppStore } from '../../stores/appStore';

/**
 * 채팅 메시지 컴포넌트
 * @param {Object} props
 * @param {string} props.type - 메시지 타입 ('user' | 'assistant')
 * @param {string} props.content - 메시지 내용
 * @param {Array} props.sources - 참조 문서 목록
 * @param {Object} props.metadata - 메타데이터 (confidence, processing_time 등)
 */
function ChatMessage({ type, content, sources = [], metadata = null }) {
  const { evidence, setEvidence } = useAppStore();

  const header = type === 'user' ? '👤 사용자' : '🤖 시스템';
  let headerText = header;

  // 사용자 메시지는 메타데이터 표시하지 않음, 시스템 메시지만 표시
  if (type === 'assistant' && metadata) {
    const confidence = ((metadata.confidence || 0) * 100).toFixed(1);
    const processingTime = (metadata.processing_time || 0).toFixed(2);
    const maxSources = metadata.max_sources || 5;
    const scoreThreshold = (metadata.score_threshold || 0.85).toFixed(2);
    headerText += ` (신뢰도: ${confidence}%, 처리시간: ${processingTime}초`;
    headerText += `, 설정: max_sources=${maxSources}, threshold=${scoreThreshold})`;
  }

  // 시스템 답변의 경우 마크다운을 HTML로 변환
  const processedContent =
    type === 'assistant' ? markdownToHtml(content) : sanitizeHTML(content);

  const handleSourceClick = () => {
    // Evidence가 없어도 탭으로 이동 (최근 답변의 근거를 보여줄 수 있음)
    window.dispatchEvent(new CustomEvent('showEvidence'));
  };

  return (
    <div className={`message ${type}`}>
      <div className="message-header">{headerText}</div>
      <div
        className="message-content"
        dangerouslySetInnerHTML={{ __html: processedContent }}
      />
      {sources && sources.length > 0 && (
        <div className="sources">
          <strong>📚 참조 문서:</strong>
          {sources.map((source, index) => {
            const displayPath =
              source.source_path ||
              source.source_file.split('\\').pop().split('/').pop();
            const safeDisplayPath = sanitizeHTML(displayPath);
            const safeScore = ((source.relevance_score || 0) * 100).toFixed(1);

            return (
              <div
                key={index}
                className="source-item"
                onClick={handleSourceClick}
                style={{ cursor: 'pointer' }}
              >
                {index + 1}. {safeDisplayPath} (관련도: {safeScore}%)
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

export default memo(ChatMessage);

