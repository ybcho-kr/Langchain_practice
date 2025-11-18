import { memo } from 'react';
import { sanitizeHTML } from '../../utils/sanitize';

/**
 * 문서 카드 컴포넌트
 * @param {Object} props
 * @param {Object} props.document - 문서 객체
 * @param {Function} props.onViewChunks - 청크 보기 핸들러
 * @param {Function} props.onDownload - 다운로드 핸들러
 * @param {Function} props.onDelete - 삭제 핸들러
 */
function DocumentCard({ document, onViewChunks, onDownload, onDelete }) {
  const fileName = document.source_file.split('\\').pop().split('/').pop();
  const safeFileName = sanitizeHTML(fileName);
  const safeFilePath = sanitizeHTML(document.source_file);

  return (
    <div className="document-card">
      <div className="document-title">{safeFileName}</div>
      <div className="document-info">파일 경로: {safeFilePath}</div>
      <div className="document-stats">
        <div className="stat">
          <div className="stat-value">{document.total_chunks}</div>
          <div className="stat-label">청크 수</div>
        </div>
        <div className="stat">
          <div className="stat-value">
            {document.first_chunk_index}-{document.last_chunk_index}
          </div>
          <div className="stat-label">인덱스 범위</div>
        </div>
      </div>
      <div className="document-actions">
        <button className="btn btn-primary" onClick={() => onViewChunks(document.source_file)}>
          청크 보기
        </button>
        <button className="btn btn-secondary" onClick={() => onDownload(document.source_file)}>
          다운로드
        </button>
        <button className="btn btn-danger" onClick={() => onDelete(document.source_file)}>
          🗑️ 삭제
        </button>
      </div>
    </div>
  );
}

export default memo(DocumentCard);

