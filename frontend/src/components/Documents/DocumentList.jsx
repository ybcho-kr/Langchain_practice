import { memo } from 'react';
import DocumentCard from './DocumentCard';

/**
 * 문서 목록 컴포넌트
 * @param {Object} props
 * @param {Array} props.documents - 문서 배열
 * @param {boolean} props.loading - 로딩 상태
 * @param {Function} props.onViewChunks - 청크 보기 핸들러
 * @param {Function} props.onDownload - 다운로드 핸들러
 * @param {Function} props.onDelete - 삭제 핸들러
 */
function DocumentList({ documents, loading, onViewChunks, onDownload, onDelete }) {
  if (loading) {
    return (
      <div className="documents-grid" aria-live="polite" aria-busy="true">
        <div className="loading" role="status" aria-live="polite">
          <div className="spinner" aria-hidden="true"></div>
          <span>문서 목록을 불러오는 중...</span>
        </div>
      </div>
    );
  }

  if (!documents || documents.length === 0) {
    return (
      <div className="documents-grid" aria-live="polite">
        <div className="error">업로드된 문서가 없습니다.</div>
      </div>
    );
  }

  return (
    <div className="documents-grid" aria-live="polite" aria-busy="false">
      {documents.map((doc) => (
        <DocumentCard
          key={doc.source_file}
          document={doc}
          onViewChunks={onViewChunks}
          onDownload={onDownload}
          onDelete={onDelete}
        />
      ))}
    </div>
  );
}

export default memo(DocumentList);

