import { useState, useEffect } from 'react';
import { useAppStore } from '../../stores/appStore';
import { useDocuments, useDeleteDocument } from '../../hooks/useApi';
import { apiClient } from '../../services/api';
import DocumentUpload from './DocumentUpload';
import DocumentList from './DocumentList';
import DeleteConfirmModal from './DeleteConfirmModal';
import VocabularyView from './VocabularyView';
import Modal from '../common/Modal';
import '../../styles/components/documents.css';

/**
 * 문서 관리 컴포넌트
 */
export default function Documents() {
  const { documents, setDocuments } = useAppStore();
  const [deleteModalOpen, setDeleteModalOpen] = useState(false);
  const [pendingDelete, setPendingDelete] = useState(null);
  const [chunkModalOpen, setChunkModalOpen] = useState(false);
  const [selectedDocumentId, setSelectedDocumentId] = useState(null);
  const [chunks, setChunks] = useState(null);
  const [chunksLoading, setChunksLoading] = useState(false);
  const [fullChunkModalOpen, setFullChunkModalOpen] = useState(false);
  const [selectedChunk, setSelectedChunk] = useState(null);

  const { execute: loadDocuments, loading } = useDocuments();
  const { execute: deleteDocument, loading: deleteLoading } = useDeleteDocument();

  // 문서 목록 로드
  const loadDocumentsList = async () => {
    try {
      const data = await loadDocuments();
      if (data?.documents) {
        setDocuments(data.documents);
      }
    } catch (error) {
      console.error('문서 목록 로드 실패:', error);
      if (window.showToast) {
        window.showToast('문서 목록을 불러오는 중 오류가 발생했습니다.', 'error');
      }
    }
  };

  // 초기 로드
  useEffect(() => {
    loadDocumentsList();
  }, []);

  // 청크 보기
  const handleViewChunks = async (documentId) => {
    setSelectedDocumentId(documentId);
    setChunkModalOpen(true);
    setChunksLoading(true);

    try {
      const data = await apiClient.getDocumentChunks(documentId);
      setChunks(data);
    } catch (error) {
      console.error('청크 조회 실패:', error);
      if (window.showToast) {
        window.showToast('청크 정보를 불러오는 중 오류가 발생했습니다.', 'error');
      }
    } finally {
      setChunksLoading(false);
    }
  };

  // 다운로드 (시뮬레이션)
  const handleDownload = (filename) => {
    if (window.showToast) {
      window.showToast(`다운로드 기능은 구현 예정입니다.\n파일: ${filename}`, 'info');
    }
  };

  // 삭제 확인
  const handleDeleteClick = (sourceFile) => {
    setPendingDelete(sourceFile);
    setDeleteModalOpen(true);
  };

  // 전체 청크 내용 보기
  const handleViewFullChunk = (chunk) => {
    setSelectedChunk(chunk);
    setFullChunkModalOpen(true);
  };

  // 삭제 실행
  const handleDeleteConfirm = async () => {
    if (!pendingDelete) return;

    try {
      const response = await deleteDocument(pendingDelete);

      if (response.success) {
        let message = `문서 삭제 완료!\n\n`;
        message += `✅ Qdrant: ${response.qdrant_deleted ? '삭제됨' : '실패'}\n`;
        message += `✅ BM25: ${response.bm25_deleted ? '삭제됨' : '실패'}\n`;
        message += `⚠️ FAISS: ${response.faiss_handled || response.faiss_deleted ? '처리됨 (재구축 필요)' : '재구축 필요'}\n`;

        if (response.warnings && response.warnings.length > 0) {
          message += `\n⚠️ 경고:\n`;
          response.warnings.forEach((warning) => {
            message += `- ${warning}\n`;
          });
        }

        if (window.showToast) {
          window.showToast(message, 'success', 8000);
        }

        // 문서 목록 새로고침
        await loadDocumentsList();

        // 모달 닫기
        setDeleteModalOpen(false);
        setPendingDelete(null);
      } else {
        if (window.showToast) {
          window.showToast(`문서 삭제 실패!\n${response.message}`, 'error');
        }
      }
    } catch (error) {
      console.error('문서 삭제 실패:', error);
      const errorMsg = error.message || '문서 삭제 중 오류가 발생했습니다.';
      if (window.showToast) {
        window.showToast(errorMsg, 'error');
      }
    } finally {
      setDeleteModalOpen(false);
      setPendingDelete(null);
    }
  };

  return (
    <div className="documents-container">
      <h2>📚 문서 관리</h2>

      <DocumentUpload onUploadComplete={loadDocumentsList} />

      <div className="documents-section" style={{ marginBottom: '20px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px' }}>
          <h3>📋 업로드된 문서</h3>
          <VocabularyView />
        </div>
        <DocumentList
          documents={documents}
          loading={loading}
          onViewChunks={handleViewChunks}
          onDownload={handleDownload}
          onDelete={handleDeleteClick}
        />
      </div>

      <DeleteConfirmModal
        isOpen={deleteModalOpen}
        onClose={() => {
          setDeleteModalOpen(false);
          setPendingDelete(null);
        }}
        onConfirm={handleDeleteConfirm}
        sourceFile={pendingDelete}
        loading={deleteLoading}
      />

      {/* 청크 보기 모달 */}
      <Modal
        isOpen={chunkModalOpen}
        onClose={() => {
          setChunkModalOpen(false);
          setSelectedDocumentId(null);
          setChunks(null);
        }}
        title={`문서 청크 정보 - ${selectedDocumentId ? selectedDocumentId.split('\\').pop() : ''}`}
      >
        {chunksLoading ? (
          <div className="loading">
            <div className="spinner"></div>
            <span>청크 정보를 불러오는 중...</span>
          </div>
        ) : chunks ? (
          <div>
            <div className="modal-chunk-info">
              <strong>문서:</strong> {selectedDocumentId}
              <br />
              <strong>총 청크 수:</strong> {chunks.total_chunks}개
            </div>
            {chunks.chunks && chunks.chunks.length > 0 ? (
              <div>
                {chunks.chunks.map((chunk) => {
                  const isTableData = chunk.content_preview?.includes('표 데이터') || false;
                  return (
                    <div key={chunk.chunk_id} className="chunk-item">
                      <div className="chunk-header">
                        <div className="chunk-title">청크 {chunk.chunk_index}</div>
                        <div className="chunk-meta">
                          ID: {chunk.chunk_id} | 길이: {chunk.content_length}자
                          {isTableData && ' | 📊 표 데이터'}
                        </div>
                      </div>
                      <div className="chunk-content">
                        <pre>{chunk.content_preview}</pre>
                      </div>
                      <div className="chunk-actions">
                        <button
                          className="btn btn-primary btn-sm"
                          onClick={() => handleViewFullChunk(chunk)}
                        >
                          📖 자세히 보기
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div>청크가 없습니다.</div>
            )}
          </div>
        ) : (
          <div>청크 정보를 불러올 수 없습니다.</div>
        )}
      </Modal>

      {/* 전체 청크 내용 모달 */}
      <Modal
        isOpen={fullChunkModalOpen}
        onClose={() => {
          setFullChunkModalOpen(false);
          setSelectedChunk(null);
        }}
        title={selectedChunk ? `청크 ${selectedChunk.chunk_index} 전체 내용` : '청크 전체 내용'}
      >
        {selectedChunk ? (
          <div>
            <div className="modal-chunk-info">
              <strong>청크 ID:</strong> {selectedChunk.chunk_id}
              <br />
              <strong>청크 인덱스:</strong> {selectedChunk.chunk_index}
              <br />
              <strong>내용 길이:</strong> {selectedChunk.content_length}자
              <br />
              <strong>타입:</strong>{' '}
              {selectedChunk.content_preview?.includes('표 데이터') ? '📊 표 데이터' : '📝 일반 텍스트'}
            </div>
            <div className="full-chunk-content">
              <pre style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                {selectedChunk.content_full || selectedChunk.content_preview}
              </pre>
            </div>
          </div>
        ) : (
          <div>청크 정보를 불러올 수 없습니다.</div>
        )}
      </Modal>
    </div>
  );
}

