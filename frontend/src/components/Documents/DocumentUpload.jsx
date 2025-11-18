import { useState } from 'react';
import { useUploadDocuments } from '../../hooks/useApi';
import { useAppStore } from '../../stores/appStore';

/**
 * 문서 업로드 컴포넌트
 */
export default function DocumentUpload({ onUploadComplete }) {
  const [files, setFiles] = useState([]);
  const [forceUpdate, setForceUpdate] = useState(false);
  const [replaceMode, setReplaceMode] = useState(false);
  const [uploadStatus, setUploadStatus] = useState({ message: '', type: '', visible: false });

  const { execute: uploadDocuments, loading } = useUploadDocuments();
  const { setDocuments } = useAppStore();

  const handleFileChange = (e) => {
    const selectedFiles = Array.from(e.target.files);
    setFiles(selectedFiles);
  };

  const handleUpload = async () => {
    if (files.length === 0) {
      setUploadStatus({
        message: '파일을 선택해주세요.',
        type: 'error',
        visible: true,
      });
      setTimeout(() => setUploadStatus({ ...uploadStatus, visible: false }), 5000);
      return;
    }

    const formData = new FormData();
    files.forEach((file) => {
      formData.append('files', file);
    });
    formData.append('force_update', forceUpdate);
    formData.append('replace_mode', replaceMode);

    try {
      const response = await uploadDocuments(formData);

      if (response.success) {
        const message = `업로드 완료! 처리된 파일: ${response.processed_files}개, 생성된 청크: ${response.total_chunks}개, 처리 시간: ${response.processing_time.toFixed(2)}초`;
        setUploadStatus({
          message,
          type: 'success',
          visible: true,
        });

        // 파일 입력 초기화
        setFiles([]);
        const fileInput = document.getElementById('fileInput');
        if (fileInput) fileInput.value = '';

        // 문서 목록 새로고침
        if (onUploadComplete) {
          onUploadComplete();
        }

        if (window.showToast) {
          window.showToast(message, 'success');
        }
      } else {
        setUploadStatus({
          message: `업로드 실패: ${response.message}`,
          type: 'error',
          visible: true,
        });
      }
    } catch (error) {
      console.error('업로드 실패:', error);
      const errorMsg =
        error.message || '업로드 중 오류가 발생했습니다. 다시 시도해주세요.';
      setUploadStatus({
        message: errorMsg,
        type: 'error',
        visible: true,
      });

      if (window.showToast) {
        window.showToast(errorMsg, 'error');
      }
    } finally {
      setTimeout(() => setUploadStatus({ ...uploadStatus, visible: false }), 5000);
    }
  };

  return (
    <div className="upload-section">
      <h3>📤 문서 업로드</h3>
      <div className="upload-form">
        <input
          type="file"
          id="fileInput"
          multiple
          accept=".md,.docx,.txt,.pdf"
          aria-label="문서 파일 선택"
          onChange={handleFileChange}
          style={{ marginBottom: '15px' }}
        />

        <div className="upload-options">
          <label className="option-label">
            <input
              type="checkbox"
              id="forceUpdate"
              checked={forceUpdate}
              onChange={(e) => setForceUpdate(e.target.checked)}
              aria-describedby="forceUpdate-desc"
            />
            강제 업데이트 (파일 변경 여부 무시)
          </label>
          <label className="option-label">
            <input
              type="checkbox"
              id="replaceMode"
              checked={replaceMode}
              onChange={(e) => setReplaceMode(e.target.checked)}
              aria-describedby="replaceMode-desc"
            />
            완전 교체 모드 (모든 파일에 대해 기존 벡터 삭제 후 새로 추가)
          </label>
          <div className="info-text" id="uploadInfo">
            <small>💡 같은 이름의 파일이 있으면 자동으로 교체 모드로 처리됩니다.</small>
          </div>
        </div>

        <button
          onClick={handleUpload}
          className="upload-btn"
          aria-label="문서 업로드"
          aria-busy={loading}
          disabled={loading || files.length === 0}
        >
          {loading ? '업로드 중...' : '문서 업로드'}
        </button>
      </div>

      {uploadStatus.visible && (
        <div
          id="uploadStatus"
          className={`upload-status ${uploadStatus.type}`}
          style={{ display: 'block' }}
        >
          {uploadStatus.message}
        </div>
      )}
    </div>
  );
}

