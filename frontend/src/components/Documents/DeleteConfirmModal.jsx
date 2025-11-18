import Modal from '../common/Modal';
import { sanitizeHTML } from '../../utils/sanitize';

/**
 * 문서 삭제 확인 모달
 * @param {Object} props
 * @param {boolean} props.isOpen - 모달 열림 상태
 * @param {Function} props.onClose - 닫기 핸들러
 * @param {Function} props.onConfirm - 확인 핸들러
 * @param {string} props.sourceFile - 삭제할 문서 파일 경로
 * @param {boolean} props.loading - 삭제 진행 중 여부
 */
export default function DeleteConfirmModal({
  isOpen,
  onClose,
  onConfirm,
  sourceFile,
  loading = false,
}) {
  const fileName = sourceFile ? sourceFile.split('\\').pop().split('/').pop() : '';
  const safeFileName = sanitizeHTML(fileName);

  const footer = (
    <div className="modal-actions">
      <button className="btn-cancel" onClick={onClose} disabled={loading}>
        취소
      </button>
      <button
        className="btn-confirm"
        onClick={onConfirm}
        disabled={loading}
        aria-busy={loading}
      >
        {loading ? '삭제 중...' : '삭제'}
      </button>
    </div>
  );

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title="문서 삭제 확인"
      footer={footer}
    >
      <div id="deleteModalBody">
        <p>정말로 다음 문서를 삭제하시겠습니까?</p>
        <p>
          <strong>{safeFileName}</strong>
        </p>
        <p style={{ color: '#ef4444', fontWeight: 'bold' }}>⚠️ 이 작업은 되돌릴 수 없습니다.</p>
        <p style={{ marginTop: '15px', padding: '10px', background: '#2d2d3a', borderRadius: '5px', border: '1px solid #565869' }}>
          <small style={{ color: '#8e8ea0' }}>
            💡 삭제 시 다음 항목들이 제거됩니다:<br />
            - Qdrant 벡터 데이터<br />
            - 메타데이터<br />
            - 관련 인덱스 정보
          </small>
        </p>
      </div>
    </Modal>
  );
}

