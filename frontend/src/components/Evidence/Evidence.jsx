import { useEffect, useState } from 'react';
import { useAppStore } from '../../stores/appStore';
import SourceCard from './SourceCard';
import Modal from '../common/Modal';
import { sanitizeHTML } from '../../utils/sanitize';
import { apiClient } from '../../services/api';
import '../../styles/components/evidence.css';

/**
 * 답변 근거 컴포넌트
 */
export default function Evidence() {
  const { evidence } = useAppStore();
  const [fullSourceModalOpen, setFullSourceModalOpen] = useState(false);
  const [selectedSource, setSelectedSource] = useState(null);
  const [fullContent, setFullContent] = useState(null);
  const [loadingFullContent, setLoadingFullContent] = useState(false);

  // 전체 청크 내용 로드
  const loadFullChunkContent = async (source) => {
    if (!source.source_file) {
      // source_file이 없으면 현재 content 사용
      setFullContent(source.content);
      return;
    }

    setLoadingFullContent(true);
    try {
      // 문서의 청크 목록 가져오기
      const chunksData = await apiClient.getDocumentChunks(source.source_file);
      
      if (chunksData && chunksData.chunks) {
        // chunk_index로 해당 청크 찾기
        const targetChunk = chunksData.chunks.find(
          (chunk) => chunk.chunk_index === source.chunk_index
        );
        
        if (targetChunk && targetChunk.content_full) {
          setFullContent(targetChunk.content_full);
        } else if (targetChunk && targetChunk.content_preview) {
          // content_full이 없으면 content_preview 사용
          setFullContent(targetChunk.content_preview);
        } else {
          // 청크를 찾지 못한 경우 현재 content 사용
          setFullContent(source.content);
        }
      } else {
        // 청크 목록을 가져오지 못한 경우 현재 content 사용
        setFullContent(source.content);
      }
    } catch (error) {
      console.error('전체 청크 내용 로드 실패:', error);
      // 에러 발생 시 현재 content 사용
      setFullContent(source.content);
    } finally {
      setLoadingFullContent(false);
    }
  };

  // 자세히 보기 핸들러
  const handleViewFull = (source) => {
    setSelectedSource(source);
    setFullSourceModalOpen(true);
    setFullContent(null);
    loadFullChunkContent(source);
  };

  // Evidence 탭 표시 이벤트 리스너 (부모 컴포넌트에서 탭 전환 처리)
  useEffect(() => {
    const handleShowEvidence = () => {
      // 탭 전환은 부모 컴포넌트에서 처리
      window.dispatchEvent(new CustomEvent('switchTab', { detail: 'evidence' }));
    };

    window.addEventListener('showEvidence', handleShowEvidence);
    return () => {
      window.removeEventListener('showEvidence', handleShowEvidence);
    };
  }, []);

  if (!evidence) {
    return (
      <div className="evidence-container">
          <div className="evidence-header">
            <h2>🔍 답변 근거 확인</h2>
            <p>채팅에서 답변을 받은 후, 해당 답변의 근거를 자세히 확인할 수 있습니다.</p>
          </div>
          <div id="evidenceContent">
            <div className="loading">
              <p>채팅에서 질문을 먼저 해주세요.</p>
            </div>
        </div>
      </div>
    );
  }

  const safeQuestion = sanitizeHTML(evidence.question);
  const safeAnswer = sanitizeHTML(evidence.answer);
  const safeModel = sanitizeHTML(evidence.model_used || 'N/A');
  const confidence = ((evidence.confidence || 0) * 100).toFixed(1);
  const processingTime = (evidence.processing_time || 0).toFixed(2);

  return (
    <div className="evidence-container">
        <div className="evidence-header">
          <h2>🔍 답변 근거 확인</h2>
          <p>채팅에서 답변을 받은 후, 해당 답변의 근거를 자세히 확인할 수 있습니다.</p>
        </div>
        <div id="evidenceContent">
          <div className="evidence-header">
            <div className="evidence-question">질문: {safeQuestion}</div>
            <div className="evidence-answer">답변: {safeAnswer}</div>
            <div style={{ marginTop: '15px', fontSize: '0.9em', color: '#666' }}>
              신뢰도: {confidence}% | 처리시간: {processingTime}초 | 모델: {safeModel}
            </div>
          </div>
          <div className="evidence-sources">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <h3>📚 답변 근거 문서들 ({evidence.sources?.length || 0}개)</h3>
              {evidence.sources && evidence.sources.length > 0 && (
                <div style={{ fontSize: '0.9em', color: '#6c757d' }}>
                  평균 관련도: {(
                    evidence.sources.reduce((sum, s) => sum + (s.relevance_score || s.score || 0), 0) / evidence.sources.length * 100
                  ).toFixed(1)}%
                </div>
              )}
            </div>
            {evidence.sources && evidence.sources.length > 0 ? (
              <div>
                {/* 관련도별 정렬 옵션 */}
                <div style={{ marginBottom: '12px', display: 'flex', gap: '8px', alignItems: 'center' }}>
                  <span style={{ fontSize: '0.9em', color: '#6c757d' }}>정렬:</span>
                  <select
                    id="sourceSort"
                    onChange={(e) => {
                      const sorted = [...evidence.sources].sort((a, b) => {
                        const scoreA = a.relevance_score || a.score || 0;
                        const scoreB = b.relevance_score || b.score || 0;
                        return e.target.value === 'desc' ? scoreB - scoreA : scoreA - scoreB;
                      });
                      // 정렬된 소스로 evidence 업데이트 (간단한 구현)
                      window.location.reload(); // 실제로는 상태 관리 필요
                    }}
                    style={{
                      padding: '4px 8px',
                      fontSize: '0.85em',
                      border: '1px solid #dee2e6',
                      borderRadius: '4px',
                    }}
                  >
                    <option value="desc">관련도 높은 순</option>
                    <option value="asc">관련도 낮은 순</option>
                  </select>
                </div>
                {evidence.sources.map((source, index) => (
                  <SourceCard
                    key={index}
                    source={source}
                    index={index}
                    onViewFull={() => handleViewFull(source)}
                  />
                ))}
              </div>
            ) : (
              <div style={{ padding: '20px', textAlign: 'center', color: '#6c757d' }}>
                근거 문서가 없습니다.
              </div>
            )}
          </div>
        </div>

        {/* 전체 소스 내용 모달 */}
        <Modal
          isOpen={fullSourceModalOpen}
          onClose={() => {
            setFullSourceModalOpen(false);
            setSelectedSource(null);
            setFullContent(null);
          }}
          title={selectedSource ? `근거 문서 ${selectedSource.source_path || selectedSource.source_file?.split('\\').pop()?.split('/').pop() || '상세'} 전체 내용` : '근거 문서 전체 내용'}
        >
          {selectedSource ? (
            <div>
              <div className="modal-chunk-info">
                {selectedSource.source_file && (
                  <>
                    <strong>파일:</strong> {selectedSource.source_file.split('\\').pop()?.split('/').pop() || selectedSource.source_file}
                    <br />
                  </>
                )}
                {selectedSource.source_path && (
                  <>
                    <strong>경로:</strong> {selectedSource.source_path}
                    <br />
                  </>
                )}
                <strong>청크 인덱스:</strong> {selectedSource.chunk_index ?? 'N/A'}
                <br />
                <strong>관련도:</strong> {((selectedSource.relevance_score || 0) * 100).toFixed(1)}%
                {selectedSource.metadata?.chunk_id && (
                  <>
                    <br />
                    <strong>청크 ID:</strong> {selectedSource.metadata.chunk_id}
                  </>
                )}
              </div>
              {loadingFullContent ? (
                <div className="loading">
                  <div className="spinner"></div>
                  <span>전체 내용을 불러오는 중...</span>
                </div>
              ) : (
                <div className="full-chunk-content">
                  <pre style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                    {fullContent || selectedSource.content}
                  </pre>
                </div>
              )}
            </div>
          ) : (
            <div>소스 정보를 불러올 수 없습니다.</div>
          )}
        </Modal>
      </div>
  );
}

