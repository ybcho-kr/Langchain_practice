import { useState, useEffect } from 'react';
import { useSparseVocabulary } from '../../hooks/useApi';
import Modal from '../common/Modal';
import '../../styles/components/settings.css';

/**
 * Sparse 벡터 Vocabulary 확인 컴포넌트
 */
export default function VocabularyView() {
  const [isOpen, setIsOpen] = useState(false);
  const [limit, setLimit] = useState(100);
  const [searchToken, setSearchToken] = useState('');
  const [vocabularyData, setVocabularyData] = useState(null);
  const { execute: loadVocabulary, loading } = useSparseVocabulary();

  // Vocabulary 로드
  const handleLoadVocabulary = async () => {
    try {
      const params = {};
      if (limit) params.limit = limit;
      if (searchToken.trim()) params.search_token = searchToken.trim();

      const data = await loadVocabulary(params);
      setVocabularyData(data);
    } catch (error) {
      console.error('Vocabulary 조회 실패:', error);
      if (window.showToast) {
        window.showToast('Vocabulary 정보를 불러오는 중 오류가 발생했습니다.', 'error');
      }
    }
  };

  // 모달 열 때 자동 로드
  useEffect(() => {
    if (isOpen && !vocabularyData) {
      handleLoadVocabulary();
    }
  }, [isOpen]);

  // 검색 실행
  const handleSearch = () => {
    handleLoadVocabulary();
  };

  // 초기화
  const handleReset = () => {
    setLimit(100);
    setSearchToken('');
    setVocabularyData(null);
  };

  return (
    <>
      <button className="btn btn-secondary" onClick={() => setIsOpen(true)}>
        📖 Sparse Vocabulary 확인
      </button>

      <Modal
        isOpen={isOpen}
        onClose={() => {
          setIsOpen(false);
          handleReset();
        }}
        title="Sparse 벡터 Vocabulary 정보"
      >
        <div className="vocabulary-view">
          {/* 검색 및 필터 컨트롤 */}
          <div className="settings-preview" style={{ marginBottom: '20px' }}>
            <div style={{ display: 'flex', gap: '10px', marginBottom: '10px', flexWrap: 'wrap' }}>
              <div style={{ flex: '1', minWidth: '200px' }}>
                <label className="settings-label" style={{ display: 'block', marginBottom: '5px' }}>
                  항목 수 제한:
                </label>
                <input
                  type="number"
                  value={limit}
                  onChange={(e) => setLimit(parseInt(e.target.value) || 100)}
                  min="1"
                  max="1000"
                  className="settings-input"
                  style={{ width: '100%' }}
                />
              </div>
              <div style={{ flex: '1', minWidth: '200px' }}>
                <label className="settings-label" style={{ display: 'block', marginBottom: '5px' }}>
                  토큰 검색:
                </label>
                <input
                  type="text"
                  value={searchToken}
                  onChange={(e) => setSearchToken(e.target.value)}
                  placeholder="토큰 검색 (예: 전기)"
                  className="settings-input"
                  style={{ width: '100%' }}
                  onKeyPress={(e) => {
                    if (e.key === 'Enter') {
                      handleSearch();
                    }
                  }}
                />
              </div>
            </div>
            <div style={{ display: 'flex', gap: '10px' }}>
              <button className="btn btn-primary" onClick={handleSearch} disabled={loading}>
                {loading ? '로딩 중...' : '🔍 조회'}
              </button>
              <button className="btn btn-secondary" onClick={handleReset} disabled={loading}>
                초기화
              </button>
            </div>
          </div>

          {/* 로딩 상태 */}
          {loading && (
            <div className="loading" style={{ textAlign: 'center', padding: '20px' }}>
              <div className="spinner"></div>
              <span>Vocabulary 정보를 불러오는 중...</span>
            </div>
          )}

          {/* Vocabulary 정보 */}
          {!loading && vocabularyData && (
            <div>
              {/* 기본 정보 */}
              <div className="settings-preview" style={{ marginBottom: '20px' }}>
                <h4 style={{ marginTop: 0, color: '#ececf1' }}>📊 기본 정보</h4>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '10px' }}>
                  <div style={{ color: '#8e8ea0' }}>
                    <strong style={{ color: '#ececf1' }}>Sparse 벡터 활성화:</strong>{' '}
                    <span style={{ color: vocabularyData.sparse_enabled ? '#4ade80' : '#f87171' }}>
                      {vocabularyData.sparse_enabled ? '✅ 활성화' : '❌ 비활성화'}
                    </span>
                  </div>
                  <div style={{ color: '#8e8ea0' }}>
                    <strong style={{ color: '#ececf1' }}>모델 학습 상태:</strong>{' '}
                    <span style={{ color: vocabularyData.model_trained ? '#4ade80' : '#f87171' }}>
                      {vocabularyData.model_trained ? '✅ 학습됨' : '❌ 미학습'}
                    </span>
                  </div>
                  <div style={{ color: '#8e8ea0' }}>
                    <strong style={{ color: '#ececf1' }}>Corpus 크기:</strong> {vocabularyData.corpus_size?.toLocaleString() || 0}개 문서
                  </div>
                  <div style={{ color: '#8e8ea0' }}>
                    <strong style={{ color: '#ececf1' }}>Vocabulary 크기:</strong> {vocabularyData.vocabulary_size?.toLocaleString() || 0}개 토큰
                  </div>
                  {vocabularyData.avgdl > 0 && (
                    <div style={{ color: '#8e8ea0' }}>
                      <strong style={{ color: '#ececf1' }}>평균 문서 길이:</strong> {vocabularyData.avgdl.toFixed(2)}
                    </div>
                  )}
                </div>
                {vocabularyData.message && (
                  <div className="info-text" style={{ marginTop: '10px' }}>
                    <small>ℹ️ {vocabularyData.message}</small>
                  </div>
                )}
              </div>

              {/* 학습되지 않은 경우 */}
              {!vocabularyData.model_trained && (
                <div className="settings-preview" style={{ padding: '20px', textAlign: 'center', color: '#fbbf24', background: '#2d2d3a', border: '1px solid #fbbf24' }}>
                  <p style={{ color: '#fbbf24' }}>⚠️ {vocabularyData.message || 'Sparse 임베딩 모델이 아직 학습되지 않았습니다.'}</p>
                  <p style={{ color: '#8e8ea0' }}>문서를 먼저 업로드해주세요.</p>
                </div>
              )}

              {/* 통계 정보 */}
              {vocabularyData.model_trained && vocabularyData.statistics && (
                <div className="settings-preview" style={{ marginBottom: '20px' }}>
                  <h4 style={{ marginTop: 0, color: '#ececf1' }}>📈 통계 정보</h4>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '10px' }}>
                    <div style={{ color: '#8e8ea0' }}>
                      <strong style={{ color: '#ececf1' }}>전체 Vocabulary 크기:</strong> {vocabularyData.statistics.total_vocabulary_size?.toLocaleString() || 0}개
                    </div>
                    <div style={{ color: '#8e8ea0' }}>
                      <strong style={{ color: '#ececf1' }}>반환된 항목 수:</strong> {vocabularyData.statistics.returned_count?.toLocaleString() || 0}개
                    </div>
                    <div style={{ color: '#8e8ea0' }}>
                      <strong style={{ color: '#ececf1' }}>IDF 최소값:</strong> {vocabularyData.statistics.min_idf?.toFixed(4) || vocabularyData.statistics.idf_min?.toFixed(4) || '0.0000'}
                    </div>
                    <div style={{ color: '#8e8ea0' }}>
                      <strong style={{ color: '#ececf1' }}>IDF 최대값:</strong> {vocabularyData.statistics.max_idf?.toFixed(4) || vocabularyData.statistics.idf_max?.toFixed(4) || '0.0000'}
                    </div>
                    <div style={{ color: '#8e8ea0' }}>
                      <strong style={{ color: '#ececf1' }}>IDF 평균값:</strong> {vocabularyData.statistics.avg_idf?.toFixed(4) || vocabularyData.statistics.idf_mean?.toFixed(4) || '0.0000'}
                    </div>
                  </div>
                  {vocabularyData.statistics.top_tokens && vocabularyData.statistics.top_tokens.length > 0 && (
                    <div style={{ marginTop: '15px' }}>
                      <strong style={{ color: '#ececf1' }}>🔝 상위 10개 토큰 (IDF 값 기준):</strong>
                      <div style={{ marginTop: '10px', display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                        {vocabularyData.statistics.top_tokens.map((tokenInfo, idx) => {
                          // tokenInfo는 객체: {index, idf, df, token_text, token_word}
                          const tokenIndex = typeof tokenInfo === 'object' ? tokenInfo.index : tokenInfo;
                          const idfValue = typeof tokenInfo === 'object' ? tokenInfo.idf : (vocabularyData.idf_values?.[tokenIndex] || 0);
                          const df = typeof tokenInfo === 'object' ? tokenInfo.df : (vocabularyData.vocabulary?.[tokenIndex]?.document_frequency || 0);
                          const tokenText = typeof tokenInfo === 'object' ? tokenInfo.token_text : null;
                          const tokenWord = typeof tokenInfo === 'object' ? tokenInfo.token_word : null;
                          const displayText = tokenWord || tokenText || tokenIndex;
                          return (
                            <div
                              key={idx}
                              style={{
                                padding: '5px 10px',
                                background: '#40414f',
                                border: '1px solid #565869',
                                borderRadius: '4px',
                                fontSize: '0.9em',
                                color: '#ececf1',
                              }}
                              title={`인덱스: ${tokenIndex}, 단어: ${tokenWord || tokenText || 'N/A'}, IDF: ${idfValue.toFixed(4)}, DF: ${df}`}
                            >
                              <strong>{displayText}</strong> <span style={{ color: '#8e8ea0', fontSize: '0.75em' }}>(#{tokenIndex}, IDF: {idfValue.toFixed(2)})</span>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Vocabulary 목록 */}
              {vocabularyData.model_trained && vocabularyData.vocabulary && Object.keys(vocabularyData.vocabulary).length > 0 && (
                <div>
                  <h4 style={{ color: '#ececf1', marginBottom: '15px' }}>📝 Vocabulary 목록</h4>
                  <div style={{ maxHeight: '400px', overflowY: 'auto', border: '1px solid #565869', borderRadius: '5px', background: '#2d2d3a' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                      <thead style={{ position: 'sticky', top: 0, background: '#40414f', zIndex: 1 }}>
                        <tr>
                          <th style={{ padding: '10px', textAlign: 'left', borderBottom: '2px solid #565869', color: '#ececf1' }}>단어/토큰</th>
                          <th style={{ padding: '10px', textAlign: 'right', borderBottom: '2px solid #565869', color: '#ececf1' }}>인덱스</th>
                          <th style={{ padding: '10px', textAlign: 'right', borderBottom: '2px solid #565869', color: '#ececf1' }}>IDF 값</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(vocabularyData.vocabulary)
                          .sort((a, b) => {
                            const tokenIdxA = a[0];
                            const tokenIdxB = b[0];
                            const idfA = vocabularyData.idf_values?.[tokenIdxA] || 0;
                            const idfB = vocabularyData.idf_values?.[tokenIdxB] || 0;
                            return idfB - idfA; // IDF 값 기준 내림차순
                          })
                          .map(([tokenIndex, vocabInfo]) => {
                            // vocabInfo는 객체: {index, document_frequency, avg_weight, max_weight, min_weight, total_occurrences, token_text, token_word}
                            const tokenIdx = typeof vocabInfo === 'object' ? (vocabInfo.index || tokenIndex) : tokenIndex;
                            const idfValue = vocabularyData.idf_values?.[tokenIndex] || 0;
                            const df = typeof vocabInfo === 'object' ? vocabInfo.document_frequency : 0;
                            const tokenText = typeof vocabInfo === 'object' ? vocabInfo.token_text : null;
                            const tokenWord = typeof vocabInfo === 'object' ? vocabInfo.token_word : null;
                            const displayText = tokenWord || tokenText || `#${tokenIndex}`;
                            return (
                              <tr key={tokenIndex} style={{ borderBottom: '1px solid #565869' }}>
                                <td style={{ padding: '8px 10px', wordBreak: 'break-word', color: '#ececf1' }}>
                                  <strong>{displayText}</strong>
                                  {tokenWord && tokenText && tokenWord !== tokenText && (
                                    <span style={{ color: '#8e8ea0', fontSize: '0.85em', marginLeft: '8px' }}>
                                      ({tokenText})
                                    </span>
                                  )}
                                </td>
                                <td style={{ padding: '8px 10px', textAlign: 'right', fontFamily: 'monospace', color: '#8e8ea0' }}>
                                  {tokenIdx}
                                </td>
                                <td style={{ padding: '8px 10px', textAlign: 'right', fontFamily: 'monospace', color: '#8e8ea0' }}>
                                  {idfValue.toFixed(4)}
                                </td>
                              </tr>
                            );
                          })}
                      </tbody>
                    </table>
                  </div>
                  {vocabularyData.statistics && vocabularyData.statistics.returned_count < vocabularyData.statistics.total_vocabulary_size && (
                    <div className="info-text" style={{ marginTop: '10px' }}>
                      <small>
                        ℹ️ 전체 {vocabularyData.statistics.total_vocabulary_size.toLocaleString()}개 중{' '}
                        {vocabularyData.statistics.returned_count.toLocaleString()}개만 표시됩니다. 더 많은 항목을 보려면 limit 값을 늘려주세요.
                      </small>
                    </div>
                  )}
                </div>
              )}

              {/* 검색 결과가 있는 경우 */}
              {searchToken && vocabularyData.filtered_count !== undefined && (
                <div className="info-text" style={{ marginTop: '15px' }}>
                  <small>🔍 검색 결과: '{searchToken}' 포함 토큰 {vocabularyData.filtered_count}개</small>
                </div>
              )}
            </div>
          )}

          {/* 오류 메시지 */}
          {!loading && vocabularyData && vocabularyData.error && (
            <div className="settings-preview" style={{ padding: '20px', textAlign: 'center', color: '#f87171', background: '#2d2d3a', border: '1px solid #f87171' }}>
              <p style={{ color: '#f87171' }}>❌ 오류 발생: {vocabularyData.error}</p>
              {vocabularyData.message && <p style={{ color: '#8e8ea0' }}>{vocabularyData.message}</p>}
            </div>
          )}
        </div>
      </Modal>
    </>
  );
}

