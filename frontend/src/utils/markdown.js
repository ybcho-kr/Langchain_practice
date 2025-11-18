import { sanitizeHTML } from './sanitize';

/**
 * 마크다운 표를 HTML 표로 변환
 * @param {string} markdown - 마크다운 텍스트
 * @returns {string} HTML 문자열
 */
function convertMarkdownTables(markdown) {
  const lines = markdown.split(/\n/);
  const out = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];
    const next = i + 1 < lines.length ? lines[i + 1] : '';
    const isHeader = /\|/.test(line);
    const isSeparator = /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(next);

    if (isHeader && isSeparator) {
      const headerCells = line.split('|').map((c) => c.trim()).filter((c) => c.length > 0);
      const rows = [];
      i += 2; // skip header and separator

      while (i < lines.length && /\|/.test(lines[i])) {
        const rowCells = lines[i].split('|').map((c) => c.trim()).filter((c) => c.length > 0);
        rows.push(rowCells);
        i++;
      }

      let tableHtml = '<table class="md-table"><thead><tr>';
      headerCells.forEach((h) => {
        tableHtml += `<th>${sanitizeHTML(h)}</th>`;
      });
      tableHtml += '</tr></thead><tbody>';
      rows.forEach((r) => {
        tableHtml += '<tr>' + r.map((c) => `<td>${sanitizeHTML(c)}</td>`).join('') + '</tr>';
      });
      tableHtml += '</tbody></table>';
      out.push(tableHtml);
      continue;
    }

    out.push(line);
    i++;
  }

  return out.join('\n');
}

/**
 * 마크다운을 HTML로 변환 (표 지원)
 * @param {string} markdown - 마크다운 텍스트
 * @returns {string} HTML 문자열
 */
export function markdownToHtml(markdown) {
  if (!markdown) return '';

  // 1) 표 먼저 변환
  let html = convertMarkdownTables(markdown);

  // 2) 인라인/블록 변환
  html = html
    // 굵은 글씨 (**text** -> <strong>text</strong>)
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    // 이탤릭 (*text* -> <em>text</em>)
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    // 제목 (# -> <h2>, ## -> <h3>, ### -> <h3>)
    .replace(/^### (.*$)/gim, '<h3>$1</h3>')
    .replace(/^## (.*$)/gim, '<h3>$1</h3>')
    .replace(/^# (.*$)/gim, '<h2>$1</h2>')
    // 번호 목록 (1. -> <li>)
    .replace(/^(\d+)\. (.*$)/gim, '<li>$2</li>')
    // 줄바꿈을 <br>로 변환
    .replace(/\n/g, '<br>')
    // 연속된 <li>를 <ol>로 감싸기
    .replace(/(<li>.*<\/li>)/g, '<ol>$1</ol>')
    // <ol> 태그 정리
    .replace(/<\/ol><br><ol>/g, '')
    .replace(/<ol><br>/g, '<ol>')
    .replace(/<br><\/ol>/g, '</ol>');

  // 최종 HTML sanitize
  return sanitizeHTML(html);
}

