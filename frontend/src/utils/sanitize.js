import DOMPurify from 'dompurify';

/**
 * DOMPurify 설정
 */
const DOMPURIFY_CONFIG = {
  ALLOWED_TAGS: [
    'p',
    'br',
    'strong',
    'em',
    'b',
    'i',
    'h2',
    'h3',
    'h4',
    'ol',
    'ul',
    'li',
    'table',
    'thead',
    'tbody',
    'tfoot',
    'tr',
    'th',
    'td',
    'div',
    'span',
    'pre',
    'code',
    'a',
  ],
  ALLOWED_ATTR: ['class', 'style', 'href', 'target', 'rel'],
  ALLOW_DATA_ATTR: false,
};

/**
 * HTML sanitization
 * @param {string} html - Sanitize할 HTML 문자열
 * @returns {string} Sanitized HTML
 */
export function sanitizeHTML(html) {
  if (typeof DOMPurify === 'undefined') {
    console.warn('DOMPurify가 로드되지 않았습니다. HTML이 그대로 표시됩니다.');
    return html;
  }
  return DOMPurify.sanitize(html, DOMPURIFY_CONFIG);
}

/**
 * 텍스트 이스케이프 (XSS 방지)
 * @param {string} text - 이스케이프할 텍스트
 * @returns {string} 이스케이프된 텍스트
 */
export function escapeHTML(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

