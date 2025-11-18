import { useEffect, useState } from 'react';
import { sanitizeHTML } from '../../utils/sanitize';
import './Toast.css';

const TOAST_ICONS = {
  success: '✓',
  error: '✕',
  warning: '⚠',
  info: 'ℹ',
};

/**
 * Toast 컴포넌트
 * @param {Object} props
 * @param {string} props.message - 표시할 메시지
 * @param {string} props.type - 타입 (success, error, warning, info)
 * @param {number} props.duration - 표시 시간 (ms)
 * @param {Function} props.onClose - 닫기 콜백
 */
export default function Toast({ message, type = 'info', duration = 5000, onClose }) {
  const [isVisible, setIsVisible] = useState(true);

  useEffect(() => {
    const timer = setTimeout(() => {
      setIsVisible(false);
      setTimeout(() => {
        onClose?.();
      }, 300); // 애니메이션 시간
    }, duration);

    return () => clearTimeout(timer);
  }, [duration, onClose]);

  if (!isVisible) return null;

  const safeMessage = sanitizeHTML(message);
  const icon = TOAST_ICONS[type] || TOAST_ICONS.info;

  return (
    <div className={`toast ${type} ${!isVisible ? 'hiding' : ''}`} role="alert" aria-live="assertive">
      <span className="toast-icon" aria-hidden="true">
        {icon}
      </span>
      <div className="toast-content" dangerouslySetInnerHTML={{ __html: safeMessage }} />
      <button
        className="toast-close"
        aria-label="닫기"
        onClick={() => {
          setIsVisible(false);
          setTimeout(() => onClose?.(), 300);
        }}
      >
        ×
      </button>
    </div>
  );
}

/**
 * Toast 컨테이너 컴포넌트
 */
export function ToastContainer() {
  const [toasts, setToasts] = useState([]);

  // 전역 toast 함수 등록
  useEffect(() => {
    window.showToast = (message, type = 'info', duration = 5000) => {
      const id = Date.now();
      setToasts((prev) => [...prev, { id, message, type, duration }]);
    };

    return () => {
      delete window.showToast;
    };
  }, []);

  const handleClose = (id) => {
    setToasts((prev) => prev.filter((toast) => toast.id !== id));
  };

  return (
    <div id="toastContainer" className="toast-container" aria-live="polite" aria-atomic="true">
      {toasts.map((toast) => (
        <Toast
          key={toast.id}
          message={toast.message}
          type={toast.type}
          duration={toast.duration}
          onClose={() => handleClose(toast.id)}
        />
      ))}
    </div>
  );
}

