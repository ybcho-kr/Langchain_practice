import { useState, useRef, useEffect } from 'react';

/**
 * 채팅 입력 컴포넌트
 * @param {Object} props
 * @param {Function} props.onSend - 메시지 전송 핸들러
 * @param {boolean} props.disabled - 비활성화 상태
 */
export default function ChatInput({ onSend, disabled = false }) {
  const [message, setMessage] = useState('');
  const inputRef = useRef(null);

  const handleSubmit = () => {
    const trimmedMessage = message.trim();
    if (trimmedMessage && !disabled) {
      onSend(trimmedMessage);
      setMessage('');
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  // 포커스 관리
  useEffect(() => {
    if (!disabled && inputRef.current) {
      inputRef.current.focus();
    }
  }, [disabled]);

  return (
    <div className="chat-input">
      <input
        ref={inputRef}
        type="text"
        id="chatInput"
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        onKeyPress={handleKeyPress}
        placeholder="질문을 입력하세요..."
        aria-label="질문 입력"
        aria-required="true"
        disabled={disabled}
      />
      <button
        onClick={handleSubmit}
        id="sendBtn"
        aria-label="메시지 전송"
        aria-busy={disabled}
        disabled={disabled || !message.trim()}
      >
        {disabled ? '전송 중...' : '전송'}
      </button>
    </div>
  );
}

