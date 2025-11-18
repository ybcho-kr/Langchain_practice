import { useState } from 'react';

const TABS = [
  { id: 'chat', label: '💬 채팅', icon: '💬' },
  { id: 'documents', label: '📚 문서 관리', icon: '📚' },
  { id: 'settings', label: '⚙️ 검색 설정', icon: '⚙️' },
  { id: 'evidence', label: '🔍 답변 근거', icon: '🔍' },
];

/**
 * 탭 네비게이션 컴포넌트
 * @param {Object} props
 * @param {string} props.activeTab - 현재 활성 탭
 * @param {Function} props.onTabChange - 탭 변경 핸들러
 */
export default function TabNavigation({ activeTab, onTabChange }) {
  return (
    <div className="nav-tabs" role="tablist">
      {TABS.map((tab) => (
        <button
          key={tab.id}
          className={`nav-tab ${activeTab === tab.id ? 'active' : ''}`}
          role="tab"
          aria-selected={activeTab === tab.id}
          aria-controls={tab.id}
          id={`tab-${tab.id}`}
          onClick={() => onTabChange(tab.id)}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}

