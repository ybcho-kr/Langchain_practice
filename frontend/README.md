# 전기설비 진단 RAG 시스템 - React 프론트엔드

## 개요

기존 단일 HTML 파일(`web_interface.html`)을 React 기반 모던 프론트엔드 아키텍처로 리팩토링한 프로젝트입니다.

## 기술 스택

- **React 18**: UI 라이브러리
- **Vite**: 빌드 도구 및 개발 서버
- **Zustand**: 상태 관리
- **Axios**: HTTP 클라이언트
- **DOMPurify**: XSS 방지

## 프로젝트 구조

```
frontend/
├── public/              # 정적 파일
├── src/
│   ├── components/      # React 컴포넌트
│   │   ├── common/      # 공통 컴포넌트 (Header, TabNavigation, Toast, Modal)
│   │   ├── Chat/        # 채팅 관련 컴포넌트
│   │   ├── Documents/   # 문서 관리 컴포넌트
│   │   ├── Settings/   # 검색 설정 컴포넌트
│   │   └── Evidence/   # 답변 근거 컴포넌트
│   ├── services/        # API 서비스 및 저장소
│   ├── stores/          # Zustand 상태 관리
│   ├── utils/           # 유틸리티 함수
│   ├── hooks/           # 커스텀 훅
│   ├── styles/          # CSS 스타일
│   ├── App.jsx          # 메인 App 컴포넌트
│   └── main.jsx         # 엔트리 포인트
├── package.json
├── vite.config.js
└── .env
```

## 시작하기

### 설치

```bash
cd frontend
npm install
```

### 개발 서버 실행

```bash
npm run dev
```

프론트엔드는 `http://localhost:5173`에서 실행됩니다.

### 빌드

```bash
npm run build
```

빌드 결과물은 `dist/` 디렉토리에 생성됩니다.

### 미리보기

```bash
npm run preview
```

## 주요 기능

### 1. 채팅 기능
- 질의응답 인터페이스
- 모델 선택 및 관리
- 마크다운 렌더링 (표 지원)
- 참조 문서 표시

### 2. 문서 관리
- 문서 업로드 (MD, DOCX, TXT, PDF)
- 문서 목록 조회
- 문서 삭제
- 청크 정보 확인

### 3. 검색 설정
- 검색기 선택 (Qdrant, FAISS, BM25)
- 가중치 조정
- 리랭커 설정
- 검색 파라미터 설정
- 설정 프리셋

### 4. 답변 근거
- 질문/답변 표시
- 참조 문서 상세 정보
- 관련도 점수 표시

## 환경 변수

`.env` 파일에서 다음 변수를 설정할 수 있습니다:

```
VITE_API_BASE_URL=http://localhost:8000
```

## API 통신

프론트엔드는 `http://localhost:8000`의 FastAPI 서버와 통신합니다.

Vite 개발 서버는 프록시를 통해 API 요청을 자동으로 전달합니다.

## 상태 관리

Zustand를 사용하여 전역 상태를 관리합니다:

- `evidence`: 답변 근거 정보
- `model`: 현재 선택된 LLM 모델
- `settings`: 검색 설정
- `documents`: 문서 목록

설정은 localStorage에 자동 저장되어 새로고침 후에도 유지됩니다.

## 성능 최적화

- React.memo를 사용한 컴포넌트 메모이제이션
- useMemo를 사용한 메시지 리스트 최적화
- useCallback을 사용한 이벤트 핸들러 최적화

## 브라우저 호환성

- Chrome (최신 버전)
- Firefox (최신 버전)
- Edge (최신 버전)
- Safari (최신 버전)

## 라이선스

이 프로젝트는 기존 RAG 시스템의 일부입니다.

