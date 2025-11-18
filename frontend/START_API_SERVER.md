# API 서버 시작 가이드

## 🔴 현재 상황

프론트엔드가 API 서버에 연결할 수 없습니다. API 서버를 먼저 시작해야 합니다.

## ✅ 해결 방법

### 방법 1: 직접 실행 (권장)

```powershell
# 프로젝트 루트 디렉토리에서 실행
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 방법 2: main.py 사용

```powershell
# 프로젝트 루트 디렉토리에서 실행
python main.py --mode dev --port 8000
```

### 방법 3: start_servers.py 사용

```powershell
# 프로젝트 루트 디렉토리에서 실행
python start_servers.py
```

## 📋 실행 확인

API 서버가 정상적으로 시작되면 다음과 같은 메시지가 표시됩니다:

```
🌐 API 서버 시작 중...
📍 로컬 주소: http://localhost:8000
🌍 네트워크 주소: http://[본인IP]:8000
📚 API 문서: http://localhost:8000/docs
```

## 🧪 테스트

브라우저에서 다음 주소로 접속하여 확인:

- **헬스 체크**: http://localhost:8000/health
- **API 문서**: http://localhost:8000/docs
- **모델 목록**: http://localhost:8000/models

## ⚠️ 주의사항

1. **포트 충돌**: 8000 포트가 이미 사용 중이면 다른 포트를 사용하거나 기존 프로세스를 종료하세요.
2. **가상환경**: `venv` 가상환경이 활성화되어 있는지 확인하세요.
3. **의존성**: `requirements_stage1.txt`의 패키지가 모두 설치되어 있는지 확인하세요.

## 🚀 프론트엔드와 함께 실행

### 터미널 1: API 서버
```powershell
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 터미널 2: 프론트엔드
```powershell
cd frontend
npm run dev
```

이제 두 서버가 모두 실행되면 프론트엔드가 API 서버에 정상적으로 연결됩니다.

