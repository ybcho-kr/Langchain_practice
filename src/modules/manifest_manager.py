"""파일 인덱싱 상태를 관리하는 manifest 관리 유틸리티."""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.utils.logger import get_logger


class ManifestManager:
    """파일 처리 상태를 JSON manifest에 영속화."""

    def __init__(self, manifest_path: Path):
        self.manifest_path = Path(manifest_path)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger()
        self._data: Dict[str, Dict[str, Any]] = {"files": {}}
        self._load()

    def _load(self) -> None:
        if not self.manifest_path.exists():
            return

        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                content = json.load(f)
                if isinstance(content, dict) and "files" in content:
                    self._data = content
        except Exception as e:
            self.logger.error(f"Manifest 로드 실패: {self.manifest_path}, 오류: {str(e)}")

    def _save(self) -> None:
        try:
            with open(self.manifest_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"Manifest 저장 실패: {self.manifest_path}, 오류: {str(e)}")

    def get_entry(self, file_path: Path) -> Optional[Dict[str, Any]]:
        return self._data.get("files", {}).get(str(file_path))

    def update_entry(self, file_path: Path, doc_id: str, content_hash: str, mtime: float) -> None:
        files = self._data.setdefault("files", {})
        files[str(file_path)] = {
            "doc_id": doc_id,
            "content_hash": content_hash,
            "mtime": mtime,
        }
        self._save()

    def remove_entries(self, file_paths: List[str]) -> None:
        files = self._data.setdefault("files", {})
        for path in file_paths:
            files.pop(str(path), None)
        self._save()

    def list_files(self) -> List[str]:
        return list(self._data.get("files", {}).keys())

