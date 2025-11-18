"""
로깅 모듈
1단계 개발용 로깅 설정 및 관리
"""

from loguru import logger
from pathlib import Path
from typing import Optional
from src.utils.config import get_logging_config


class LoggerManager:
    """로깅 관리자"""
    
    def __init__(self):
        self._initialized = False
    
    def setup_logging(self, config: Optional[dict] = None):
        """로깅 설정"""
        if self._initialized:
            return
        
        if config is None:
            config = get_logging_config()
        
        # 기존 핸들러 제거
        logger.remove()
        
        # 콘솔 핸들러 추가
        logger.add(
            sink=lambda msg: print(msg, end=""),
            level=config.level,
            format=config.format,
            colorize=True
        )
        
        # 파일 핸들러 추가
        log_file = Path(config.file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            sink=config.file,
            level=config.level,
            format=config.format,
            rotation="10 MB",
            retention="7 days",
            compression="zip"
        )
        
        self._initialized = True
        logger.info("로깅 시스템이 초기화되었습니다.")
    
    def get_logger(self):
        """로거 반환"""
        if not self._initialized:
            try:
                self.setup_logging()
            except Exception:
                # 설정 파일이 없거나 로드 실패 시 기본 로거 반환
                logger.remove()
                logger.add(lambda msg: None, level="INFO")
                self._initialized = True
        return logger


# 전역 로거 관리자
logger_manager = LoggerManager()


def setup_logging(config: Optional[dict] = None):
    """로깅 설정"""
    logger_manager.setup_logging(config)


def get_logger():
    """로거 반환"""
    return logger_manager.get_logger()


# 편의 함수들
def log_info(message: str, **kwargs):
    """정보 로그"""
    logger.info(message, **kwargs)


def log_warning(message: str, **kwargs):
    """경고 로그"""
    logger.warning(message, **kwargs)


def log_error(message: str, **kwargs):
    """에러 로그"""
    logger.error(message, **kwargs)


def log_debug(message: str, **kwargs):
    """디버그 로그"""
    logger.debug(message, **kwargs)
