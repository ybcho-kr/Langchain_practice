"""
재시도 로직 모듈
지수 백오프 재시도, 서킷 브레이커 등을 제공합니다.
"""

import asyncio
import time
from typing import Callable, Any, Optional, Type, Tuple, List
from functools import wraps
from dataclasses import dataclass
from src.utils.logger import get_logger


@dataclass
class RetryConfig:
    """재시도 설정"""
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retry_on: Tuple[Type[Exception], ...] = (Exception,)
    retry_on_condition: Optional[Callable[[Exception], bool]] = None


class CircuitBreaker:
    """서킷 브레이커"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ):
        """
        Args:
            failure_threshold: 연속 실패 횟수 임계값
            recovery_timeout: 복구 대기 시간 (초)
            expected_exception: 예상되는 예외 타입
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "closed"  # 'closed', 'open', 'half_open'
        self.logger = get_logger()
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """함수 호출 (서킷 브레이커 적용)"""
        if self.state == "open":
            # 복구 시간 확인
            if self.last_failure_time and (time.time() - self.last_failure_time) > self.recovery_timeout:
                self.state = "half_open"
                self.logger.info("서킷 브레이커: half-open 상태로 전환 (복구 시도)")
            else:
                raise Exception(f"서킷 브레이커가 열려 있습니다. {self.recovery_timeout}초 후 재시도하세요.")
        
        try:
            result = func(*args, **kwargs)
            # 성공 시 상태 리셋
            if self.state == "half_open":
                self.state = "closed"
                self.failure_count = 0
                self.logger.info("서킷 브레이커: closed 상태로 복구")
            return result
        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                self.logger.warning(f"서킷 브레이커: open 상태로 전환 (연속 {self.failure_count}회 실패)")
            
            raise
    
    async def acall(self, func: Callable, *args, **kwargs) -> Any:
        """비동기 함수 호출 (서킷 브레이커 적용)"""
        if self.state == "open":
            # 복구 시간 확인
            if self.last_failure_time and (time.time() - self.last_failure_time) > self.recovery_timeout:
                self.state = "half_open"
                self.logger.info("서킷 브레이커: half-open 상태로 전환 (복구 시도)")
            else:
                raise Exception(f"서킷 브레이커가 열려 있습니다. {self.recovery_timeout}초 후 재시도하세요.")
        
        try:
            result = await func(*args, **kwargs)
            # 성공 시 상태 리셋
            if self.state == "half_open":
                self.state = "closed"
                self.failure_count = 0
                self.logger.info("서킷 브레이커: closed 상태로 복구")
            return result
        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                self.logger.warning(f"서킷 브레이커: open 상태로 전환 (연속 {self.failure_count}회 실패)")
            
            raise


def retry_with_backoff(
    config: Optional[RetryConfig] = None,
    circuit_breaker: Optional[CircuitBreaker] = None
):
    """
    재시도 데코레이터 (지수 백오프)
    
    Args:
        config: 재시도 설정 (None이면 기본값 사용)
        circuit_breaker: 서킷 브레이커 (선택적)
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            logger = get_logger()
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    # 서킷 브레이커 적용
                    if circuit_breaker:
                        return circuit_breaker.call(func, *args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                except config.retry_on as e:
                    last_exception = e
                    
                    # 재시도 조건 확인
                    if config.retry_on_condition and not config.retry_on_condition(e):
                        raise
                    
                    # 마지막 시도면 예외 발생
                    if attempt == config.max_attempts - 1:
                        logger.error(f"재시도 실패 ({config.max_attempts}회 시도): {str(e)}")
                        raise
                    
                    # 지수 백오프 계산
                    delay = min(
                        config.initial_delay * (config.exponential_base ** attempt),
                        config.max_delay
                    )
                    
                    # 지터 추가 (선택적)
                    if config.jitter:
                        import random
                        delay = delay * (0.5 + random.random() * 0.5)
                    
                    logger.warning(
                        f"재시도 {attempt + 1}/{config.max_attempts}: {delay:.2f}초 후 재시도 "
                        f"(오류: {str(e)})"
                    )
                    time.sleep(delay)
                except Exception as e:
                    # 재시도 대상이 아닌 예외는 즉시 발생
                    raise
            
            # 이론적으로 도달 불가능
            if last_exception:
                raise last_exception
            raise Exception("재시도 로직 오류")
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            logger = get_logger()
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    # 서킷 브레이커 적용
                    if circuit_breaker:
                        return await circuit_breaker.acall(func, *args, **kwargs)
                    else:
                        return await func(*args, **kwargs)
                except config.retry_on as e:
                    last_exception = e
                    
                    # 재시도 조건 확인
                    if config.retry_on_condition and not config.retry_on_condition(e):
                        raise
                    
                    # 마지막 시도면 예외 발생
                    if attempt == config.max_attempts - 1:
                        logger.error(f"재시도 실패 ({config.max_attempts}회 시도): {str(e)}")
                        raise
                    
                    # 지수 백오프 계산
                    delay = min(
                        config.initial_delay * (config.exponential_base ** attempt),
                        config.max_delay
                    )
                    
                    # 지터 추가 (선택적)
                    if config.jitter:
                        import random
                        delay = delay * (0.5 + random.random() * 0.5)
                    
                    logger.warning(
                        f"재시도 {attempt + 1}/{config.max_attempts}: {delay:.2f}초 후 재시도 "
                        f"(오류: {str(e)})"
                    )
                    await asyncio.sleep(delay)
                except Exception as e:
                    # 재시도 대상이 아닌 예외는 즉시 발생
                    raise
            
            # 이론적으로 도달 불가능
            if last_exception:
                raise last_exception
            raise Exception("재시도 로직 오류")
        
        # 비동기 함수인지 확인
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def retry_ollama_api(
    max_attempts: int = 3,
    initial_delay: float = 2.0,
    max_delay: float = 10.0
):
    """
    Ollama API 전용 재시도 데코레이터
    
    Args:
        max_attempts: 최대 재시도 횟수
        initial_delay: 초기 지연 시간 (초)
        max_delay: 최대 지연 시간 (초)
    """
    from requests.exceptions import RequestException, Timeout, ConnectionError
    
    config = RetryConfig(
        max_attempts=max_attempts,
        initial_delay=initial_delay,
        max_delay=max_delay,
        exponential_base=2.0,
        jitter=True,
        retry_on=(RequestException, Timeout, ConnectionError, Exception)
    )
    
    return retry_with_backoff(config=config)

