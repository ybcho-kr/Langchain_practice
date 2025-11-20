"""
LLM 모델 클라이언트
Ollama 서버를 통한 LLM 모델 관리
langchain-ollama를 사용하여 표준화된 인터페이스 제공
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time
import requests

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from src.utils.langchain_utils import create_chat_ollama, convert_to_langchain_messages

from src.utils.logger import get_logger, log_info, log_error
from src.utils.config import get_llm_config
from src.utils.helpers import is_general_question
from src.utils.retry import retry_ollama_api, RetryConfig, CircuitBreaker


@dataclass
class LLMResponse:
    """LLM 응답"""
    text: str
    model: str
    processing_time: float
    tokens_used: Optional[int] = None
    is_general: bool = False  # 일반 답변 여부
    has_rag_context: bool = False  # RAG 컨텍스트 포함 여부


class OllamaLLMClient:
    """Ollama LLM 클라이언트"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = get_logger()
        
        if config is None:
            config = get_llm_config()
        
        # 딕셔너리와 ModelConfig 객체 모두 처리
        if isinstance(config, dict):
            self.model_name = config.get('name', 'gemma3:12b')
            self.base_url = config.get('base_url', 'http://localhost:11434')
            self.max_tokens = config.get('max_tokens', 1000)
            self.temperature = config.get('temperature', 0.1)
            self.top_p = config.get('top_p', 0.9)
            self.timeout = config.get('timeout', 300)  # 기본값 300초 (5분)
        else:
            # ModelConfig 객체인 경우
            self.model_name = config.name
            self.base_url = config.base_url
            self.max_tokens = config.max_tokens
            self.temperature = config.temperature
            self.top_p = config.top_p
            self.timeout = getattr(config, 'timeout', 300)  # 기본값 300초 (5분)
        
        # Ollama GPU 최적화 환경 변수 체크 및 경고
        self._check_ollama_gpu_optimization()
        
        # Ollama 프로세스 개수 확인
        process_count = self._check_ollama_processes()
        if process_count > 1:
            self.logger.warning("=" * 80)
            self.logger.warning(f"⚠️  Ollama 프로세스가 {process_count}개 실행 중입니다!")
            self.logger.warning("   여러 프로세스가 실행 중이면 환경 변수 설정이 일관되지 않을 수 있습니다.")
            self.logger.warning("   GPU 최적화를 위해 모든 Ollama 프로세스를 종료하고 다시 시작하세요:")
            self.logger.warning("")
            self.logger.warning("   PowerShell에서:")
            self.logger.warning("   # 모든 Ollama 프로세스 종료")
            self.logger.warning("   Stop-Process -Name ollama -Force")
            self.logger.warning("")
            self.logger.warning("   # GPU 최적화 환경 변수 설정")
            self.logger.warning("   $env:OLLAMA_FLASH_ATTENTION=\"1\"")
            self.logger.warning("   $env:OLLAMA_NUM_GPU=\"1\"")
            self.logger.warning("   $env:OLLAMA_KV_OFFLOAD=\"0\"")
            self.logger.warning("   $env:OLLAMA_NUM_THREADS=\"4\"")
            self.logger.warning("")
            self.logger.warning("   # Ollama 서버 재시작")
            self.logger.warning("   ollama serve")
            self.logger.warning("=" * 80)
        elif process_count == 1:
            self.logger.info("✅ Ollama 프로세스가 1개만 실행 중입니다.")
        else:
            self.logger.warning("⚠️  Ollama 프로세스가 실행되지 않았습니다. Ollama 서버를 시작하세요.")
        
        # langchain-ollama ChatOllama 인스턴스는 필요 시 생성 (파라미터가 동적이므로)
        # self.llm은 사용하지 않고, 필요 시마다 ChatOllama 인스턴스를 생성
        
        # 서킷 브레이커 초기화 (Ollama API 전용)
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60.0,
            expected_exception=Exception
        )
        
        # 모델 정보 캐시 (동적 조회용)
        self._model_info_cache: Optional[Dict[str, Any]] = None
        
        self.logger.info(
            f"Ollama LLM 클라이언트 초기화 (langchain-ollama): {self.model_name}, "
            f"base_url={self.base_url}, timeout={self.timeout}초"
        )
    
    def _check_ollama_gpu_optimization(self):
        """Ollama GPU 최적화 환경 변수 체크 및 경고"""
        # 권장 환경 변수 목록
        recommended_vars = {
            'OLLAMA_FLASH_ATTENTION': '1',  # Flash Attention 활성화 (메모리 효율성 및 속도 향상)
            'OLLAMA_NUM_GPU': '1',  # GPU 1개 사용
            'OLLAMA_KV_OFFLOAD': '0',  # KV 캐시 오프로드 비활성화 (성능 우선)
            'OLLAMA_NUM_THREADS': '4',  # 스레드 수 제한 (CPU 코어 수에 맞춰 조정)
        }
        
        missing_vars = []
        wrong_vars = []
        
        for var_name, recommended_value in recommended_vars.items():
            current_value = os.environ.get(var_name)
            
            if current_value is None:
                missing_vars.append(var_name)
            elif current_value != recommended_value:
                wrong_vars.append((var_name, current_value, recommended_value))
        
        # 경고 메시지 출력
        if missing_vars or wrong_vars:
            self.logger.warning("=" * 80)
            self.logger.warning("⚠️  Ollama GPU 최적화 환경 변수가 설정되지 않았거나 권장값과 다릅니다!")
            self.logger.warning("   GPU 성능을 최대로 활용하려면 Ollama 서버 시작 전에 다음을 설정하세요:")
            self.logger.warning("")
            self.logger.warning("   PowerShell에서:")
            self.logger.warning("   $env:OLLAMA_FLASH_ATTENTION=\"1\"")
            self.logger.warning("   $env:OLLAMA_NUM_GPU=\"1\"")
            self.logger.warning("   $env:OLLAMA_KV_OFFLOAD=\"0\"")
            self.logger.warning("   $env:OLLAMA_NUM_THREADS=\"4\"")
            self.logger.warning("")
            self.logger.warning("   또는 시스템 환경 변수로 설정:")
            self.logger.warning("   [System.Environment]::SetEnvironmentVariable('OLLAMA_FLASH_ATTENTION', '1', 'User')")
            self.logger.warning("")
            
            if missing_vars:
                self.logger.warning(f"   누락된 변수: {', '.join(missing_vars)}")
            if wrong_vars:
                for var_name, current, recommended in wrong_vars:
                    self.logger.warning(f"   {var_name}: 현재={current}, 권장={recommended}")
            
            self.logger.warning("=" * 80)
        else:
            self.logger.info("✅ Ollama GPU 최적화 환경 변수가 올바르게 설정되었습니다.")
    
    def _check_ollama_processes(self) -> int:
        """Ollama 프로세스 개수 확인 (Windows)"""
        try:
            import subprocess
            # PowerShell 명령어로 ollama 프로세스 개수 확인
            result = subprocess.run(
                ['powershell', '-Command', 'Get-Process ollama -ErrorAction SilentlyContinue | Measure-Object | Select-Object -ExpandProperty Count'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                count = int(result.stdout.strip())
                return count
            return 0
        except Exception as e:
            # 실패 시 경고 없이 0 반환 (체크 실패는 치명적이지 않음)
            self.logger.debug(f"Ollama 프로세스 체크 실패: {str(e)}")
            return 0
    
    def _check_server_status(self) -> bool:
        """Ollama 서버 상태 확인"""
        try:
            # langchain-ollama는 내부적으로 연결 상태를 확인하므로 간단히 체크
            # 실제 사용 시점에 에러가 발생하면 처리됨
            return True
        except Exception as e:
            self.logger.error(f"Ollama 서버 연결 실패: {str(e)}")
            return False
    
    def get_model_info(self, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """
        Ollama API를 통해 모델 정보 조회 (컨텍스트 윈도우 크기 등)
        
        Args:
            use_cache: 캐시된 정보 사용 여부 (기본값: True)
            
        Returns:
            모델 정보 딕셔너리 (context_length, parameter_size 등 포함) 또는 None
        """
        if use_cache and self._model_info_cache is not None:
            return self._model_info_cache
        
        try:
            # Ollama API /api/show 엔드포인트 호출
            api_url = f"{self.base_url}/api/show"
            response = requests.post(
                api_url,
                json={"name": self.model_name},
                timeout=10
            )
            
            if response.status_code == 200:
                # 응답이 JSON인지 확인
                try:
                    model_info = response.json()
                    # 딕셔너리인지 확인
                    if not isinstance(model_info, dict):
                        self.logger.warning(
                            f"모델 정보가 딕셔너리가 아닙니다. 타입: {type(model_info)}, "
                            f"값: {str(model_info)[:200]}"
                        )
                        return None
                    
                    self._model_info_cache = model_info
                    self.logger.debug(f"모델 정보 조회 성공: {self.model_name}")
                    return model_info
                except ValueError as e:
                    # JSON 파싱 실패 (텍스트 형식일 수 있음)
                    self.logger.warning(
                        f"모델 정보 JSON 파싱 실패: {str(e)}, "
                        f"응답 타입: {type(response.text)}, "
                        f"응답 내용: {response.text[:200]}"
                    )
                    return None
            else:
                self.logger.warning(
                    f"모델 정보 조회 실패: HTTP {response.status_code}, "
                    f"응답: {response.text[:200]}"
                )
                return None
                
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"Ollama API 호출 실패 (모델 정보 조회): {str(e)}")
            return None
        except Exception as e:
            self.logger.warning(f"모델 정보 조회 중 오류: {str(e)}")
            return None
    
    def get_context_window_size(self) -> Optional[int]:
        """
        모델의 실제 컨텍스트 윈도우 크기 조회
        
        Returns:
            컨텍스트 윈도우 크기 (토큰 수) 또는 None (조회 실패 시)
        """
        model_info = self.get_model_info()
        if model_info is None:
            return None
        
        # 딕셔너리인지 확인
        if not isinstance(model_info, dict):
            self.logger.warning(
                f"모델 정보가 딕셔너리가 아닙니다. 타입: {type(model_info)}"
            )
            return None
        
        try:
            # Ollama 모델 정보에서 컨텍스트 윈도우 크기 추출
            # 여러 가능한 키 확인
            context_size = None
            
            # 1. modelfile.parameter.num_ctx 확인
            if 'modelfile' in model_info:
                modelfile = model_info.get('modelfile')
                if isinstance(modelfile, dict) and 'parameter' in modelfile:
                    parameter = modelfile.get('parameter')
                    if isinstance(parameter, dict):
                        context_size = parameter.get('num_ctx') or parameter.get('context_length')
            
            # 2. details.context_length 확인
            if context_size is None and 'details' in model_info:
                details = model_info.get('details')
                if isinstance(details, dict):
                    context_size = details.get('context_length')
            
            # 3. parameter_size 확인
            if context_size is None:
                context_size = model_info.get('parameter_size')
            
            # 4. modelfile 텍스트에서 num_ctx 파싱 시도 (텍스트 형식인 경우)
            if context_size is None and 'modelfile' in model_info:
                modelfile = model_info.get('modelfile')
                if isinstance(modelfile, str):
                    # Modelfile 텍스트에서 num_ctx 파싱
                    import re
                    match = re.search(r'num_ctx\s+(\d+)', modelfile)
                    if match:
                        context_size = match.group(1)
            
            # 문자열로 반환되는 경우 정수로 변환
            if context_size is not None:
                try:
                    if isinstance(context_size, str):
                        # "8192" 같은 문자열 처리
                        context_size = int(context_size)
                    return int(context_size)
                except (ValueError, TypeError):
                    self.logger.warning(f"컨텍스트 윈도우 크기를 숫자로 변환할 수 없음: {context_size}")
                    return None
            
            return None
            
        except Exception as e:
            self.logger.warning(f"컨텍스트 윈도우 크기 추출 중 오류: {str(e)}")
            return None
    
    def get_effective_max_tokens(self) -> int:
        """
        실제 사용 가능한 max_tokens 반환 (동적 조회 시도)
        
        Returns:
            설정 파일의 max_tokens 또는 모델의 컨텍스트 윈도우 크기 중 작은 값
        """
        # 먼저 모델의 실제 컨텍스트 윈도우 크기 조회 시도
        context_window = self.get_context_window_size()
        
        if context_window is not None:
            # 컨텍스트 윈도우 크기를 찾았으면, 설정 파일의 max_tokens와 비교하여 작은 값 사용
            # (컨텍스트 윈도우는 입력+출력 전체이므로, 출력용으로 일부 여유를 둠)
            effective_max = min(self.max_tokens, context_window - 100)  # 100 토큰 여유
            if effective_max != self.max_tokens:
                self.logger.debug(
                    f"모델 컨텍스트 윈도우 크기 ({context_window}) 기반으로 "
                    f"max_tokens 조정: {self.max_tokens} → {effective_max}"
                )
            return effective_max
        
        # 모델 정보 조회 실패 시 설정 파일 값 사용
        return self.max_tokens
    
    def generate_text(self, prompt: str, **kwargs) -> Optional[LLMResponse]:
        """텍스트 생성 (langchain-ollama 사용)"""
        try:
            # langchain-ollama를 사용하여 텍스트 생성
            # HumanMessage로 단일 프롬프트 전달
            start_time = time.time()
            
            # kwargs에서 파라미터 추출 (있으면 사용, 없으면 기본값)
            temperature = kwargs.get('temperature', self.temperature)
            top_p = kwargs.get('top_p', self.top_p)
            max_tokens = kwargs.get('max_tokens', self.max_tokens)
            
            # LangChain 유틸리티를 사용하여 ChatOllama 인스턴스 생성
            self.logger.debug(f"ChatOllama 생성: 모델={self.model_name}, num_predict={max_tokens}, temperature={temperature}, top_p={top_p}, timeout={self.timeout}초")
            llm = create_chat_ollama(
                model_name=self.model_name,
                base_url=self.base_url,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                timeout=self.timeout,
            )
            
            if not llm:
                self.logger.error("ChatOllama 인스턴스 생성 실패")
                return None
            
            # HumanMessage로 프롬프트 전달
            response = llm.invoke([HumanMessage(content=prompt)])
            processing_time = time.time() - start_time
            
            generated_text = response.content if hasattr(response, 'content') else str(response)
            
            # 사용된 토큰 수 추출 (response.response_metadata에 있을 수 있음)
            tokens_used = None
            if hasattr(response, 'response_metadata'):
                usage = response.response_metadata.get('usage', {})
                tokens_used = usage.get('total_tokens') or usage.get('eval_count')
            
            self.logger.debug(f"텍스트 생성 완료: {len(generated_text)}자, {processing_time:.2f}초")
            
            return LLMResponse(
                text=generated_text,
                model=self.model_name,
                processing_time=processing_time,
                tokens_used=tokens_used
            )
                
        except Exception as e:
            self.logger.error(f"텍스트 생성 중 오류: {str(e)}")
            return None
    
    def generate_answer(self, question: str, context: str = "", **kwargs) -> str:
        """질문에 대한 답변 생성 (문자열 반환 - 호환성 유지)"""
        result = self.generate_answer_with_metadata(question, context, **kwargs)
        return result.text if result else "죄송합니다. 답변을 생성할 수 없습니다."
    
    def _get_system_prompt(self, has_rag_context: bool) -> str:
        """
        시스템 프롬프트 생성 (백업 버전 복원)
        
        Args:
            has_rag_context: RAG 컨텍스트 포함 여부
            
        Returns:
            시스템 프롬프트 문자열
        """
        if has_rag_context:
            # RAG 컨텍스트가 있는 경우
            return """당신은 전기설비 진단 전문가입니다. 제공된 전문 지식을 바탕으로 질문에 정확하고 간결하게 답변해야 합니다.


#답변 작성 지침
1. 답변은 한국어로 작성하고, 전문적이면서도 이해하기 쉽게 설명하세요.
2. 구조화된 형태로 답변하세요 (제목, 목록, 단계별 설명 등).
3. 질문에 직접적으로 답변하는 핵심 내용만 포함하세요.
4. 불필요한 반복이나 중복 정보는 제외하세요.
5. 구체적인 수치나 기준이 있다면 명확히 제시하세요.
6. 답변은 간결하고 명확하게 작성하세요.
7. 질문과 관련 없는 부가 설명은 최소화하세요.
8. 표 데이터가 존재하면 테이블 처리를 해주세요.
9. /no_thinking,/nothink,/no think,/no_thinking"""

        else:
            # 일반 답변 또는 전문 답변 (컨텍스트 없음)
            return """당신은 전기설비 진단 전문가입니다. 질문에 정확하고 상세하게 답변해야 합니다.
/no_thinking
/nothink
/no think
/no_thinking

#답변 작성 지침
1. 답변은 한국어로 작성하고, 전문적이면서도 이해하기 쉽게 설명하세요.
2. 구조화된 형태로 답변하세요 (제목, 목록, 단계별 설명 등).
3. 질문에 직접적으로 답변하는 핵심 내용만 포함하세요.
4. 불필요한 반복이나 중복 정보는 제외하세요.
5. 구체적인 수치나 기준이 있다면 명확히 제시하세요.
6. 답변은 간결하고 명확하게 작성하세요.
7. 질문과 관련 없는 부가 설명은 최소화하세요.
8. 표 데이터가 존재하면 테이블 처리를 해주세요.
9. /no_thinking,/nothink,/no think,/no_thinking"""

    def _get_user_prompt(self, question: str, context: str = "") -> str:
        """
        유저 프롬프트 생성 (백업 버전 복원)
        
        Args:
            question: 사용자 질문
            context: RAG 컨텍스트
            
        Returns:
            유저 프롬프트 문자열
        """
        if context:
            # RAG 컨텍스트가 있는 경우
            return f"""다음은 전기설비 진단에 관한 전문 지식입니다:

{context}

#질문
{question}

위의 전문 지식을 바탕으로 질문에 정확하고 간결하게 답변해주세요."""
        else:
            # 컨텍스트가 없는 경우
            return question

    def generate_answer_with_metadata(self, question: str, context: str = "", **kwargs) -> Optional[LLMResponse]:
        """
        질문에 대한 답변 생성 (메타데이터 포함) - Chat API 사용 (백업 버전 복원)
        
        Args:
            question: 사용자 질문
            context: RAG 컨텍스트
            **kwargs: 추가 파라미터
        """
        # 일반적인 질문인지 확인
        is_general = is_general_question(question) and not context
        has_rag_context = bool(context and context.strip())
        
        # 시스템 프롬프트와 유저 프롬프트 분리 (직접 생성)
        system_prompt = self._get_system_prompt(has_rag_context)
        user_prompt = self._get_user_prompt(question, context)
        
        # Chat API를 사용하여 메시지 전송
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self._generate_chat(messages, **kwargs)
        
        if response:
            # 메타데이터 추가
            response.is_general = is_general
            response.has_rag_context = has_rag_context
            self.logger.info(f"답변 생성 완료: 일반답변={is_general}, RAG컨텍스트={'예' if has_rag_context else '아니오'}")
            return response
        else:
            return None

    def _generate_chat(self, messages: List[Dict[str, str]], **kwargs) -> Optional[LLMResponse]:
        """Chat API를 사용한 텍스트 생성 (langchain-ollama ChatOllama 사용, 재시도 로직 포함)"""
        if not self._check_server_status():
            self.logger.error("Ollama 서버가 실행되지 않았습니다")
            return None
        
        # 재시도 로직 적용
        @retry_ollama_api(max_attempts=3, initial_delay=2.0, max_delay=10.0)
        def _call_ollama():
            return self._generate_chat_internal(messages, **kwargs)
        
        try:
            return _call_ollama()
        except Exception as e:
            self.logger.error(f"Ollama API 호출 최종 실패: {str(e)}")
            return None
    
    def _generate_chat_internal(self, messages: List[Dict[str, str]], **kwargs) -> Optional[LLMResponse]:
        """Chat API 내부 구현 (재시도 로직 제외)"""
        try:
            # kwargs에서 파라미터 추출
            temperature = kwargs.get('temperature', self.temperature)
            top_p = kwargs.get('top_p', self.top_p)
            max_tokens = kwargs.get('max_tokens', self.max_tokens)
            
            # LangChain 유틸리티를 사용하여 ChatOllama 인스턴스 생성
            llm = create_chat_ollama(
                model_name=self.model_name,
                base_url=self.base_url,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                timeout=self.timeout,
            )
            
            if not llm:
                self.logger.error("ChatOllama 인스턴스 생성 실패")
                return None
            
            # LangChain 유틸리티를 사용하여 메시지 변환
            langchain_messages = convert_to_langchain_messages(messages)
            
            if not langchain_messages:
                self.logger.error("메시지 변환 실패")
                return None
            
            start_time = time.time()
            response = llm.invoke(langchain_messages)
            processing_time = time.time() - start_time
            
            generated_text = response.content if hasattr(response, 'content') else str(response)
            
            # 사용된 토큰 수 추출
            tokens_used = None
            if hasattr(response, 'response_metadata'):
                usage = response.response_metadata.get('usage', {})
                tokens_used = usage.get('total_tokens') or usage.get('eval_count')
            
            self.logger.debug(f"Chat API 텍스트 생성 완료 (langchain-ollama): {len(generated_text)}자, {processing_time:.2f}초")
            
            return LLMResponse(
                text=generated_text,
                model=self.model_name,
                processing_time=processing_time,
                tokens_used=tokens_used
            )
                
        except Exception as e:
            self.logger.error(f"Chat API 텍스트 생성 중 오류: {str(e)}")
            # Chat API 실패 시 기존 generate_text로 폴백
            self.logger.warning("Chat API 실패, 기존 generate_text로 폴백")
            # 메시지를 단일 프롬프트로 변환
            prompt_parts = []
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                if role == 'system':
                    prompt_parts.append(f"시스템 지시사항:\n{content}")
                elif role == 'user':
                    prompt_parts.append(f"사용자 질문:\n{content}")
            prompt = "\n\n".join(prompt_parts)
            return self.generate_text(prompt, **kwargs)
    
    def generate_summary(self, text: str, max_length: int = 200, **kwargs) -> str:
        """텍스트 요약"""
        prompt = f"""다음 텍스트를 {max_length}자 이내로 요약해주세요:

{text}

요약:"""
        
        response = self.generate_text(prompt, **kwargs)
        
        if response:
            return response.text.strip()
        else:
            return "요약을 생성할 수 없습니다."
    
    def generate_explanation(self, term: str, context: str = "", **kwargs) -> str:
        """용어 설명 생성"""
        if context:
            prompt = f"""다음은 전기설비 진단 관련 문서입니다:

{context}

위 문서의 맥락에서 '{term}'에 대해 자세히 설명해주세요."""
        else:
            prompt = f"""전기설비 진단 전문가로서 '{term}'에 대해 자세히 설명해주세요."""
        
        response = self.generate_text(prompt, **kwargs)
        
        if response:
            return response.text.strip()
        else:
            return f"'{term}'에 대한 설명을 생성할 수 없습니다."
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Optional[LLMResponse]:
        """채팅 형태의 대화 (Chat API 사용)"""
        return self._generate_chat(messages, **kwargs)
    
    # ========== 비동기 메서드 (Phase 1: LLM 호출 비동기화) ==========
    
    async def generate_text_async(self, prompt: str, **kwargs) -> Optional[LLMResponse]:
        """비동기 텍스트 생성 (langchain-ollama 사용)"""
        try:
            start_time = time.time()
            
            # kwargs에서 파라미터 추출
            temperature = kwargs.get('temperature', self.temperature)
            top_p = kwargs.get('top_p', self.top_p)
            max_tokens = kwargs.get('max_tokens', self.max_tokens)
            
            # LangChain 유틸리티를 사용하여 ChatOllama 인스턴스 생성
            llm = create_chat_ollama(
                model_name=self.model_name,
                base_url=self.base_url,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                timeout=self.timeout,
            )
            
            if not llm:
                self.logger.error("ChatOllama 인스턴스 생성 실패")
                return None
            
            # 비동기 호출
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            processing_time = time.time() - start_time
            
            generated_text = response.content if hasattr(response, 'content') else str(response)
            
            # 사용된 토큰 수 추출
            tokens_used = None
            if hasattr(response, 'response_metadata'):
                usage = response.response_metadata.get('usage', {})
                tokens_used = usage.get('total_tokens') or usage.get('eval_count')
            
            self.logger.debug(f"비동기 텍스트 생성 완료: {len(generated_text)}자, {processing_time:.2f}초")
            
            return LLMResponse(
                text=generated_text,
                model=self.model_name,
                processing_time=processing_time,
                tokens_used=tokens_used
            )
                
        except Exception as e:
            self.logger.error(f"비동기 텍스트 생성 중 오류: {str(e)}")
            return None
    
    async def generate_answer_async(self, question: str, context: str = "", **kwargs) -> Optional[LLMResponse]:
        """
        비동기 질문에 대한 답변 생성 (메타데이터 포함) - 백업 버전 복원
        
        Args:
            question: 사용자 질문
            context: RAG 컨텍스트
            **kwargs: 추가 파라미터
        """
        # 일반적인 질문인지 확인
        is_general = is_general_question(question) and not context
        has_rag_context = bool(context and context.strip())
        
        # 시스템 프롬프트와 유저 프롬프트 분리 (직접 생성)
        system_prompt = self._get_system_prompt(has_rag_context)
        user_prompt = self._get_user_prompt(question, context)
        
        # Chat API를 사용하여 메시지 전송
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = await self._generate_chat_async(messages, **kwargs)
        
        if response:
            # 메타데이터 추가
            response.is_general = is_general
            response.has_rag_context = has_rag_context
            self.logger.info(f"비동기 답변 생성 완료: 일반답변={is_general}, RAG컨텍스트={'예' if has_rag_context else '아니오'}")
            return response
        else:
            return None
    
    async def _generate_chat_async(self, messages: List[Dict[str, str]], **kwargs) -> Optional[LLMResponse]:
        """비동기 Chat API를 사용한 텍스트 생성 (재시도 로직 포함)"""
        if not self._check_server_status():
            self.logger.error("Ollama 서버가 실행되지 않았습니다")
            return None
        
        # 재시도 로직 적용
        from src.utils.retry import retry_with_backoff, RetryConfig
        
        retry_config = RetryConfig(
            max_attempts=3,
            initial_delay=2.0,
            max_delay=10.0,
            exponential_base=2.0,
            jitter=True,
            retry_on=(Exception,)
        )
        
        @retry_with_backoff(config=retry_config, circuit_breaker=self.circuit_breaker)
        async def _call_ollama_async():
            return await self._generate_chat_async_internal(messages, **kwargs)
        
        try:
            return await _call_ollama_async()
        except Exception as e:
            self.logger.error(f"Ollama API 비동기 호출 최종 실패: {str(e)}")
            return None
    
    async def _generate_chat_async_internal(self, messages: List[Dict[str, str]], **kwargs) -> Optional[LLMResponse]:
        """비동기 Chat API 내부 구현 (재시도 로직 제외)"""
        try:
            # kwargs에서 파라미터 추출
            temperature = kwargs.get('temperature', self.temperature)
            top_p = kwargs.get('top_p', self.top_p)
            max_tokens = kwargs.get('max_tokens', self.max_tokens)
            
            # LangChain 유틸리티를 사용하여 ChatOllama 인스턴스 생성
            self.logger.debug(
                f"ChatOllama 생성 (비동기): 모델={self.model_name}, num_predict={max_tokens} "
                f"(설정값: {self.max_tokens}), temperature={temperature}, top_p={top_p}, timeout={self.timeout}초"
            )
            llm = create_chat_ollama(
                model_name=self.model_name,
                base_url=self.base_url,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                timeout=self.timeout,
            )
            
            if not llm:
                self.logger.error("ChatOllama 인스턴스 생성 실패")
                return None
            
            # LangChain 유틸리티를 사용하여 메시지 변환
            langchain_messages = convert_to_langchain_messages(messages)
            
            if not langchain_messages:
                self.logger.error("메시지 변환 실패")
                return None
            
            # 프롬프트 크기 확인
            total_prompt_length = sum(len(msg.get('content', '')) for msg in messages)
            estimated_prompt_tokens = int(total_prompt_length / 2.5)  # 한국어 기준 토큰 추정
            self.logger.debug(
                f"Ollama API 호출 시작: 모델={self.model_name}, num_predict={max_tokens}, "
                f"프롬프트 길이={total_prompt_length}자 (추정 {estimated_prompt_tokens}토큰), timeout={self.timeout}초"
            )
            
            start_time = time.time()
            # 비동기 호출
            response = await llm.ainvoke(langchain_messages)
            processing_time = time.time() - start_time
            
            generated_text = response.content if hasattr(response, 'content') else str(response)
            
            # 사용된 토큰 수 추출 (다양한 경로 시도)
            tokens_used = None
            prompt_tokens = None
            completion_tokens = None
            
            # 1. response_metadata에서 추출 시도
            if hasattr(response, 'response_metadata'):
                metadata = response.response_metadata
                self.logger.debug(f"response_metadata 내용: {metadata}")
                
                if isinstance(metadata, dict):
                    usage = metadata.get('usage', {})
                    if isinstance(usage, dict):
                        tokens_used = (
                            usage.get('total_tokens') or 
                            usage.get('eval_count') or
                            usage.get('prompt_eval_count')
                        )
                        prompt_tokens = usage.get('prompt_tokens') or usage.get('prompt_eval_count')
                        completion_tokens = usage.get('completion_tokens') or usage.get('eval_count')
            
            # 2. response 객체의 다른 속성 확인
            if tokens_used is None:
                # response 객체의 모든 속성 확인 (디버깅용)
                response_attrs = dir(response)
                self.logger.debug(f"response 객체 속성: {[attr for attr in response_attrs if not attr.startswith('_')]}")
                
                # usage 속성 직접 확인
                if hasattr(response, 'usage'):
                    usage_attr = getattr(response, 'usage')
                    self.logger.debug(f"response.usage: {usage_attr}")
                    if isinstance(usage_attr, dict):
                        tokens_used = usage_attr.get('total_tokens') or usage_attr.get('eval_count')
            
            # 3. 생성된 텍스트로부터 추정 토큰 수 계산 (한국어 기준: 1토큰 ≈ 2.5자)
            estimated_output_tokens = int(len(generated_text) / 2.5) if generated_text else 0
            estimated_input_tokens = estimated_prompt_tokens
            
            # 토큰 정보 로깅
            tokens_per_sec = estimated_output_tokens / processing_time if processing_time > 0 else 0
            
            if tokens_used:
                self.logger.info(
                    f"비동기 Chat API 텍스트 생성 완료: {len(generated_text)}자, {processing_time:.2f}초, "
                    f"토큰={tokens_used} (입력: {prompt_tokens or estimated_input_tokens}, 출력: {completion_tokens or estimated_output_tokens}), "
                    f"속도={tokens_per_sec:.1f}토큰/초, 모델={self.model_name}"
                )
            else:
                # 토큰 정보가 없으면 추정값 사용
                total_estimated_tokens = estimated_input_tokens + estimated_output_tokens
                actual_used = estimated_output_tokens if estimated_output_tokens < max_tokens else f"{max_tokens}(제한)"
                self.logger.info(
                    f"비동기 Chat API 텍스트 생성 완료: {len(generated_text)}자, {processing_time:.2f}초, "
                    f"토큰=추정 {total_estimated_tokens} (입력: {estimated_input_tokens}, 출력: {estimated_output_tokens}), "
                    f"속도={tokens_per_sec:.1f}토큰/초, 모델={self.model_name}, "
                    f"num_predict={max_tokens} (실제 생성: {actual_used})"
                )
                tokens_used = total_estimated_tokens
            
            return LLMResponse(
                text=generated_text,
                model=self.model_name,
                processing_time=processing_time,
                tokens_used=tokens_used
            )
                
        except Exception as e:
            import traceback
            self.logger.error(f"비동기 Chat API 텍스트 생성 중 오류: {str(e)}")
            self.logger.error(f"상세 오류 정보:\n{traceback.format_exc()}")
            
            # Ollama 서버 연결 실패인지 확인
            error_str = str(e).lower()
            if 'connection' in error_str or 'timeout' in error_str or 'refused' in error_str:
                self.logger.error(f"Ollama 서버 연결 실패: {self.base_url}, 모델: {self.model_name}")
                return None
            
            # Chat API 실패 시 기존 generate_text_async로 폴백
            self.logger.warning("비동기 Chat API 실패, 기존 generate_text_async로 폴백")
            try:
                # 메시지를 단일 프롬프트로 변환
                prompt_parts = []
                for msg in messages:
                    role = msg.get('role', 'user')
                    content = msg.get('content', '')
                    if role == 'system':
                        prompt_parts.append(f"시스템 지시사항:\n{content}")
                    elif role == 'user':
                        prompt_parts.append(f"사용자 질문:\n{content}")
                prompt = "\n\n".join(prompt_parts)
                fallback_result = await self.generate_text_async(prompt, **kwargs)
                if fallback_result:
                    return fallback_result
                else:
                    self.logger.error("폴백 generate_text_async도 실패했습니다.")
                    return None
            except Exception as fallback_error:
                self.logger.error(f"폴백 generate_text_async 중 오류: {str(fallback_error)}")
                return None


def create_llm_client(config: Optional[Dict[str, Any]] = None) -> OllamaLLMClient:
    """LLM 클라이언트 생성"""
    return OllamaLLMClient(config)


def generate_answer(question: str, context: str = "", config: Optional[Dict[str, Any]] = None) -> str:
    """질문에 대한 답변 생성 (편의 함수)"""
    client = create_llm_client(config)
    return client.generate_answer(question, context)
