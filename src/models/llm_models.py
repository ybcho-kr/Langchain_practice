"""
LLM 모델 클라이언트
Ollama 서버를 통한 LLM 모델 관리
langchain-ollama를 사용하여 표준화된 인터페이스 제공
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from src.utils.logger import get_logger, log_info, log_error
from src.utils.config import get_llm_config
from src.utils.helpers import is_general_question


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
            self.model_name = config.get('name', 'llama3.1:8b')
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
            
            # langchain-ollama ChatOllama 인스턴스 생성 (kwargs 파라미터 반영)
            llm = ChatOllama(
                model=self.model_name,
                base_url=self.base_url,
                temperature=temperature,
                top_p=top_p,
                num_predict=max_tokens,
                timeout=self.timeout,
            )
            
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
        """시스템 프롬프트 생성"""
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
        """유저 프롬프트 생성"""
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
        """질문에 대한 답변 생성 (메타데이터 포함) - Chat API 사용"""
        # 일반적인 질문인지 확인
        is_general = is_general_question(question) and not context
        has_rag_context = bool(context and context.strip())
        
        # 시스템 프롬프트와 유저 프롬프트 분리
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
        """Chat API를 사용한 텍스트 생성 (langchain-ollama ChatOllama 사용)"""
        if not self._check_server_status():
            self.logger.error("Ollama 서버가 실행되지 않았습니다")
            return None
        
        try:
            # kwargs에서 파라미터 추출
            temperature = kwargs.get('temperature', self.temperature)
            top_p = kwargs.get('top_p', self.top_p)
            max_tokens = kwargs.get('max_tokens', self.max_tokens)
            
            # langchain-ollama ChatOllama 인스턴스 생성 (kwargs 파라미터 반영)
            llm = ChatOllama(
                model=self.model_name,
                base_url=self.base_url,
                temperature=temperature,
                top_p=top_p,
                num_predict=max_tokens,
                timeout=self.timeout,
            )
            
            # LangChain 메시지 형식으로 변환
            langchain_messages = []
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                
                if role == 'system':
                    langchain_messages.append(SystemMessage(content=content))
                elif role == 'user':
                    langchain_messages.append(HumanMessage(content=content))
                elif role == 'assistant':
                    langchain_messages.append(AIMessage(content=content))
                else:
                    # 기타 역할은 HumanMessage로 처리
                    langchain_messages.append(HumanMessage(content=content))
            
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
            
            # langchain-ollama ChatOllama 인스턴스 생성
            llm = ChatOllama(
                model=self.model_name,
                base_url=self.base_url,
                temperature=temperature,
                top_p=top_p,
                num_predict=max_tokens,
                timeout=self.timeout,
            )
            
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
        """비동기 질문에 대한 답변 생성 (메타데이터 포함)"""
        # 일반적인 질문인지 확인
        is_general = is_general_question(question) and not context
        has_rag_context = bool(context and context.strip())
        
        # 시스템 프롬프트와 유저 프롬프트 분리
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
        """비동기 Chat API를 사용한 텍스트 생성"""
        if not self._check_server_status():
            self.logger.error("Ollama 서버가 실행되지 않았습니다")
            return None
        
        try:
            # kwargs에서 파라미터 추출
            temperature = kwargs.get('temperature', self.temperature)
            top_p = kwargs.get('top_p', self.top_p)
            max_tokens = kwargs.get('max_tokens', self.max_tokens)
            
            # langchain-ollama ChatOllama 인스턴스 생성
            llm = ChatOllama(
                model=self.model_name,
                base_url=self.base_url,
                temperature=temperature,
                top_p=top_p,
                num_predict=max_tokens,
                timeout=self.timeout,
            )
            
            # LangChain 메시지 형식으로 변환
            langchain_messages = []
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                
                if role == 'system':
                    langchain_messages.append(SystemMessage(content=content))
                elif role == 'user':
                    langchain_messages.append(HumanMessage(content=content))
                elif role == 'assistant':
                    langchain_messages.append(AIMessage(content=content))
                else:
                    langchain_messages.append(HumanMessage(content=content))
            
            start_time = time.time()
            # 비동기 호출
            response = await llm.ainvoke(langchain_messages)
            processing_time = time.time() - start_time
            
            generated_text = response.content if hasattr(response, 'content') else str(response)
            
            # 사용된 토큰 수 추출
            tokens_used = None
            if hasattr(response, 'response_metadata'):
                usage = response.response_metadata.get('usage', {})
                tokens_used = usage.get('total_tokens') or usage.get('eval_count')
            
            self.logger.debug(f"비동기 Chat API 텍스트 생성 완료: {len(generated_text)}자, {processing_time:.2f}초")
            
            return LLMResponse(
                text=generated_text,
                model=self.model_name,
                processing_time=processing_time,
                tokens_used=tokens_used
            )
                
        except Exception as e:
            self.logger.error(f"비동기 Chat API 텍스트 생성 중 오류: {str(e)}")
            # Chat API 실패 시 기존 generate_text_async로 폴백
            self.logger.warning("비동기 Chat API 실패, 기존 generate_text_async로 폴백")
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
            return await self.generate_text_async(prompt, **kwargs)


def create_llm_client(config: Optional[Dict[str, Any]] = None) -> OllamaLLMClient:
    """LLM 클라이언트 생성"""
    return OllamaLLMClient(config)


def generate_answer(question: str, context: str = "", config: Optional[Dict[str, Any]] = None) -> str:
    """질문에 대한 답변 생성 (편의 함수)"""
    client = create_llm_client(config)
    return client.generate_answer(question, context)
