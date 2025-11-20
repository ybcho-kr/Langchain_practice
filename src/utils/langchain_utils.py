"""
LangChain 유틸리티 모듈
LangChain 컴포넌트 생성 및 변환을 위한 유틸리티 함수 모음
"""

from typing import List, Dict, Any, Optional, Union
from src.utils.logger import get_logger
from src.utils.config import ModelConfig

try:
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    ChatOllama = None
    OllamaEmbeddings = None
    SystemMessage = None
    HumanMessage = None
    AIMessage = None
    BaseMessage = None


def create_chat_ollama(
    model_name: str,
    base_url: str = "http://localhost:11434",
    temperature: float = 0.1,
    top_p: float = 0.9,
    max_tokens: Optional[int] = None,
    timeout: int = 300,
    **kwargs
) -> Optional[Any]:
    """
    ChatOllama 인스턴스 생성 팩토리 함수
    
    Args:
        model_name: Ollama 모델명 (예: 'gemma3:4b', 'llama3.1:8b')
        base_url: Ollama 서버 URL (기본값: 'http://localhost:11434')
        temperature: 샘플링 온도 (0.0-1.0, 기본값: 0.1)
        top_p: Top-p 샘플링 (0.0-1.0, 기본값: 0.9)
        max_tokens: 최대 생성 토큰 수 (None이면 모델 기본값 사용)
        timeout: 타임아웃 (초, 기본값: 300)
        **kwargs: 추가 ChatOllama 파라미터
    
    Returns:
        ChatOllama 인스턴스 또는 None (LangChain이 설치되지 않은 경우)
    
    Example:
        ```python
        llm = create_chat_ollama(
            model_name='gemma3:4b',
            temperature=0.1,
            max_tokens=1000
        )
        response = llm.invoke([HumanMessage(content="Hello")])
        ```
    """
    logger = get_logger()
    
    if not LANGCHAIN_AVAILABLE:
        logger.error("langchain-ollama가 설치되지 않았습니다. pip install langchain-ollama")
        return None
    
    try:
        # ChatOllama 파라미터 구성
        chat_params = {
            'model': model_name,
            'base_url': base_url,
            'temperature': temperature,
            'top_p': top_p,
            'timeout': timeout,
        }
        
        # max_tokens가 제공된 경우 num_predict로 전달
        if max_tokens is not None:
            chat_params['num_predict'] = max_tokens
        
        # 추가 kwargs 병합
        chat_params.update(kwargs)
        
        # ChatOllama 인스턴스 생성
        llm = ChatOllama(**chat_params)
        
        logger.debug(
            f"ChatOllama 인스턴스 생성 완료: "
            f"model={model_name}, base_url={base_url}, "
            f"temperature={temperature}, top_p={top_p}, "
            f"max_tokens={max_tokens}, timeout={timeout}초"
        )
        
        return llm
    
    except Exception as e:
        logger.error(f"ChatOllama 인스턴스 생성 실패: {str(e)}", exc_info=True)
        return None


def create_chat_ollama_from_config(
    config: Union[Dict[str, Any], ModelConfig],
    model_name: Optional[str] = None,
    **override_kwargs
) -> Optional[Any]:
    """
    설정 객체/딕셔너리로부터 ChatOllama 인스턴스 생성
    
    Args:
        config: ModelConfig 객체 또는 설정 딕셔너리
        model_name: 모델명 (None이면 config에서 가져옴)
        **override_kwargs: 설정을 덮어쓰는 추가 파라미터
    
    Returns:
        ChatOllama 인스턴스 또는 None
    
    Example:
        ```python
        from src.utils.config import get_llm_config
        
        llm_config = get_llm_config()
        llm = create_chat_ollama_from_config(llm_config)
        ```
    """
    logger = get_logger()
    
    # 설정 파싱
    if isinstance(config, dict):
        name = model_name or config.get('name', 'gemma3:12b')
        base_url = config.get('base_url', 'http://localhost:11434')
        max_tokens = config.get('max_tokens', 1000)
        temperature = config.get('temperature', 0.1)
        top_p = config.get('top_p', 0.9)
        timeout = config.get('timeout', 300)
    elif hasattr(config, 'name'):
        # ModelConfig 객체
        name = model_name or config.name
        base_url = getattr(config, 'base_url', 'http://localhost:11434')
        max_tokens = getattr(config, 'max_tokens', 1000)
        temperature = getattr(config, 'temperature', 0.1)
        top_p = getattr(config, 'top_p', 0.9)
        timeout = getattr(config, 'timeout', 300)
    else:
        logger.error(f"지원하지 않는 설정 타입: {type(config)}")
        return None
    
    # override_kwargs로 덮어쓰기
    final_params = {
        'model_name': name,
        'base_url': base_url,
        'temperature': override_kwargs.get('temperature', temperature),
        'top_p': override_kwargs.get('top_p', top_p),
        'max_tokens': override_kwargs.get('max_tokens', max_tokens),
        'timeout': override_kwargs.get('timeout', timeout),
    }
    
    # override_kwargs에서 이미 처리한 파라미터 제외
    remaining_kwargs = {k: v for k, v in override_kwargs.items() 
                       if k not in ['temperature', 'top_p', 'max_tokens', 'timeout']}
    final_params.update(remaining_kwargs)
    
    return create_chat_ollama(**final_params)


def create_ollama_embeddings(
    model_name: str = "bge-m3-korean",
    base_url: str = "http://localhost:11434",
    **kwargs
) -> Optional[Any]:
    """
    OllamaEmbeddings 인스턴스 생성 팩토리 함수
    
    Args:
        model_name: Ollama 임베딩 모델명 (기본값: 'bge-m3-korean')
        base_url: Ollama 서버 URL (기본값: 'http://localhost:11434')
        **kwargs: 추가 OllamaEmbeddings 파라미터
    
    Returns:
        OllamaEmbeddings 인스턴스 또는 None (LangChain이 설치되지 않은 경우)
    
    Example:
        ```python
        embeddings = create_ollama_embeddings(model_name='bge-m3-korean')
        vector = embeddings.embed_query("Hello world")
        ```
    """
    logger = get_logger()
    
    if not LANGCHAIN_AVAILABLE:
        logger.error("langchain-ollama가 설치되지 않았습니다. pip install langchain-ollama")
        return None
    
    try:
        # OllamaEmbeddings 파라미터 구성
        embedding_params = {
            'model': model_name,
            'base_url': base_url,
        }
        
        # 추가 kwargs 병합
        embedding_params.update(kwargs)
        
        # OllamaEmbeddings 인스턴스 생성
        embeddings = OllamaEmbeddings(**embedding_params)
        
        logger.debug(
            f"OllamaEmbeddings 인스턴스 생성 완료: "
            f"model={model_name}, base_url={base_url}"
        )
        
        return embeddings
    
    except Exception as e:
        logger.error(f"OllamaEmbeddings 인스턴스 생성 실패: {str(e)}", exc_info=True)
        return None


def convert_to_langchain_messages(
    messages: List[Dict[str, str]]
) -> List[BaseMessage]:
    """
    딕셔너리 형식 메시지를 LangChain 메시지 형식으로 변환
    
    Args:
        messages: 메시지 딕셔너리 리스트
            각 메시지는 {'role': 'system'|'user'|'assistant', 'content': str} 형식
    
    Returns:
        LangChain 메시지 리스트 (SystemMessage, HumanMessage, AIMessage)
    
    Example:
        ```python
        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi there!'}
        ]
        langchain_messages = convert_to_langchain_messages(messages)
        # [SystemMessage(...), HumanMessage(...), AIMessage(...)]
        ```
    """
    logger = get_logger()
    
    if not LANGCHAIN_AVAILABLE:
        logger.error("langchain-core가 설치되지 않았습니다. pip install langchain-core")
        return []
    
    langchain_messages = []
    
    for msg in messages:
        if not isinstance(msg, dict):
            logger.warning(f"메시지가 딕셔너리 형식이 아닙니다: {type(msg)}")
            continue
        
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        
        if not content:
            logger.warning(f"메시지 내용이 비어있습니다: role={role}")
            continue
        
        try:
            if role == 'system':
                langchain_messages.append(SystemMessage(content=content))
            elif role == 'user':
                langchain_messages.append(HumanMessage(content=content))
            elif role == 'assistant':
                langchain_messages.append(AIMessage(content=content))
            else:
                # 기타 역할은 HumanMessage로 처리
                logger.debug(f"알 수 없는 역할 '{role}', HumanMessage로 처리")
                langchain_messages.append(HumanMessage(content=content))
        except Exception as e:
            logger.error(f"메시지 변환 실패 (role={role}): {str(e)}")
            continue
    
    return langchain_messages


def create_simple_message(content: str, role: str = 'user') -> Optional[BaseMessage]:
    """
    단일 메시지를 LangChain 메시지 형식으로 변환 (간편 함수)
    
    Args:
        content: 메시지 내용
        role: 메시지 역할 ('system', 'user', 'assistant', 기본값: 'user')
    
    Returns:
        LangChain 메시지 객체 또는 None
    
    Example:
        ```python
        user_msg = create_simple_message("Hello", role='user')
        system_msg = create_simple_message("You are helpful.", role='system')
        ```
    """
    if not LANGCHAIN_AVAILABLE:
        return None
    
    if role == 'system':
        return SystemMessage(content=content)
    elif role == 'user':
        return HumanMessage(content=content)
    elif role == 'assistant':
        return AIMessage(content=content)
    else:
        # 기본값: HumanMessage
        return HumanMessage(content=content)

