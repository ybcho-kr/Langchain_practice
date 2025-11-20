"""
기본 도구 인터페이스
LangChain BaseTool을 확장한 기본 도구 클래스
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from langchain_core.tools import BaseTool as LangChainBaseTool


class BaseTool(ABC):
    """기본 도구 인터페이스"""
    
    def __init__(self, name: str, description: str):
        """
        Args:
            name: 도구 이름
            description: 도구 설명
        """
        self.name = name
        self.description = description
    
    @abstractmethod
    def execute(self, **kwargs: Any) -> Any:
        """
        도구 실행
        
        Args:
            **kwargs: 도구별 파라미터
            
        Returns:
            도구 실행 결과
        """
        pass
    
    def to_langchain_tool(self) -> LangChainBaseTool:
        """
        LangChain BaseTool로 변환
        
        Returns:
            LangChain BaseTool 인스턴스
        """
        from langchain_core.tools import tool
        
        @tool(name=self.name, description=self.description)
        def tool_func(**kwargs: Any) -> Any:
            return self.execute(**kwargs)
        
        return tool_func

