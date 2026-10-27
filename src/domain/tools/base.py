"""
工具抽象基类

每个工具实现此基类，Agent 通过注册表获取工具列表和调用工具。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseTool(ABC):
    """工具抽象基类"""

    # 工具名称（唯一标识，对应 Function Calling 的 function.name）
    name: str = ""

    # 工具描述（给 LLM 看的功能说明）
    description: str = ""

    # 参数 JSON Schema
    parameters: Dict[str, Any] = {}

    # 是否为强制工具（如 check_safety 必须第一个被考虑）
    priority: int = 0  # 值越大越重要，LLM 会优先看到

    @abstractmethod
    async def execute(self, **kwargs) -> str:
        """
        执行工具，返回结果字符串（JSON 格式）

        Args:
            **kwargs: 对应 parameters 中定义的参数

        Returns:
            JSON 格式的结果字符串
        """
        ...

    def to_openai_schema(self) -> Dict[str, Any]:
        """转换为 OpenAI Function Calling 格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def __repr__(self):
        return f"<Tool:{self.name}>"
