"""
工具注册表 — Agent 工具集的中央管理

特性：
- 注册/注销工具
- 生成 OpenAI Function Calling schema 列表
- 按名称查找和执行工具
- 按优先级排序
"""

from typing import Dict, List, Optional

from src.domain.tools.base import BaseTool
from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


class ToolRegistry:
    """工具注册表"""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool):
        """注册一个工具（相同名称已注册时跳过）"""
        if tool.name in self._tools:
            logger.debug(f"Tool '{tool.name}' already registered, skipping")
            return
        self._tools[tool.name] = tool
        logger.debug(f"Tool registered: {tool.name}")

    def register_all(self, tools: List[BaseTool]):
        """批量注册工具"""
        for tool in tools:
            self.register(tool)

    def unregister(self, name: str):
        """注销工具"""
        if name in self._tools:
            del self._tools[name]
            logger.debug(f"Tool unregistered: {name}")

    def get(self, name: str) -> Optional[BaseTool]:
        """按名称获取工具"""
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """列出所有工具名称"""
        return list(self._tools.keys())

    def to_openai_schemas(self) -> List[Dict]:
        """
        生成所有工具的 OpenAI Function Calling schema 列表
        按优先级降序排列，让高优先级工具（如 check_safety）先被 LLM 看到
        """
        tools = sorted(self._tools.values(), key=lambda t: (-t.priority, t.name))
        return [t.to_openai_schema() for t in tools]

    async def execute(self, name: str, arguments: Dict) -> str:
        """
        按名称执行工具

        Args:
            name: 工具名称
            arguments: 参数字典

        Returns:
            工具执行结果（JSON 字符串）

        Raises:
            ValueError: 工具不存在
        """
        from src.infrastructure.tracing import get_tracer

        tool = self._tools.get(name)
        if tool is None:
            raise ValueError(f"Unknown tool: {name}")

        logger.info(f"Tool executing: {name}({str(arguments)[:100]})")

        with get_tracer().span(f"tool.{name}", tool=name) as span:
            try:
                result = await tool.execute(**arguments)
                logger.debug(f"Tool {name} result: {result[:150]}")
                span.set_attr("result_preview", result[:200])
                span.set_attr("result_len", len(result))
                return result
            except Exception as e:
                logger.error(f"Tool {name} execution failed: {e}")
                span.set_attr("error", str(e)[:200])
                import json
                return json.dumps(
                    {"error": f"Tool execution error: {str(e)[:200]}"},
                    ensure_ascii=False,
                )


# ============================================================
# 全局单例
# ============================================================

_registry_instance: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """获取全局工具注册表单例"""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ToolRegistry()
    return _registry_instance
