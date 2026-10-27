"""
工具初始化 — 注册所有工具到全局注册表

在应用启动时调用此函数，确保所有工具已注册
"""

from src.domain.tools.registry import get_tool_registry
from src.domain.tools.safety_check import SafetyCheckTool
from src.domain.tools.knowledge_search import KnowledgeSearchTool
from src.domain.tools.intent_classify import IntentClassifyTool
from src.domain.tools.regulation_lookup import RegulationLookupTool
from src.domain.tools.service_info import ServiceInfoTool
from src.domain.tools.related_questions import RelatedQuestionsTool
from src.domain.tools.human_transfer import HumanTransferTool

from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


def register_all_tools():
    """注册所有工具到全局注册表"""
    registry = get_tool_registry()

    tools = [
        SafetyCheckTool(),        # 安全检查 ★★★ 最高优先级
        KnowledgeSearchTool(),    # 知识库检索
        IntentClassifyTool(),     # 意图分类
        ServiceInfoTool(),        # 服务信息查询
        RegulationLookupTool(),   # 法规查询
        RelatedQuestionsTool(),   # 关联问题推荐
        HumanTransferTool(),      # 转人工
    ]

    registry.register_all(tools)
    logger.info(f"All {len(tools)} tools registered: {registry.list_tools()}")

    return registry
