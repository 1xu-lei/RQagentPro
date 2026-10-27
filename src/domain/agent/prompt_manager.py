"""
Prompt 管理器 — 从 YAML 配置加载并构建系统提示词

将 Prompt 与代码分离，支持：
- YAML 外部管理
- 变量替换（关键信息动态注入）
- 多 Prompt 组合
"""

from typing import Dict, List, Optional

from src.infrastructure.config import get_config


class PromptManager:
    """Prompt 管理器"""

    def __init__(self):
        self._config = get_config()

    def build_system_prompt(self, tools_description: str = "") -> str:
        """
        构建完整的系统提示词

        Args:
            tools_description: 工具的能力描述文本（由代码自动生成）

        Returns:
            完整的系统提示词字符串
        """
        system_prompt = self._config.prompt("system")

        # 如果没有配置，返回默认提示词
        if not system_prompt:
            return self._default_system_prompt(tools_description)

        prompt = system_prompt
        role = prompt.get("role", "衡阳燃气AI客服助手")
        name = prompt.get("name", "衡阳燃气AI客服")

        sections = [f"你是{role}。你的名字是'{name}'。"]

        # 工具指引
        guidance = prompt.get("tool_guidance", {})
        if guidance:
            sections.append("\n## 你的能力")
            sections.append("你拥有一套工具，可以在需要时调用它们来获取信息：")
            for i, (tool_name, tool_desc) in enumerate(guidance.items(), 1):
                sections.append(f"{i}. **{tool_name}** — {tool_desc}")

        # 核心原则
        principles = prompt.get("principles", {})
        if principles:
            sections.append("\n## 核心原则")
            for key, p in sorted(
                principles.items(),
                key=lambda x: x[1].get("priority", 99) if isinstance(x[1], dict) else 99,
            ):
                if isinstance(p, dict):
                    sections.append(f"\n### {p.get('priority', '')}. {p.get('text', '')}")
                else:
                    sections.append(f"\n### {p}")

        # 关键信息
        key_info = prompt.get("key_info", {})
        if key_info:
            sections.append("\n## 衡阳燃气关键信息")
            for k, v in key_info.items():
                sections.append(f"- {v}")

        # 禁止行为
        forbidden = prompt.get("forbidden", [])
        if forbidden:
            sections.append("\n## 禁止行为")
            for fb in forbidden:
                sections.append(f"- {fb}")

        # 额外工具描述
        if tools_description:
            sections.append(f"\n## 工具详细说明\n{tools_description}")

        return "\n".join(sections)

    def get_safety_prompt(self) -> Dict:
        """获取安全相关提示词"""
        return self._config.prompt("safety")

    def get_intent_prompt(self) -> Dict:
        """获取意图分类提示词"""
        return self._config.prompt("intent")

    def _default_system_prompt(self, tools_description: str) -> str:
        """默认系统提示词（配置加载失败时的 fallback）"""
        return f"""你是衡阳天然气有限责任公司（衡阳燃气）的AI智能客服助手。你的名字是"衡阳燃气AI客服"。

## 你的能力
你拥有一套工具来处理用户的各种燃气业务问题。

## 核心原则
1. 安全优先：用户提到燃气泄漏、中毒、火灾等 → 第一步必须调用 check_safety
2. 事实来源唯一：业务知识只能来自 search_knowledge_base 检索结果
3. 效率原则：基础服务信息用 get_service_info，不重复检索
4. 法规追溯：检索结果有 regulation_code → 调用 lookup_regulation
5. 诚实边界：搜不到答案或涉及投诉法律 → request_human_transfer

## 关键信息
- 24小时抢修专线：95777
- 人工客服电话：0734-8192944
- 微信公众号：衡阳天然气

{tools_description}
"""


# 单例
_prompt_manager_instance: Optional[PromptManager] = None


def get_prompt_manager() -> PromptManager:
    global _prompt_manager_instance
    if _prompt_manager_instance is None:
        _prompt_manager_instance = PromptManager()
    return _prompt_manager_instance
