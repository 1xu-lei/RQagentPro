"""
安全检查工具 — 双层防御（关键词 + 标签体系）

v3.1: 支持意图上下文传入，实现关键词+标签联合检测
"""

import json
from src.domain.tools.base import BaseTool


class SafetyCheckTool(BaseTool):
    name = "check_safety"
    description = (
        "★★★ 最高优先级 ★★★ 检测用户问题是否涉及燃气安全紧急情况"
        "（泄漏、中毒、火灾、爆炸、报警器等）。"
        "如果用户提到煤气味、报警器响、头晕恶心、着火爆炸等词，"
        "必须在回答任何问题前先调用此工具。"
        "如果此工具返回 is_safety=true，必须直接使用返回的安全模板回答，"
        "禁止调用其他工具，禁止改写安全话术。"
    )
    priority = 100  # 最高优先级

    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "用户原始输入文本",
            },
            "intent_name": {
                "type": "string",
                "description": "可选，已分类的意图一级域名，用于标签级安全校验",
            },
            "sub_intent": {
                "type": "string",
                "description": "可选，已分类的意图二级标签名",
            },
        },
        "required": ["query"],
    }

    def __init__(self):
        self._gateway = None

    @property
    def gateway(self):
        if self._gateway is None:
            from src.domain.safety.gateway import SafetyGateway
            self._gateway = SafetyGateway()
        return self._gateway

    async def execute(
        self,
        query: str,
        intent_name: str = "",
        sub_intent: str = "",
    ) -> str:
        # 信息查询类句式不触发安全模式
        info_patterns = ["电话", "多少", "是什么", "怎么查", "在哪", "号码", "热线"]
        is_info_query = any(p in query for p in info_patterns)

        if is_info_query:
            return json.dumps({
                "is_safety": False,
                "keywords": [],
                "label_code": None,
                "template_name": None,
                "action_required": "信息查询类句式，跳过安全检查",
            }, ensure_ascii=False)

        # 联合检测：关键词 + 标签
        is_safety, matched_kws, safety_label = self.gateway.check_safety_combined(
            query=query,
            intent_name=intent_name,
            sub_intent=sub_intent,
        )

        if not is_safety:
            return json.dumps({
                "is_safety": False,
                "keywords": [],
                "label_code": None,
                "template_name": None,
                "action_required": "无安全风险，可以继续正常处理",
            }, ensure_ascii=False)

        # 命中 → 获取安全模板
        template = self.gateway.get_safety_template(matched_kws or [])

        return json.dumps({
            "is_safety": True,
            "keywords": matched_kws,
            "label_code": safety_label,
            "template_name": template.get("title", "安全处置"),
            "template_content": template,
            "action_required": (
                "★★★ 强制走安全模板 + 转人工/抢修 ★★★ "
                "禁止调用其他工具，禁止LLM改写安全话术。"
                "请直接用安全模板内容回复用户。"
            ),
        }, ensure_ascii=False)
