"""
转人工工具 — 无法处理时的兜底方案
"""

import json
from src.domain.tools.base import BaseTool


class HumanTransferTool(BaseTool):
    name = "request_human_transfer"
    description = (
        "当以下情况发生时，将对话转接至人工客服："
        "1) 多次检索仍无法找到匹配答案；"
        "2) 用户明确要求转人工；"
        "3) 问题涉及投诉、法律纠纷等复杂事务；"
        "4) 系统无法准确理解用户意图且澄清失败。"
        "调用此工具意味着本轮对话以转人工结束。"
    )
    priority = 10

    parameters = {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "转人工的具体原因，如'知识库未覆盖该问题'、'用户明确要求转人工'等",
            },
            "severity": {
                "type": "string",
                "enum": ["normal", "urgent"],
                "description": "紧急程度：normal=普通转接，urgent=紧急转接",
            },
        },
        "required": ["reason"],
    }

    # 联系方式信息
    CONTACT = {
        "normal": {
            "phone": "0734-8192944",
            "hours": "工作日 8:30-17:30",
            "wechat": "「衡阳燃气」微信公众号 - 在线客服",
        },
        "urgent": {
            "phone": "95777",
            "hours": "24小时抢修专线",
            "wechat": "",
        },
    }

    async def execute(self, reason: str, severity: str = "normal") -> str:
        info = self.CONTACT.get(severity, self.CONTACT["normal"])

        return json.dumps({
            "status": "transferred",
            "severity": severity,
            "reason": reason,
            "contact": {
                "phone": info["phone"],
                "hours": info["hours"],
                "wechat": info["wechat"],
            },
            "note": "已将用户请求记录为待转人工。请生成包含联系方式和人转理由的礼貌结束语。",
        }, ensure_ascii=False)
