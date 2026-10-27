"""
服务信息查询工具 — 客服电话、营业网点、时间等静态信息
"""

import json
from src.domain.tools.base import BaseTool


# 服务信息库
SERVICE_INFO = {
    "抢修电话": "95777（24小时）",
    "客服电话": "0734-8192944",
    "服务热线": "0734-8677777",
    "监督投诉": "0734-8801377",
    "报装咨询": "0734-8718579",
    "营业网点": "石鼓区营业厅（解放路xxx号）、雁峰区营业厅（黄白路xxx号）、蒸湘区营业厅（船山大道xxx号）、珠晖区营业厅（东风路xxx号）",
    "营业时间": "工作日 8:30-17:30，周六 9:00-16:00，周日休息",
    "微信公众号": "衡阳天然气",
    "官方网站": "hengyangtianranqi.cn",
}


class ServiceInfoTool(BaseTool):
    name = "get_service_info"
    description = (
        "查询衡阳燃气公司的服务信息。用于回答用户关于客服电话、抢修电话、"
        "营业网点地址、营业时间、微信公众号等基础服务信息的问题。"
        "用户问电话/地址/营业时间时直接调用此工具，不要调用 search_knowledge_base。"
    )
    priority = 60

    parameters = {
        "type": "object",
        "properties": {
            "info_type": {
                "type": "string",
                "description": "信息类型。可选值：'电话'、'地址'、'营业时间'、'公众号'、'网站'、'全部'",
            }
        },
        "required": ["info_type"],
    }

    async def execute(self, info_type: str) -> str:
        # 当用户问电话时，返回所有联系方式（避免二次 tool call）
        if info_type in ("电话", "联系方式", "联系"):
            phones = "\n".join([
                f"24h抢修专线：{SERVICE_INFO['抢修电话']}",
                f"人工客服：{SERVICE_INFO['客服电话']}",
                f"服务热线：{SERVICE_INFO['服务热线']}",
                f"监督投诉：{SERVICE_INFO['监督投诉']}",
            ])
            return json.dumps({
                "found": True,
                "info_type": "电话（全量）",
                "content": phones,
                "_hint": "已返回所有联系电话，无需再调此工具获取其他电话",
            }, ensure_ascii=False)

        if info_type == "全部":
            content = "\n".join(f"{k}: {v}" for k, v in SERVICE_INFO.items())
            return json.dumps({
                "found": True, "info_type": "全部", "content": content,
            }, ensure_ascii=False)

        # 模糊匹配
        for key, value in SERVICE_INFO.items():
            if info_type in key or key in info_type:
                return json.dumps({
                    "found": True, "info_type": key, "content": value,
                }, ensure_ascii=False)

        return json.dumps({
            "found": False, "info_type": info_type,
            "content": "未找到该类信息，建议转人工咨询",
        }, ensure_ascii=False)
