"""
法规查询工具 — 通过知识图谱查询法规条款详情
"""

import json
from src.domain.tools.base import BaseTool


class RegulationLookupTool(BaseTool):
    name = "lookup_regulation"
    description = (
        "根据法规编码(FG-XX-XXX格式)查询具体的法律法规条款详情，"
        "包括法规名称、层级（法律/行政法规/部门规章/地方规范/行业规范）等信息。"
        "在回答中需要引用法规依据时调用此工具。"
    )
    priority = 50

    parameters = {
        "type": "object",
        "properties": {
            "fg_code": {
                "type": "string",
                "description": (
                    "法规编码，格式如 FG-FL-001(法律)、FG-XZ-002(行政法规)、"
                    "FG-BM-001(部门规章)、FG-DF-001(地方规范)、FG-HY-001(行业规范)"
                ),
            }
        },
        "required": ["fg_code"],
    }

    def __init__(self):
        self._graph = None

    @property
    def graph(self):
        if self._graph is None:
            from src.domain.knowledge_graph.graph import get_graph_query
            self._graph = get_graph_query()
        return self._graph

    async def execute(self, fg_code: str) -> str:
        if self.graph is None:
            return json.dumps({
                "status": "unavailable",
                "message": "知识图谱查询服务暂不可用",
                "code": fg_code,
            }, ensure_ascii=False)

        try:
            reg = self.graph.lookup_regulation(fg_code)
            if reg and reg.get("name"):
                return json.dumps({
                    "status": "ok",
                    "code": fg_code,
                    "name": reg["name"],
                    "level": reg.get("level", ""),
                    "full_ref": f"{reg['name']}({fg_code}) [{reg.get('level', '')}]",
                }, ensure_ascii=False)
            else:
                return json.dumps({
                    "status": "not_found",
                    "code": fg_code,
                    "message": f"未找到编码 {fg_code} 对应的法规详情",
                }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({
                "status": "error",
                "code": fg_code,
                "message": f"查询异常: {str(e)[:100]}",
            }, ensure_ascii=False)
