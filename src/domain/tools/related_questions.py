"""
关联问题推荐工具
"""

import json
from src.domain.tools.base import BaseTool


class RelatedQuestionsTool(BaseTool):
    name = "get_related_questions"
    description = (
        "获取某个业务意图下用户常问的相关问题列表。"
        "用于在回答完用户当前问题后，向用户推荐可能感兴趣的后续问题，提升服务体验。"
    )
    priority = 30

    parameters = {
        "type": "object",
        "properties": {
            "intent_name": {
                "type": "string",
                "description": "意图名称，格式为'一级域-二级标签'，如'安全用气-疑似漏气处置'",
            }
        },
        "required": ["intent_name"],
    }

    def __init__(self):
        self._graph = None

    @property
    def graph(self):
        if self._graph is None:
            from src.domain.knowledge_graph.graph import get_graph_query
            self._graph = get_graph_query()
        return self._graph

    async def execute(self, intent_name: str) -> str:
        if self.graph is None:
            return json.dumps({"status": "unavailable", "questions": []}, ensure_ascii=False)

        try:
            questions = self.graph.get_related_questions(intent_name, limit=5)
            return json.dumps({
                "status": "ok",
                "intent": intent_name,
                "questions": questions,
            }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": str(e)[:100],
                "questions": [],
            }, ensure_ascii=False)
