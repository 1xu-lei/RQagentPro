"""
知识库检索工具 — 混合检索（FAISS + BM25 + RRF 融合）
"""

import json
from typing import Optional
from src.domain.tools.base import BaseTool


class KnowledgeSearchTool(BaseTool):
    name = "search_knowledge_base"
    description = (
        "在衡阳燃气客服知识库中检索与用户问题相关的内容。"
        "知识库包含报装、缴费、安检、抢修、安全用气、政策咨询等全业务场景的标准答案。"
        "返回最匹配的知识条目（含标准答案、标签分类、法规编码）。"
        "当需要回答用户关于燃气业务的具体问题时，优先使用此工具检索。"
        "如果首次检索结果不理想，可以调整查询词重新检索。"
    )
    priority = 80

    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "检索查询词。应提炼用户问题的核心意图和关键词，"
                    "而非直接使用原始口语化表达。"
                ),
            },
            "intent_filter": {
                "type": "string",
                "description": "可选，按一级业务域过滤结果。不传则不过滤。",
            },
            "intent_hint": {
                "type": "object",
                "description": (
                    "可选，当前对话的意图上下文，"
                    "格式 {'intent_name':'气费管理','sub_intent':'欠费停气'}。"
                    "传入后可提升同类问题的检索精度。"
                ),
            },
        },
        "required": ["query"],
    }

    def __init__(self):
        self._retriever = None

    @property
    def retriever(self):
        if self._retriever is None:
            from src.domain.retriever.hybrid import get_retriever
            self._retriever = get_retriever()
        return self._retriever

    async def execute(
        self,
        query: str,
        intent_filter: Optional[str] = None,
        intent_hint: Optional[dict] = None,
    ) -> str:
        intent = None
        if intent_hint:
            intent = intent_hint
        elif intent_filter:
            intent = {
                "intent_name": intent_filter,
                "sub_intent": "",
                "confidence": 0.7,
            }

        # 查询改写：口语化 → 标准化
        from src.domain.retriever.query_rewriter import get_query_rewriter
        rewriter = get_query_rewriter()
        rewritten_query = rewriter.rewrite(query, context={"last_intent": intent})

        results = await self.retriever.hybrid_search(rewritten_query, intent=intent, top_k=5)

        if not results:
            return json.dumps({
                "status": "no_results",
                "message": "未找到与查询匹配的知识条目。建议调整查询词或告知用户该问题暂未收录。",
                "results": [],
            }, ensure_ascii=False)

        formatted = []
        for r in results:
            entry = r["entry"]
            formatted.append({
                "score": round(r["score"], 3),
                "vector_score": r.get("vector_score", 0.0),
                "category": f"{entry['一级域']} > {entry['二级标签']}",
                "question": entry.get("标准问", "")[:100],
                "answer": entry.get("标准答", "")[:500],
                "regulation_code": entry.get("法规编码", ""),
                "keywords": entry.get("关键词", "")[:100],
            })

        return json.dumps({
            "status": "ok",
            "total_found": len(formatted),
            "results": formatted,
        }, ensure_ascii=False)
