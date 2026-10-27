"""
知识库/图谱查询接口
"""

import json
from pathlib import Path
from fastapi import APIRouter
from src.infrastructure.config import get_config

router = APIRouter(prefix="/api/v1")


@router.get("/tags")
async def get_tags():
    """获取标签体系树"""
    from src.infrastructure.storage.kb_store import get_kb_store
    kb = get_kb_store().get_all()

    tag_tree = {}
    for item in kb:
        cat1 = item["一级域"]
        cat2 = item["二级标签"]
        if cat1 not in tag_tree:
            tag_tree[cat1] = []
        if cat2 not in tag_tree[cat1]:
            tag_tree[cat1].append(cat2)
    return tag_tree


@router.get("/graph/summary")
async def get_graph_summary():
    """知识图谱概览"""
    try:
        from src.domain.knowledge_graph.graph import get_graph_query
        gq = get_graph_query()
        if gq:
            return gq.get_graph_summary()
        return {"error": "Graph not loaded"}
    except Exception as e:
        return {"error": str(e)}


@router.get("/graph/regulation/{fg_code}")
async def get_regulation(fg_code: str):
    """法规详情查询"""
    try:
        from src.domain.knowledge_graph.graph import get_graph_query
        gq = get_graph_query()
        if gq:
            return gq.lookup_regulation(fg_code)
        return {"error": "Graph not loaded"}
    except Exception as e:
        return {"error": str(e)}
