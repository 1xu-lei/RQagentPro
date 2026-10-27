"""
知识图谱查询模块 — 法规追溯、关联推荐、安全路径

图结构:
  Domain(12) → SubLabel(65) → Question(1204) → Answer(1204) → Regulation(28)
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx

from src.infrastructure.logging import get_logger

logger = get_logger(__name__)

GRAPH_PATH = Path(__file__).parent.parent.parent.parent / "data" / "graph_export.json"


class GraphQuery:
    """知识图谱查询器"""

    def __init__(self):
        self.g = nx.MultiDiGraph()
        self._loaded = False

    def load(self):
        if self._loaded:
            return

        if not GRAPH_PATH.exists():
            logger.warning(f"Graph file not found: {GRAPH_PATH}")
            self._loaded = True
            return

        with open(GRAPH_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        for node in data.get("nodes", []):
            self.g.add_node(node["id"], **{k: v for k, v in node.items() if k != "id"})

        for edge in data.get("edges", []):
            self.g.add_edge(edge["source"], edge["target"], relation=edge.get("relation", ""))

        self._loaded = True
        logger.info(f"Graph loaded: {self.g.number_of_nodes()} nodes, "
                   f"{self.g.number_of_edges()} edges")

    # ================================================================
    # 法规追溯
    # ================================================================

    def lookup_regulation(self, fg_code: str) -> Optional[Dict]:
        """根据 FG 编码查询法规详情（名称/层级/摘要来自 taxonomy，关联问题来自图谱）"""
        self.load()
        from src.domain.taxonomy import get_regulation_info

        info = get_regulation_info(fg_code) or {}

        questions = []
        node_id = f"Regulation:{fg_code}"
        if node_id in self.g:
            attrs = self.g.nodes[node_id]
            for _, __, edge_attrs in self.g.in_edges(node_id, data=True):
                if edge_attrs.get("relation") == "cites":
                    # 从 Answer → Question（将 A:xxx 转为 Q:xxx）
                    src = self.g.nodes.get("Q:" + _.split(":")[1], {})
                    if src:
                        questions.append(src.get("label", "")[:60])
        else:
            attrs = {}

        # taxonomy 未定义的编码回退到图谱节点属性（可能为空）
        return {
            "code": fg_code,
            "name": info.get("name", ""),
            "level": info.get("level", ""),
            "title": info.get("title") or attrs.get("label", ""),
            "summary": info.get("summary") or attrs.get("summary", ""),
            "scenarios": info.get("scenarios") or attrs.get("scenarios", []),
            "related_questions": questions[:5],
        }

    def get_regulations_by_domain(self, domain_name: str) -> List[Dict]:
        """查询某域下所有关联法规"""
        self.load()
        regs = set()
        for node_id in self.g.nodes:
            if node_id.startswith("Q:") and self.g.nodes[node_id].get("domain") == domain_name:
                for _, aid, _ in self.g.out_edges(node_id, data=True):
                    for _, rid, _ in self.g.out_edges(aid, data=True):
                        if rid.startswith("Regulation:"):
                            regs.add(rid)
        return [{"code": r.replace("Regulation:", ""),
                 "title": self.g.nodes[r].get("label", "")}
                for r in sorted(regs)]

    # ================================================================
    # 关联问题推荐
    # ================================================================

    def get_related_questions(self, question_id: int = None, query: str = "",
                              limit: int = 5) -> List[Dict]:
        """
        获取关联问题
        - 同子标签下的问题 (via related_to 边)
        - 同法规引用的问题
        """
        self.load()

        if question_id:
            qid = f"Q:{question_id}"
            if qid not in self.g:
                return []

            related = []
            for _, nbr, edge_attrs in list(self.g.out_edges(qid, data=True)) + list(self.g.in_edges(qid, data=True)):
                if edge_attrs.get("relation") == "related_to" and nbr.startswith("Q:"):
                    node = self.g.nodes[nbr]
                    related.append({
                        "id": int(nbr.replace("Q:", "")),
                        "question": node.get("label", ""),
                        "domain": node.get("domain", ""),
                        "sub_label": node.get("sub_label", ""),
                    })
            return related[:limit]

        # 如果没有指定 id，找 query 所在子标签下的所有问题
        if query:
            sub_label = None
            for node_id in self.g.nodes:
                if node_id.startswith("Q:") and query in self.g.nodes[node_id].get("label", ""):
                    sub_label = self.g.nodes[node_id].get("sub_label")
                    domain = self.g.nodes[node_id].get("domain")
                    if sub_label:
                        break

            if sub_label:
                results = []
                for node_id in self.g.nodes:
                    if (node_id.startswith("Q:")
                            and self.g.nodes[node_id].get("sub_label") == sub_label
                            and self.g.nodes[node_id].get("domain") == domain):
                        results.append({
                            "id": int(node_id.replace("Q:", "")),
                            "question": self.g.nodes[node_id].get("label", ""),
                            "domain": domain,
                            "sub_label": sub_label,
                        })
                return results[:limit]

        return []

    # ================================================================
    # 安全路径
    # ================================================================

    def get_safety_paths(self) -> List[Dict]:
        """获取所有安全敏感路径（问题→安全域）"""
        self.load()
        paths = []
        for src, dst, attrs in self.g.edges(data=True):
            if attrs.get("relation") == "safety_alert":
                src_node = self.g.nodes.get(src, {})
                paths.append({
                    "question": src_node.get("label", "")[:60],
                    "domain": src_node.get("domain", ""),
                    "sub_label": src_node.get("sub_label", ""),
                })
        return paths

    # ================================================================
    # 概览
    # ================================================================

    def get_graph_summary(self) -> Dict:
        self.load()
        type_counts = defaultdict(int)
        safety_count = 0
        for _, attrs in self.g.nodes(data=True):
            type_counts[attrs.get("type", "?")] += 1
            if attrs.get("safety"):
                safety_count += 1

        return {
            "total_nodes": self.g.number_of_nodes(),
            "total_edges": self.g.number_of_edges(),
            "node_types": dict(type_counts),
            "domains": [self.g.nodes[f"Domain:{d.name}"]["label"]
                       for d in self._get_domains() if f"Domain:{d.name}" in self.g],
            "safety_questions": safety_count,
            "regulations": type_counts.get("Regulation", 0),
        }

    def _get_domains(self):
        from src.domain.taxonomy import DOMAINS
        return DOMAINS


# 单例
_graph_query_instance: Optional[GraphQuery] = None


def get_graph_query() -> Optional[GraphQuery]:
    global _graph_query_instance
    if _graph_query_instance is None:
        gq = GraphQuery()
        gq.load()
        if gq.g.number_of_nodes() > 0:
            _graph_query_instance = gq
        else:
            return None
    return _graph_query_instance
