"""
同步法规详情到知识图谱 — 以 src/domain/taxonomy.REGULATION_MAP 为唯一真相源。

将 data/graph_export.json 中所有 Regulation 节点的 label/summary/scenarios
与 taxonomy 对齐（补充/修正空壳占位节点）。

用法：
    python data/sync_regulations_to_graph.py
"""

import json
import os
import sys

# 项目根目录加入 sys.path，以便 import src.domain.taxonomy
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.domain.taxonomy import REGULATION_MAP  # noqa: E402

GRAPH_PATH = os.path.join(PROJECT_ROOT, "data", "graph_export.json")


def build_label(code: str, info: dict) -> str:
    """标签格式：FG-xx 《法规名》条款（整部法规无条款后缀）"""
    name = info.get("name", "")
    title = info.get("title", "")
    label = f"{code} 《{name}》"
    if title and title != name:
        label += title
    return label


def main():
    with open(GRAPH_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    updated = 0
    missing = []
    for node in data.get("nodes", []):
        if node.get("type") != "Regulation":
            continue
        code = node["id"].replace("Regulation:", "")
        info = REGULATION_MAP.get(code)
        if not info:
            missing.append(code)
            continue
        node["label"] = build_label(code, info)
        node["summary"] = info.get("summary", "")
        node["scenarios"] = info.get("scenarios", [])
        updated += 1

    with open(GRAPH_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"已同步 {updated} 个 Regulation 节点")
    if missing:
        print(f"图谱中存在但 taxonomy 未定义的 Regulation 节点: {missing}")


if __name__ == "__main__":
    main()
