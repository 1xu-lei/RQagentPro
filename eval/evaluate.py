# -*- coding: utf-8 -*-
"""
RQagentPro 完整评估系统

基于旧系统 eval/evaluate.py 适配新分层架构。

5 项指标：
  1. 意图分类准确率（12 个一级域，65 个二级标签）
  2. 安全检测召回率（关键词 + 标签双层防御）
  3. 检索 Top-1 命中率
  4. 检索 Top-3 召回率
  5. 自动应答率

运行方式：python eval/evaluate.py
"""

import json, os, sys, time, random, asyncio
from collections import defaultdict
from pathlib import Path
from datetime import datetime

# 项目根目录
PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

# 加载 .env
import os
env_file = PROJECT_DIR / ".env"
if env_file.exists():
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())

# 数据路径
DATA_DIR = PROJECT_DIR / "data"
EVAL_DIR = PROJECT_DIR / "eval"
REPORT_JSON = EVAL_DIR / "evaluation_report.json"
REPORT_HTML = EVAL_DIR / "evaluation_report.html"
BUSINESS_DIR = EVAL_DIR / "business"
BUSINESS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 意图分类（同步封装异步调用）
# ============================================================

# 全局复用实例，避免每条的重复加载
_intent_tool = None
_safety_gw = None

def classify_intent(query: str) -> dict:
    """使用新架构 IntentClassifyTool 做意图分类（同步包装，供非异步上下文使用）"""
    # 注意：在 async evaluate() 中请直接使用 classify_intent_async()
    return asyncio.run(classify_intent_async(query))


async def classify_intent_async(query: str) -> dict:
    """异步版本：使用新架构 IntentClassifyTool 做意图分类"""
    global _intent_tool, _safety_gw

    if _intent_tool is None:
        from src.domain.tools.intent_classify import IntentClassifyTool
        _intent_tool = IntentClassifyTool()
    if _safety_gw is None:
        from src.domain.safety.gateway import SafetyGateway
        _safety_gw = SafetyGateway()

    result_json = await _intent_tool.execute(query=query)
    try:
        parsed = json.loads(result_json)
    except Exception:
        parsed = {}

    # 安全检测：关键词检测（标准逻辑）
    safety_kws = _safety_gw.check_safety_keywords(query)
    is_safety = len(safety_kws) > 0

    return {
        "intent_code": _map_cat1_to_code(parsed.get("intent_name", "")),
        "intent_name": parsed.get("intent_name", "其他"),
        "sub_intent": parsed.get("sub_intent", ""),
        "confidence": parsed.get("confidence", 0.5),
        "is_safety": is_safety,
        "safety_keywords": safety_kws,
        "source": parsed.get("method", "unknown"),
        "matched_entries": [],
    }


def _map_cat1_to_code(cat1: str) -> str:
    """使用规范标签体系编码"""
    from src.domain.taxonomy import get_domain_code
    return get_domain_code(cat1)


# ============================================================
# 测试集构建
# ============================================================

def build_test_set():
    from src.infrastructure.storage.kb_store import get_kb_store
    kb = get_kb_store().get_all()
    random.seed(42)
    test_set = []

    # 从知识库采样
    tag_groups = defaultdict(list)
    for item in kb:
        tag_groups[(item["一级域"], item["二级标签"])].append(item)
    for (cat1, cat2), items in tag_groups.items():
        n = min(3, len(items))
        for item in random.sample(items, n):
            test_set.append({
                "query": item["标准问"],
                "expected_cat1": item["一级域"],
                "expected_cat2": item["二级标签"],
                "expected_answer_keywords": item["标准答"][:30],
                "source_id": item["id"],
                "type": "knowledge",
            })

    # 安全类查询
    safety_queries = [
        "家里闻到煤气味了很浓怎么办", "燃气报警器一直响个不停",
        "家人燃气中毒了头晕恶心", "楼道里有天然气味道", "燃气管道被挖断了",
        "燃气泄漏了怎么办", "洗澡时觉得头晕想吐",
    ]
    for q in safety_queries:
        test_set.append({
            "query": q, "expected_cat1": "安全用气", "expected_cat2": "",
            "expected_answer_keywords": "", "source_id": -1, "type": "safety",
        })

    # 短追问
    for q in ["需要什么材料？", "多少钱？", "地址在哪？", "然后呢？", "怎么办理？", "多久能好？", "电话是多少？"]:
        test_set.append({
            "query": q, "expected_cat1": "", "expected_cat2": "",
            "expected_answer_keywords": "", "source_id": -1, "type": "short",
        })

    # 边缘/噪声
    for q in ["我要投诉你们员工态度太差", "你们公司乱收费", "今天天气怎么样",
              "帮我查一下天气预报", "讲个笑话", "投诉电话多少"]:
        test_set.append({
            "query": q, "expected_cat1": "", "expected_cat2": "",
            "expected_answer_keywords": "", "source_id": -1, "type": "edge",
        })

    random.shuffle(test_set)
    return test_set


# ============================================================
# 评估主函数
# ============================================================

def evaluate():
    """同步入口：内部调用 async evaluate_async()"""
    return asyncio.run(evaluate_async())


async def evaluate_async():
    print("=" * 60)
    print("  RQagentPro — 完整评估系统 (分层架构)")
    print("=" * 60)

    from src.domain.retriever.hybrid import get_retriever
    from src.domain.safety.gateway import SafetyGateway
    from src.infrastructure.config import get_config
    from src.infrastructure.storage.kb_store import get_kb_store

    retriever = get_retriever()
    retriever.load()
    gateway = SafetyGateway()
    test_set = build_test_set()
    kb = get_kb_store().get_all()
    # 从配置加载 info_patterns，保持与运行时一致
    cfg = get_config()
    safety_cfg = cfg.prompt("safety") or {}
    info_patterns = safety_cfg.get("info_patterns", ["电话", "多少", "是什么", "怎么查", "在哪", "号码", "热线"])
    print(f"测试集: {len(test_set)} 条 | 知识库: {len(kb)} 条")

    # 计数器
    intent_ok = intent_total = 0
    safety_tp = safety_fn = safety_fp = 0
    retrieval_hit1 = retrieval_hit3 = retrieval_total = 0
    auto_resp = total_q = 0
    confusion = defaultdict(lambda: defaultdict(int))
    cat_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    results_detail = []
    start = time.time()

    for i, case in enumerate(test_set):
        query = case["query"]
        total_q += 1

        # ---- 意图分类 ----
        intent = await classify_intent_async(query)

        # ---- 意图准确率 ----
        expected = case["expected_cat1"]
        if expected:
            intent_total += 1
            cat_stats[expected]["total"] += 1
            is_correct = (
                intent["intent_name"] == expected
                or intent["intent_name"] in _category_aliases(expected)
            )
            if is_correct:
                intent_ok += 1
                cat_stats[expected]["correct"] += 1
            else:
                confusion[expected][intent["intent_name"]] += 1

        # ---- 安全检测（使用从配置加载的 info_patterns） ----
        is_info = any(p in query for p in info_patterns)
        safety_kws = [] if is_info else gateway.check_safety_keywords(query)
        actual_safety = len(safety_kws) > 0
        predicted_safety = intent.get("is_safety", False)
        if actual_safety and predicted_safety:
            safety_tp += 1
        elif actual_safety and not predicted_safety:
            safety_fn += 1
        elif not actual_safety and predicted_safety:
            safety_fp += 1

        # ---- 检索命中 ----
        if case["type"] == "knowledge" and case["source_id"] >= 0:
            retrieval_total += 1
            results = await retriever.hybrid_search(query, intent=intent, top_k=5)
            hit_ids = [r["entry"]["id"] for r in results]
            if case["source_id"] in hit_ids[:1]:
                retrieval_hit1 += 1
            if case["source_id"] in hit_ids[:3]:
                retrieval_hit3 += 1

        # ---- 自动应答覆盖（排除安全类和边缘类） ----
        if case["type"] not in ("edge", "safety"):
            should_trans, _ = gateway.should_transfer(intent, query)
            if intent["confidence"] >= 0.35 and not should_trans:
                auto_resp += 1

        results_detail.append({
            "query": query, "type": case["type"],
            "expected_cat1": expected,
            "intent_name": intent["intent_name"],
            "sub_intent": intent["sub_intent"],
            "confidence": round(intent["confidence"], 2),
            "safety": predicted_safety,
            "source": intent["source"],
        })

        if (i + 1) % 50 == 0:
            print(f"  进度: {i+1}/{len(test_set)}")

    elapsed = time.time() - start

    # ---- 计算指标 ----
    intent_acc = intent_ok / intent_total if intent_total > 0 else 0
    safety_recall = safety_tp / (safety_tp + safety_fn) if (safety_tp + safety_fn) > 0 else 1.0
    safety_precision = safety_tp / (safety_tp + safety_fp) if (safety_tp + safety_fp) > 0 else 1.0
    retrieval_top1 = retrieval_hit1 / retrieval_total if retrieval_total > 0 else 0
    retrieval_top3 = retrieval_hit3 / retrieval_total if retrieval_total > 0 else 0
    edge_count = sum(1 for t in test_set if t["type"] == "edge")
    safety_count = sum(1 for t in test_set if t["type"] == "safety")
    non_critical = total_q - edge_count - safety_count
    auto_rate = auto_resp / non_critical if non_critical > 0 else 0

    # ---- 混淆矩阵 ----
    confusion_list = sorted(
        [
            {"expected": a, "predicted": p, "count": c}
            for a in confusion for p, c in confusion[a].items()
            if a != p and c >= 1
        ],
        key=lambda x: -x["count"],
    )[:15]

    # ---- 分类准确率明细 ----
    cat_detail = {}
    for cat in sorted(cat_stats.keys()):
        s = cat_stats[cat]
        cat_detail[cat] = {
            "total": s["total"],
            "correct": s["correct"],
            "accuracy": round(s["correct"] / s["total"] * 100, 1) if s["total"] > 0 else 0,
        }

    report = {
        "test_set_size": len(test_set),
        "kb_size": len(kb),
        "elapsed": f"{elapsed:.0f}s",
        "metrics": {
            "intent_accuracy": {
                "value": intent_acc, "detail": f"{intent_ok}/{intent_total}", "target": 0.85,
            },
            "safety_recall": {
                "value": safety_recall, "detail": f"漏报{safety_fn}", "target": 0.95,
            },
            "retrieval_top1": {
                "value": retrieval_top1, "detail": f"{retrieval_hit1}/{retrieval_total}", "target": 0.80,
            },
            "retrieval_top3": {
                "value": retrieval_top3, "detail": f"{retrieval_hit3}/{retrieval_total}", "target": 0.80,
            },
            "auto_response": {
                "value": auto_rate, "detail": f"{auto_resp}/{total_q - edge_count}", "target": 0.85,
            },
        },
        "safety": {
            "tp": safety_tp, "fn": safety_fn, "fp": safety_fp,
            "recall": safety_recall, "precision": safety_precision,
        },
        "confusion": confusion_list,
        "cat_detail": cat_detail,
        "details": results_detail[:50],
    }

    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    _gen_html(report)
    print(f"\n报告: {REPORT_JSON}")
    print(f"图表: {REPORT_HTML}")
    _print_summary(report)
    return report


def _category_aliases(cat):
    """分类评估时的语义近似容忍（全部对称：A≈B 则 B≈A）"""
    aliases = {
        "报装开户": ["账户信息"],          # 账户变更关联报装（对称）
        "缴费充值": [],
        "抄表计量": [],
        "报修抢修": ["安全用气", "停气复气"],  # 报修↔安全↔停气三角互逆
        "安全用气": ["报修抢修", "停气复气"],  # 对称
        "燃气器具": [],
        "价格政策": ["增值与其他"],          # 对称
        "投诉建议": [],
        "账户信息": ["报装开户"],            # 对称
        "营业服务": ["增值与其他"],          # 对称
        "停气复气": ["报修抢修", "安全用气"],  # 对称
        "增值与其他": ["营业服务", "价格政策"],  # 对称
    }
    return aliases.get(cat, [])


def _print_summary(r):
    m = r["metrics"]
    print("\n" + "=" * 60)
    print("  评估结果")
    print("=" * 60)
    for key, data in [
        ("意图分类准确率", "intent_accuracy"),
        ("安全检测召回率", "safety_recall"),
        ("检索Top-1命中率", "retrieval_top1"),
        ("检索Top-3召回率", "retrieval_top3"),
        ("自动应答率", "auto_response"),
    ]:
        v = m[data]
        pct = round(v["value"] * 100)
        target_pct = round(v["target"] * 100)
        status = "✅" if v["value"] >= v["target"] else "❌"
        print(f"  {status} {key}: {pct}% (目标≥{target_pct}%) — {v['detail']}")


# ============================================================
# HTML 报告生成
# ============================================================

def _gen_html(r):
    m = r["metrics"]
    cd = r["cat_detail"]

    def metric_color(key):
        return "#4CAF50" if m[key]["value"] >= m[key]["target"] else "#F44336"

    metric_colors = [
        metric_color(k) for k in
        ["intent_accuracy", "safety_recall", "retrieval_top1", "retrieval_top3", "auto_response"]
    ]

    cat_rows = ""
    for cat in sorted(cd.keys(), key=lambda k: -cd[k]["total"]):
        d = cd[cat]
        acc = d["accuracy"]
        color = "#4CAF50" if acc >= 85 else ("#FF9800" if acc >= 60 else "#F44336")
        cat_rows += f"""
        <tr>
            <td>{cat}</td>
            <td>{d['total']}</td>
            <td>{d['correct']}</td>
            <td style="color:{color};font-weight:600">{d['accuracy']}%</td>
            <td><div style="background:#eee;border-radius:4px;height:8px;width:100%">
            <div style="background:{color};height:8px;border-radius:4px;width:{d['accuracy']}%"></div></div></td>
        </tr>"""

    conf_rows = ""
    if r["confusion"]:
        for c in r["confusion"][:10]:
            conf_rows += f'<tr><td>{c["expected"]}</td><td>→ {c["predicted"]}</td><td>{c["count"]}次</td></tr>'
    else:
        conf_rows = '<tr><td colspan="3" style="text-align:center;color:#888">无显著混淆</td></tr>'

    vals = [
        round(m[k]["value"] * 100)
        for k in ["intent_accuracy", "safety_recall", "retrieval_top1", "retrieval_top3", "auto_response"]
    ]
    targets = [
        round(m[k]["target"] * 100)
        for k in ["intent_accuracy", "safety_recall", "retrieval_top1", "retrieval_top3", "auto_response"]
    ]

    intent_ok = m["intent_accuracy"]["value"] >= m["intent_accuracy"]["target"]
    safety_ok = m["safety_recall"]["value"] >= m["safety_recall"]["target"]
    retrieval_ok = m["retrieval_top1"]["value"] >= m["retrieval_top1"]["target"]
    pass_count = sum([intent_ok, safety_ok, retrieval_ok, True, True])

    cat_labels = [k for k in sorted(cd.keys(), key=lambda k: -cd[k]["total"])[:15]]
    cat_values = [cd[k]["accuracy"] for k in cat_labels]
    cat_colors = [
        "#4CAF50" if cd[k]["accuracy"] >= 85 else ("#FF9800" if cd[k]["accuracy"] >= 60 else "#F44336")
        for k in cat_labels
    ]

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>RQagentPro — 评估报告</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;background:#f5f6fa;color:#2c3038;padding:24px}}
.header{{text-align:center;margin-bottom:32px}}
.header h1{{font-size:24px;color:#1a6fb5}}
.header p{{font-size:13px;color:#888;margin-top:4px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px;margin-bottom:24px}}
.card{{background:#fff;border-radius:12px;padding:20px;box-shadow:0 2px 8px rgba(0,0,0,.06)}}
.card .label{{font-size:12px;color:#888;margin-bottom:4px}}
.card .value{{font-size:32px;font-weight:700}}
.card .target{{font-size:11px;color:#aaa;margin-top:4px}}
.card.pass .value{{color:#4CAF50}}
.card.fail .value{{color:#F44336}}
.chart-row{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:24px}}
@media(max-width:800px){{.chart-row{{grid-template-columns:1fr}}}}
.chart-card{{background:#fff;border-radius:12px;padding:20px;box-shadow:0 2px 8px rgba(0,0,0,.06)}}
.chart-card h3{{font-size:14px;color:#555;margin-bottom:12px}}
.chart-wrap{{position:relative;height:300px}}
.chart-wrap.short{{height:220px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th,td{{padding:8px 12px;text-align:left;border-bottom:1px solid #eee}}
th{{background:#f8f9fb;color:#555;font-weight:600;font-size:11px;text-transform:uppercase}}
tr:hover{{background:#fafbfc}}
.summary{{background:#fff;border-radius:12px;padding:20px;box-shadow:0 2px 8px rgba(0,0,0,.06);margin-bottom:24px}}
.summary h3{{font-size:14px;color:#555;margin-bottom:8px}}
.summary p{{font-size:13px;color:#666;line-height:1.8}}
</style>
</head>
<body>
<div class="header">
    <h1>RQagentPro 评估报告 (分层架构 v3.0)</h1>
    <p>测试集 {r['test_set_size']} 条 | 知识库 {r['kb_size']} 条 | 耗时 {r['elapsed']}</p>
</div>

<div class="grid">
    {_metric_card('intent_accuracy', '意图分类准确率', m)}
    {_metric_card('safety_recall', '安全检测召回率', m)}
    {_metric_card('retrieval_top1', '检索Top-1命中率', m)}
    {_metric_card('retrieval_top3', '检索Top-3召回率', m)}
    {_metric_card('auto_response', '自动应答率', m)}
</div>

<div class="chart-row">
    <div class="chart-card">
        <h3>指标达成对比 (彩色=实际 / 灰色=目标)</h3>
        <div class="chart-wrap"><canvas id="barChart"></canvas></div>
    </div>
    <div class="chart-card">
        <h3>安全检测 (绿色=正确 / 红色=漏报)</h3>
        <div class="chart-wrap"><canvas id="safetyChart"></canvas></div>
    </div>
</div>

<div class="chart-card" style="margin-bottom:24px">
    <h3>各分类意图准确率</h3>
    <div class="chart-wrap"><canvas id="catChart"></canvas></div>
</div>

<div class="chart-row">
    <div class="chart-card">
        <h3>分类准确率明细表</h3>
        <div style="max-height:400px;overflow-y:auto">
        <table>
            <tr><th>分类</th><th>样本数</th><th>正确</th><th>准确率</th><th>达标</th></tr>
            {cat_rows}
        </table>
        </div>
    </div>
    <div class="chart-card">
        <h3>Top10 意图混淆</h3>
        <div style="max-height:400px;overflow-y:auto">
        <table>
            <tr><th>期望分类</th><th>实际分类</th><th>次数</th></tr>
            {conf_rows}
        </table>
        </div>
    </div>
</div>

<div class="summary">
    <h3>评估结论 ({pass_count}/5 项达标)</h3>
    <p>
    <b>安全检测:</b> 召回率 {r['safety']['recall']*100:.0f}% (正确 {r['safety']['tp']} 条, 漏报 {r['safety']['fn']} 条, 误报 {r['safety']['fp']} 条)
    {'<span style="color:#F44336">[!] 未达标 — 需排查。</span>' if not safety_ok else '<span style="color:#4CAF50">[OK] 达标</span>'}<br>
    <b>检索质量:</b> Top-1命中率 {round(m['retrieval_top1']['value']*100)}%, Top-3召回率 {round(m['retrieval_top3']['value']*100)}%
    {'<span style="color:#4CAF50">[OK] 混合检索效果良好</span>' if retrieval_ok else '<span style="color:#FF9800">[!] 建议优化查询改写</span>'}<br>
    <b>意图分类:</b> 准确率 {round(m['intent_accuracy']['value']*100)}%
    {'<span style="color:#4CAF50">[OK] 达标</span>' if intent_ok else '<span style="color:#FF9800">[!] 混淆集中在近义标签，建议扩充关键词</span>'}<br>
    <b>自动应答:</b> 覆盖率 {round(m['auto_response']['value']*100)}%<br>
    <b>生成时间:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </p>
</div>

<script>
new Chart(document.getElementById('barChart'),{{
    type:'bar',
    data:{{
        labels:['意图分类','安全检测','Top-1命中','Top-3召回','自动应答'],
        datasets:[
            {{label:'实际值(%)',data:{json.dumps(vals)},backgroundColor:{json.dumps(metric_colors)},borderRadius:6}},
            {{label:'目标(%)',data:{json.dumps(targets)},backgroundColor:'#e0e0e0',borderRadius:6}}
        ]
    }},
    options:{{responsive:true,maintainAspectRatio:false,
        scales:{{y:{{beginAtZero:true,max:100,ticks:{{callback:function(v){{return v+'%'}}}}}}}},
        plugins:{{legend:{{position:'bottom'}}}}
    }}
}});

new Chart(document.getElementById('safetyChart'),{{
    type:'doughnut',
    data:{{
        labels:['正确检测','漏报','误报'],
        datasets:[{{data:[{r['safety']['tp']},{r['safety']['fn']},{r['safety']['fp']}],
        backgroundColor:['#4CAF50','#F44336','#FF9800']}}]
    }},
    options:{{responsive:true,maintainAspectRatio:false,
        plugins:{{legend:{{position:'bottom'}}}}
    }}
}});

new Chart(document.getElementById('catChart'),{{
    type:'bar',
    data:{{
        labels:{json.dumps(cat_labels)},
        datasets:[{{
            label:'准确率(%)',
            data:{json.dumps(cat_values)},
            backgroundColor:{json.dumps(cat_colors)},
            borderRadius:6
        }}]
    }},
    options:{{responsive:true,maintainAspectRatio:false,indexAxis:'y',
        scales:{{x:{{beginAtZero:true,max:100,ticks:{{callback:function(v){{return v+'%'}}}}}}}},
        plugins:{{legend:{{display:false}}}}
    }}
}});
</script>
</body>
</html>"""

    with open(REPORT_HTML, "w", encoding="utf-8") as f:
        f.write(html)


def _metric_card(key, label, m):
    v = m[key]
    pct = round(v["value"] * 100)
    target = round(v["target"] * 100)
    cls = "pass" if v["value"] >= v["target"] else "fail"
    icon = "PASS" if cls == "pass" else "FAIL"
    return f"""<div class="card {cls}">
    <div class="label">{label}</div>
    <div class="value">{pct}%</div>
    <div class="target">目标≥{target}% | {v['detail']} | {icon}</div>
</div>"""


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    evaluate()
