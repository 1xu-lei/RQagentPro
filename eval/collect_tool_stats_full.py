"""Collect tool call stats from the full eval test set (knowledge type only)."""
import asyncio, json, os, sys, time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
# Load .env
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

from src.application.chat_service import ChatService
from eval.evaluate import build_test_set

async def main():
    test_set = build_test_set()
    # Only knowledge-type queries (safety gets hard-precheck bypass, short/edge are not normal)
    knowledge_cases = [c for c in test_set if c["type"] == "knowledge"]
    print(f"Test set: {len(test_set)} total, {len(knowledge_cases)} knowledge-type queries")

    service = ChatService()
    tool_counter = Counter()
    total_tc = 0
    total_iter = 0
    errors = 0
    results = []

    t0 = time.time()
    for i, case in enumerate(knowledge_cases):
        query = case["query"]
        try:
            result = await service.chat(query, session_id="tool_stats_full")
            tc = result.get("tool_calls_count", 0)
            total_tc += tc
            total_iter += result.get("agent_iterations", 0)
            for call in result.get("tool_calls_detail", []):
                tool_counter[call.get("tool", "unknown")] += 1
            results.append({"query": query, "tc": tc, "src": result.get("source", "?")})
            if (i + 1) % 30 == 0:
                elapsed = time.time() - t0
                print(f"  [{i+1:3d}/{len(knowledge_cases)}] {total_tc} calls from {i+1-errors} ok, {elapsed:.0f}s elapsed")
        except Exception as e:
            errors += 1
            results.append({"query": query, "error": str(e)[:80]})

    elapsed = time.time() - t0
    total = sum(tool_counter.values())

    # Build table
    print(f"\nDone in {elapsed:.0f}s: {len(knowledge_cases)} queries, {errors} errors, {total} tool calls")
    print(f"Avg calls/query: {total_tc/len(knowledge_cases):.1f}, avg iterations: {total_iter/len(knowledge_cases):.1f}")
    print()
    print("| 工具名称 | 调用次数 | 占比 | 说明 |")
    print("|---------|:------:|:---:|------|")
    for name, cnt in tool_counter.most_common():
        pct = cnt / total * 100 if total else 0
        desc = {
            "search_knowledge_base": "知识检索，首轮工具调用",
            "lookup_regulation": "法规追溯，检索后自动查询FG编码",
            "get_service_info": "基础信息查询（电话/地址/时间）",
            "get_related_questions": "相关问题推荐",
            "classify_user_intent": "意图分类",
            "check_safety": "安全语义判断（硬预检未触发时）",
            "request_human_transfer": "转人工",
        }.get(name, "")
        print(f"| {name} | {cnt} | {pct:.1f}% | {desc} |")
    print(f"| **合计** | **{total}** | **100%** | |")
    print()
    print(f"Avg tools/query: {total_tc/len(knowledge_cases):.1f}, avg iterations: {total_iter/len(knowledge_cases):.1f}")

    # Save
    output = {
        "queries": len(knowledge_cases),
        "errors": errors,
        "total_tool_calls": total,
        "avg_calls_per_query": round(total_tc / len(knowledge_cases), 1),
        "avg_iterations": round(total_iter / len(knowledge_cases), 1),
        "distribution": {name: cnt for name, cnt in tool_counter.most_common()},
        "elapsed_s": round(elapsed),
    }
    out_path = Path(__file__).parent / "tool_stats_full.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    asyncio.run(main())
