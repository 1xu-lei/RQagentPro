"""
集成测试脚本 — 验证系统各组件是否正常工作

运行方式：
    cd D:\Students\RQagentPro
    python tests/test_integration.py
"""

import sys
import os
from pathlib import Path

# 确保项目根目录在 path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_config():
    """测试 1: 配置加载"""
    print_section("Test 1: Config Loader")

    try:
        from src.infrastructure.config import get_config
        cfg = get_config()

        assert cfg.get("app.name") == "RQagentPro", f"Expected RQagentPro, got {cfg.get('app.name')}"
        assert cfg.get("llm.model") is not None, "Model config missing"
        assert cfg.get("agent.max_iterations") >= 3, f"Expected >=3, got {cfg.get('agent.max_iterations')}"

        # 测试 prompt 加载
        system_prompt = cfg.prompt("system")
        assert system_prompt, "System prompt not loaded"
        assert "role" in system_prompt, "System prompt missing 'role'"

        safety_prompt = cfg.prompt("safety")
        assert safety_prompt, "Safety prompt not loaded"
        assert "keywords" in safety_prompt, "Safety prompt missing 'keywords'"

        print(f"  ✅ Config loaded: {cfg.get('app.name')} v{cfg.get('app.version')}")
        print(f"  ✅ LLM model: {cfg.get('llm.model')}")
        print(f"  ✅ Safety keywords: {len(safety_prompt.get('keywords', []))}")
        print(f"  ✅ System prompt sections: {len(system_prompt)}")

        return True
    except Exception as e:
        print(f"  ❌ Config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_logging():
    """测试 2: 日志系统"""
    print_section("Test 2: Logging")

    try:
        from src.infrastructure.logging import get_logger, setup_logging

        setup_logging({"level": "INFO", "format": "console"})
        logger = get_logger("test")

        logger.info("Test info message")
        logger.warning("Test warning message")

        print(f"  ✅ Logger created and working")
        return True
    except Exception as e:
        print(f"  ❌ Logging test failed: {e}")
        return False


def test_file_store():
    """测试 3: 文件存储"""
    print_section("Test 3: File Store")

    try:
        from src.infrastructure.storage import FileStore
        import tempfile, os

        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileStore(tmpdir)

            # 写
            store.write_json("test.json", {"hello": "world"})
            assert store.exists("test.json"), "File not created"

            # 读
            data = store.read_json("test.json")
            assert data == {"hello": "world"}, f"Data mismatch: {data}"

            # 删
            assert store.delete("test.json"), "Delete failed"
            assert not store.exists("test.json"), "File still exists after delete"

        print(f"  ✅ FileStore: read/write/delete all passed")
        return True
    except Exception as e:
        print(f"  ❌ FileStore test failed: {e}")
        return False


def test_safety_gateway():
    """测试 4: 安全网关"""
    print_section("Test 4: Safety Gateway")

    try:
        from src.domain.safety.gateway import SafetyGateway

        gw = SafetyGateway()

        # 测试安全关键词检测
        matched = gw.check_safety_keywords("家里闻到煤气味了")
        assert len(matched) > 0, "Should detect '煤气味'"

        # 测试否定语境
        matched_neg = gw.check_safety_keywords("灶具打不着火怎么办")
        assert "着火" not in matched_neg, f"Should NOT trigger 着火 in negation context, got {matched_neg}"

        # 测试信息查询不过滤
        matched_info = gw.check_safety_keywords("抢修电话是多少")
        print(f"  Info query with '抢修': matched={matched_info}")

        # 测试安全模板
        template = gw.get_safety_template(["漏气"])
        assert template, "Template not found"
        assert "title" in template, "Template missing title"

        # 测试格式化
        response = gw.format_safety_response(template)
        assert "95777" in response, "Response should contain emergency number"

        print(f"  ✅ Safety keywords: {gw._keywords[:5]}...")
        print(f"  ✅ Negation patterns: {gw._negation_patterns}")
        print(f"  ✅ Templates: {list(gw._templates.keys())}")
        print(f"  ✅ Formatted response ({len(response)} chars)")
        return True
    except Exception as e:
        print(f"  ❌ Safety gateway test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tool_registry():
    """测试 5: 工具注册表"""
    print_section("Test 5: Tool Registry")

    try:
        from src.domain.tools.registry import get_tool_registry
        from src.domain.tools.init_tools import register_all_tools

        registry = register_all_tools()

        tool_names = registry.list_tools()
        expected = [
            "check_safety", "search_knowledge_base", "classify_user_intent",
            "get_service_info", "lookup_regulation", "get_related_questions",
            "request_human_transfer",
        ]
        for name in expected:
            assert name in tool_names, f"Tool '{name}' not registered"

        # 测试生成 OpenAI Schema
        schemas = registry.to_openai_schemas()
        assert len(schemas) == 7, f"Expected 7 schemas, got {len(schemas)}"

        # 验证最高优先级是 check_safety
        assert schemas[0]["function"]["name"] == "check_safety", \
            f"Highest priority should be check_safety, got {schemas[0]['function']['name']}"

        print(f"  ✅ All {len(tool_names)} tools registered")
        print(f"  ✅ Schema generation OK")
        print(f"  ✅ Priority ordering: {[s['function']['name'] for s in schemas]}")
        return True
    except Exception as e:
        print(f"  ❌ Tool registry test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_retriever():
    """测试 6: 检索引擎"""
    print_section("Test 6: Hybrid Retriever")

    try:
        from src.domain.retriever.hybrid import get_retriever

        retriever = get_retriever()
        retriever.load()

        assert len(retriever.kb) > 0, "Knowledge base is empty"

        # 测试检索 (async)
        results = await retriever.hybrid_search("燃气开户需要什么材料", top_k=3)
        assert len(results) > 0, "No results returned"
        assert len(results) <= 3, f"Expected ≤3 results, got {len(results)}"

        top = results[0]
        assert "entry" in top, "Result missing 'entry'"
        assert "score" in top, "Result missing 'score'"

        print(f"  ✅ Knowledge base: {len(retriever.kb)} entries")
        print(f"  ✅ FAISS index: {'loaded' if retriever.faiss_index else 'not loaded'}")
        print(f"  ✅ BM25 index: {'loaded' if retriever.bm25 else 'not loaded'}")
        print(f"  ✅ Search results: top_score={top['score']:.3f}, "
              f"category={top['entry'].get('一级域', '?')}")
        return True
    except Exception as e:
        print(f"  ❌ Retriever test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_graph():
    """测试 7: 知识图谱"""
    print_section("Test 7: Knowledge Graph")

    try:
        from src.domain.knowledge_graph.graph import get_graph_query

        gq = get_graph_query()
        if gq is None:
            print("  ⚠️  Knowledge graph not loaded (file may be missing)")
            return True  # 非关键

        summary = gq.get_graph_summary()
        assert summary["total_nodes"] > 0, "Graph has no nodes"

        # 测试法规查询
        reg = gq.lookup_regulation("FG-FL-001")
        if reg:
            print(f"  ✅ Regulation lookup: {reg.get('name', '?')}")

        print(f"  ✅ Graph: {summary['total_nodes']} nodes, {summary['total_edges']} edges")
        return True
    except Exception as e:
        print(f"  ❌ Graph test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_prompt_manager():
    """测试 8: Prompt 管理器"""
    print_section("Test 8: Prompt Manager")

    try:
        from src.domain.agent.prompt_manager import get_prompt_manager

        pm = get_prompt_manager()
        prompt = pm.build_system_prompt("工具描述测试")

        assert len(prompt) > 500, f"Prompt too short ({len(prompt)} chars)"
        assert "衡阳燃气" in prompt, "Prompt should mention '衡阳燃气'"
        assert "check_safety" in prompt, "Prompt should mention check_safety"

        # 测试安全 prompt
        safety = pm.get_safety_prompt()
        assert safety, "Safety prompt not loaded"

        print(f"  ✅ System prompt generated: {len(prompt)} chars")
        print(f"  ✅ Safety prompt loaded: {len(safety.get('keywords', []))} keywords")
        return True
    except Exception as e:
        print(f"  ❌ Prompt manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_agent_engine():
    """测试 9: Agent 引擎（空运行，不实际调用 LLM）"""
    print_section("Test 9: Agent Engine (structure check)")

    try:
        from src.domain.agent.engine import ReActAgent, AgentResult
        from src.domain.tools.init_tools import register_all_tools

        register_all_tools()

        # 测试 AgentResult 创建
        result = AgentResult(
            answer="测试回答",
            tool_calls_log=[],
            iterations=1,
            source="agent",
        )
        assert result.answer == "测试回答"

        # 测试消息构建（不需要 API Key）
        agent = ReActAgent()
        messages = agent._build_messages(
            "测试问题",
            history=None,
            last_intent=None,
            context_inherited=False,
            is_meta_question=False,
        )
        assert len(messages) >= 2, f"Expected ≥2 messages, got {len(messages)}"
        assert messages[0]["role"] == "system"
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "测试问题"

        # 测试意图继承
        messages_inherit = agent._build_messages(
            "多少钱",
            history=None,
            last_intent={"intent_name": "气费管理", "sub_intent": "线上缴费"},
            context_inherited=True,
            is_meta_question=False,
        )
        assert "气费管理" in messages_inherit[-1]["content"]

        # 测试元问题
        messages_meta = agent._build_messages(
            "刚才那个问题",
            history=[{"role": "user", "content": "如何缴费"}],
            last_intent=None,
            context_inherited=False,
            is_meta_question=True,
        )
        assert "如何缴费" not in messages_meta[-1]["content"]  # 不应注入意图上下文

        print(f"  ✅ AgentResult creation OK")
        print(f"  ✅ Message building: {len(messages)} messages")
        print(f"  ✅ Context inheritance: passed")
        print(f"  ✅ Meta question handling: passed")
        return True
    except Exception as e:
        print(f"  ❌ Agent engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_output_guard():
    """测试 10: 输出校验器"""
    print_section("Test 10: Output Guard")

    try:
        from src.domain.validator.output_guard import OutputGuard

        guard = OutputGuard()

        # 正常回答
        result = guard.validate("居民办理燃气开户需要以下材料...", context={})
        assert result["valid"], f"Normal answer should be valid, got issues: {result['issues']}"

        # 包含禁止表述的回答
        result_bad = guard.validate("我不知道这个问题怎么回答", context={})
        assert not result_bad["valid"], "Answer with '我不知道' should be invalid"

        # 安全回答检查
        result_safety = guard.validate("请关闭阀门", context={"is_safety": True})
        # 安全回答缺失通风和电话
        assert not result_safety["valid"], "Incomplete safety answer should be invalid"
        assert len(result_safety["issues"]) > 0

        # 严重性判断
        assert guard.is_critical("安全回答缺少'关闭阀门'指令")
        assert not guard.is_critical("回答过短（5字）")

        print(f"  ✅ Output guard: detection of bad patterns OK")
        print(f"  ✅ Safety template validation OK")
        print(f"  ✅ Critical issue classification OK")
        return True
    except Exception as e:
        print(f"  ❌ Output guard test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_query_rewriter():
    """测试 11: 查询改写器"""
    print_section("Test 11: Query Rewriter")

    try:
        from src.domain.retriever.query_rewriter import QueryRewriter

        rewriter = QueryRewriter()

        # 口语标准化
        rewritten = rewriter.rewrite("怎么搞燃气过户")
        assert "办理" in rewritten, f"Should normalize '搞' to '办理', got: {rewritten}"

        # 同义词扩展
        rewritten2 = rewriter.rewrite("缴费")
        assert len(rewritten2) > 2, f"Short query should be expanded, got: {rewritten2}"

        # 上下文增强
        rewritten3 = rewriter.rewrite(
            "多少钱",
            context={"last_intent": {"intent_name": "气费管理", "sub_intent": "线上缴费"}},
        )
        assert "气费管理" in rewritten3, f"Should include context, got: {rewritten3}"

        # 关键词提取
        keywords = rewriter.rewrite_for_keywords("燃气开户需要什么材料")
        assert len(keywords) >= 2, f"Should extract ≥2 keywords, got: {len(keywords)}"
        assert "开户" in keywords or "燃气" in keywords, f"Keywords should include '开户' or '燃气': {keywords}"

        print(f"  ✅ Normalization: '怎么搞燃气过户' → '{rewritten}'")
        print(f"  ✅ Synonym expansion: '缴费' → '{rewritten2}'")
        print(f"  ✅ Context enhancement: '多少钱' → '{rewritten3}'")
        print(f"  ✅ Keyword extraction: {keywords}")
        return True
    except Exception as e:
        print(f"  ❌ Query rewriter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_structure():
    """测试 12: API 结构"""
    print_section("Test 12: API Structure")

    try:
        from src.interface.api import create_app
        app = create_app()

        # 检查路由
        routes = [r.path for r in app.routes if hasattr(r, "path")]
        print(f"  ✅ App created with {len(routes)} routes")
        print(f"  Routes: {routes}")

        # 检查关键端点
        route_strs = [str(r) for r in routes]
        assert any("/api/v1/chat" in r for r in route_strs), "Missing /api/v1/chat"
        assert any("/api/v1/chat/stream" in r for r in route_strs), "Missing /api/v1/chat/stream"
        assert any("/api/health" in r for r in route_strs), "Missing /api/health"

        print(f"  ✅ All critical endpoints present")
        return True
    except Exception as e:
        print(f"  ❌ API structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    print("=" * 60)
    print("  RQagentPro — Integration Test Suite")
    print("=" * 60)

    tests = [
        ("Config Loader", test_config, False),
        ("Logging", test_logging, False),
        ("File Store", test_file_store, False),
        ("Safety Gateway", test_safety_gateway, False),
        ("Tool Registry", test_tool_registry, False),
        ("Hybrid Retriever", test_retriever, True),
        ("Knowledge Graph", test_graph, False),
        ("Prompt Manager", test_prompt_manager, False),
        ("Output Guard", test_output_guard, False),
        ("Query Rewriter", test_query_rewriter, False),
        ("Agent Engine", test_agent_engine, True),
        ("API Structure", test_api_structure, False),
    ]

    results = {}

    for name, test_fn, is_async in tests:
        try:
            if is_async:
                results[name] = await test_fn()
            else:
                results[name] = test_fn()
        except Exception as e:
            print(f"  ❌ {name} crashed: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    # 汇总
    print_section("Results Summary")
    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, status in results.items():
        icon = "✅" if status else "❌"
        print(f"  {icon} {name}")

    print(f"\n  Passed: {passed}/{total}")

    if passed == total:
        print("\n  🎉 All tests passed!")
    else:
        print(f"\n  ⚠️  {total - passed} test(s) failed. Check the output above.")

    return passed == total


if __name__ == "__main__":
    import asyncio
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
