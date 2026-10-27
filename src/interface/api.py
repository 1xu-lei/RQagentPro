"""
FastAPI 应用工厂 — 组装所有路由和中间件

启动方式：
    uvicorn src.interface.api:create_app --factory
"""

import sys
import os
from pathlib import Path

# 确保项目根目录在 path 中
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, FileResponse, HTMLResponse, JSONResponse

from src.infrastructure.config import get_config
from src.infrastructure.logging import get_logger, setup_logging
from src.interface.routes import chat, session, knowledge, company
from src.interface.middleware.error_handler import ErrorHandlerMiddleware
from src.interface.middleware.request_log import RequestLogMiddleware
from src.interface.schemas.chat import HealthResponse, StatsResponse

logger = get_logger(__name__)


def create_app() -> FastAPI:
    """创建并配置 FastAPI 应用"""

    cfg = get_config()

    # 初始化日志
    setup_logging(cfg.get("logging", {}))

    # 创建应用
    app = FastAPI(
        title=cfg.get("app.name", "RQagentPro"),
        description=cfg.get("app.description", "衡阳燃气AI客服系统"),
        version=cfg.get("app.version", "3.0.0"),
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.get("server.cors_origins", ["*"]),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 自定义中间件
    app.add_middleware(ErrorHandlerMiddleware)
    app.add_middleware(RequestLogMiddleware)

    # 注册路由
    app.include_router(chat.router)
    app.include_router(session.router)
    app.include_router(knowledge.router)
    app.include_router(company.router)

    # 静态文件（Web 前端）
    web_dir = PROJECT_ROOT / "web"
    if web_dir.exists():
        app.mount("/web", StaticFiles(directory=str(web_dir), html=True), name="web")

    # 知识图谱可视化
    graph_html = PROJECT_ROOT / "data" / "graph_viz.html"
    if graph_html.exists():
        @app.get("/graph")
        async def graph_viewer():
            return FileResponse(str(graph_html))

    # ---- 基础路由 ----

    @app.get("/")
    async def root():
        return RedirectResponse(url="/web/")

    @app.get("/api/health", response_model=HealthResponse)
    async def health_check():
        """健康检查"""
        from src.infrastructure.storage.kb_store import get_kb_store
        kb_entries = get_kb_store().count()

        return HealthResponse(
            status="ok",
            version=cfg.get("app.version", "3.1.0"),
            architecture="Interface → Application → Domain → Infrastructure",
            engines_ready=True,
            kb_entries=kb_entries,
            llm_connected=bool(sys.modules.get("src.infrastructure.llm")),
        )

    @app.get("/api/stats", response_model=StatsResponse)
    async def get_stats():
        """系统统计"""
        from src.application.session_service import get_session_service
        from src.infrastructure.storage.kb_store import get_kb_store

        kb_entries = get_kb_store().count()

        graph_nodes = 0
        try:
            from src.domain.knowledge_graph.graph import get_graph_query
            gq = get_graph_query()
            if gq:
                graph_nodes = gq.g.number_of_nodes()
        except Exception as e:
            logger.warning(f"Graph stats query failed: {e}")

        session_mgr = get_session_service()

        return StatsResponse(
            active_sessions=len(session_mgr.sessions),
            knowledge_base_size=kb_entries,
            graph_nodes=graph_nodes,
        )

    @app.get("/api/debug")
    async def debug_check():
        """诊断各组件状态"""
        import os
        results = {}

        # 知识库
        try:
            from src.infrastructure.storage.kb_store import get_kb_store
            store = get_kb_store()
            results["knowledge_base"] = {
                "status": "ok",
                "entries": store.count(),
                "backend": "sqlite3",
            }
        except Exception as e:
            results["knowledge_base"] = {"status": "error", "msg": str(e)[:100]}

        # 工具注册表
        try:
            from src.domain.tools.registry import get_tool_registry
            registry = get_tool_registry()
            results["tools"] = {
                "status": "ok",
                "registered": registry.list_tools(),
            }
        except Exception as e:
            results["tools"] = {"status": "error", "msg": str(e)[:100]}

        # DeepSeek API 连通性
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            results["deepseek_api"] = {"status": "error", "msg": "DEEPSEEK_API_KEY 未配置"}
        else:
            results["deepseek_api"] = {"status": "ok", "model": cfg.get("llm.model")}

        # 会话数
        try:
            from src.application.session_service import get_session_service
            session_mgr = get_session_service()
            results["sessions"] = {"status": "ok", "active": len(session_mgr.sessions)}
        except Exception as e:
            results["sessions"] = {"status": "error", "msg": str(e)[:100]}

        # 检索缓存统计
        try:
            from src.infrastructure.embedding import get_embedder
            emb = get_embedder()
            if emb._loaded:
                results["embed_cache"] = emb.cache_stats
        except Exception as e:
            results["embed_cache"] = {"status": "error", "msg": str(e)[:100]}

        return results

    # 启动事件
    @app.on_event("startup")
    async def startup():
        """应用启动时的初始化"""
        logger.info("=" * 50)
        logger.info(f"RQagentPro v{cfg.get('app.version')} starting...")
        logger.info(f"Architecture: Interface → Application → Domain → Infrastructure")
        logger.info("=" * 50)

        # 初始化 Span 追踪
        from src.infrastructure.tracing import init_tracer
        trace_dir = str(PROJECT_ROOT / "data" / "traces")
        init_tracer(enabled=True, trace_dir=trace_dir)
        logger.info(f"Span tracing enabled, traces → {trace_dir}")

        # 注册工具
        from src.domain.tools.init_tools import register_all_tools
        register_all_tools()

        # 预热检索引擎（含 BGE 模型 + jieba + FAISS 真实搜索）
        try:
            from src.domain.retriever.hybrid import get_retriever
            retriever = get_retriever()
            retriever.load()
            logger.info(f"Retriever ready: {len(retriever.kb)} entries")

            # ★ 真实搜索预热：完成 BGE 编码 + FAISS 搜索 + jieba 分词
            # 避免首个请求的 15-20s 冷启动延迟
            logger.info("Pre-warming BGE model with real search...")
            _ = await retriever.hybrid_search("预热查询 燃气开户", top_k=3)
            logger.info("BGE model fully warmed up")
        except Exception as e:
            logger.warning(f"Retriever warmup failed: {e}")

        # 预热知识图谱
        try:
            from src.domain.knowledge_graph.graph import get_graph_query
            gq = get_graph_query()
            if gq:
                logger.info(f"Graph ready: {gq.g.number_of_nodes()} nodes")
        except Exception as e:
            logger.warning(f"Graph warmup failed: {e}")

        logger.info("Startup complete. Ready to accept requests.")

        # 预热完成后打印访问链接
        port = os.environ.get("RQAGENT_PORT", "8001")
        print(f"\n{'='*60}")
        print(f"  服务已启动")
        print(f"  {'─'*40}")
        print(f"  对话界面    http://localhost:{port}")
        print(f"  调用链追踪  http://localhost:{port}/api/traces")
        print(f"  知识图谱    http://localhost:{port}/graph")
        print(f"  API 文档    http://localhost:{port}/docs")
        print(f"  {'='*60}\n")

    @app.on_event("shutdown")
    async def shutdown():
        """应用关闭时的清理"""
        logger.info("Shutting down...")

        # 关闭 LLM 客户端连接
        try:
            from src.domain.agent.engine import get_agent
            agent = get_agent()
            await agent.close()
            logger.info("LLM client closed")
        except Exception as e:
            logger.warning(f"LLM client cleanup failed: {e}")

        # 关闭检索线程池
        try:
            from src.domain.retriever.hybrid import _search_executor
            _search_executor.shutdown(wait=True)
            logger.info("Search thread pool shut down")
        except Exception as e:
            logger.warning(f"Thread pool cleanup failed: {e}")

        # 保存所有活跃会话
        try:
            from src.application.session_service import get_session_service
            session_mgr = get_session_service()
            for sid in list(session_mgr.sessions.keys()):
                session_mgr.save_session(sid)
            logger.info(f"All sessions saved ({len(session_mgr.sessions)} active)")
        except Exception as e:
            logger.warning(f"Session cleanup failed: {e}")

        logger.info("Shutdown complete.")

    # ----------------------------------------------------------
    # Span 追踪查看接口
    # ----------------------------------------------------------

    @app.get("/api/traces", response_class=HTMLResponse)
    async def view_traces():
        """HTML 页面：查看最近的调用链追踪"""
        from src.infrastructure.tracing import get_tracer

        tracer = get_tracer()
        traces = tracer.store.list(n=50)

        return _render_traces_html(traces)

    @app.get("/api/traces/json")
    async def get_traces_json(limit: int = Query(default=20, le=100)):
        """JSON API：获取最近的 trace 数据"""
        from src.infrastructure.tracing import get_tracer

        tracer = get_tracer()
        traces = tracer.store.list(n=limit)

        return JSONResponse({
            "total": tracer.store.count,
            "traces": [t.to_dict() for t in traces],
        })

    @app.get("/api/traces/{trace_id}")
    async def get_trace_detail(trace_id: str):
        """获取单条 trace 详情"""
        from src.infrastructure.tracing import get_tracer

        tracer = get_tracer()
        trace = tracer.store.get(trace_id)

        if trace is None:
            return JSONResponse({"error": "trace not found"}, status_code=404)

        return JSONResponse(trace.to_dict())

    return app


# ============================================================
# Traces HTML 渲染器（内嵌，不依赖外部文件）
# ============================================================

def _render_traces_html(traces) -> str:
    """生成查看 Trace 的 HTML 页面"""

    trace_cards = ""
    for t in traces:
        if t.root_span is None:
            continue
        children_html = _render_span_html(t.root_span, 0)

        trace_cards += f"""
        <div class="trace-card">
            <div class="trace-header" onclick="this.parentElement.classList.toggle('expanded')">
                <div class="trace-meta">
                    <span class="trace-time">{t.start_time_str}</span>
                    <span class="trace-duration">{t.duration_ms}ms</span>
                    <span class="trace-session">{t.session_id or '-'}</span>
                </div>
                <div class="trace-query">{t.user_query[:120] or '(no query)'}</div>
                <span class="trace-id">{t.trace_id}</span>
            </div>
            <div class="trace-tree">
                {children_html}
            </div>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RQagentPro — Trace Viewer</title>
<style>
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    body {{ font-family: 'Segoe UI', Consolas, monospace; background:#0d1117; color:#c9d1d9; padding:20px; }}
    .header {{ margin-bottom:24px; border-bottom:1px solid #30363d; padding-bottom:16px; }}
    .header h1 {{ font-size:22px; color:#58a6ff; }}
    .header .sub {{ font-size:13px; color:#8b949e; margin-top:4px; }}
    .toolbar {{ display:flex; gap:8px; margin-bottom:16px; }}
    .toolbar button {{ background:#21262d; border:1px solid #30363d; color:#c9d1d9; padding:6px 14px; border-radius:6px; cursor:pointer; font-size:13px; }}
    .toolbar button:hover {{ background:#30363d; }}
    .trace-card {{ background:#161b22; border:1px solid #30363d; border-radius:8px; margin-bottom:12px; overflow:hidden; }}
    .trace-header {{ padding:12px 16px; cursor:pointer; display:flex; flex-wrap:wrap; gap:8px; align-items:center; }}
    .trace-header:hover {{ background:#1c2128; }}
    .trace-meta {{ display:flex; gap:12px; align-items:center; }}
    .trace-time {{ color:#8b949e; font-size:12px; }}
    .trace-duration {{ background:#23863622; color:#3fb950; padding:2px 8px; border-radius:4px; font-size:12px; font-weight:bold; }}
    .trace-session {{ color:#8b949e; font-size:11px; max-width:120px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
    .trace-query {{ color:#c9d1d9; font-size:13px; flex:1; min-width:200px; }}
    .trace-id {{ color:#484f58; font-size:10px; }}
    .trace-tree {{ display:none; padding:8px 16px 16px 16px; border-top:1px solid #30363d; background:#0d1117; }}
    .trace-card.expanded .trace-tree {{ display:block; }}
    .span-row {{ display:flex; align-items:baseline; padding:2px 0; font-size:13px; white-space:nowrap; }}
    .span-indent {{ display:inline-block; }}
    .span-icon {{ width:16px; color:#484f58; }}
    .span-name {{ font-weight:500; }}
    .span-name.llm {{ color:#79c0ff; }}
    .span-name.tool {{ color:#d2a8ff; }}
    .span-name.retrieve {{ color:#7ee787; }}
    .span-name.agent {{ color:#f0883e; }}
    .span-name.http {{ color:#f0883e; }}
    .span-duration {{ margin-left:8px; font-size:11px; padding:1px 6px; border-radius:3px; }}
    .span-duration.fast {{ color:#3fb950; background:#23863622; }}
    .span-duration.medium {{ color:#d29922; background:#9e6a0322; }}
    .span-duration.slow {{ color:#f85149; background:#da363322; }}
    .span-attrs {{ margin-left:8px; color:#8b949e; font-size:11px; }}
    .span-error {{ color:#f85149; margin-left:8px; font-size:11px; }}
    .empty {{ text-align:center; padding:40px; color:#484f58; }}
</style>
</head>
<body>
<div class="header">
    <h1>🔍 RQagentPro Trace Viewer</h1>
    <div class="sub">最近 {len(traces)} 条调用链追踪 · 自动刷新: 5 秒</div>
</div>
<div class="toolbar">
    <button onclick="location.reload()">🔄 刷新</button>
    <button onclick="location.href='/api/traces/json'">📋 JSON</button>
</div>
{"".join(trace_cards) if trace_cards else '<div class="empty">暂无 Trace 数据<br><small>发起一次对话后回来查看</small></div>'}
<script>
    // 默认展开第一条
    (function() {{
        const first = document.querySelector('.trace-card');
        if (first) first.classList.add('expanded');
    }})();
    // 每 5 秒自动刷新，便于观察实时调用链
    setInterval(function() {{ location.reload(); }}, 5000);
</script>
</body>
</html>"""


def _render_span_html(span, depth: int) -> str:
    """递归渲染 Span 树为 HTML"""
    indent = '<span class="span-indent" style="width:' + str(depth * 20) + 'px"></span>'

    # 图标
    icon = "├─"
    if not span.children:
        icon = "└─"
    if depth == 0:
        indent = ""
        icon = "●"

    # 名称样式
    css_class = ""
    if "llm" in span.name:
        css_class = "llm"
    elif "tool" in span.name:
        css_class = "tool"
    elif any(kw in span.name for kw in ("search", "retrieve", "faiss", "bm25")):
        css_class = "retrieve"
    elif any(kw in span.name for kw in ("agent", "react")):
        css_class = "agent"
    elif "http" in span.name:
        css_class = "http"

    # 耗时样式
    ms = span.duration_ms
    dur_class = "fast" if ms < 200 else ("medium" if ms < 1000 else "slow")

    # 属性
    attrs = ""
    if span.attrs:
        parts = []
        for k, v in span.attrs.items():
            if k in ("error", "result_preview", "result_len"):
                continue
            v_str = str(v)
            if len(v_str) > 60:
                v_str = v_str[:57] + "..."
            parts.append(f"{k}={v_str}")
        if parts:
            attrs = '<span class="span-attrs">' + " · ".join(parts) + "</span>"

    # 错误
    error_html = ""
    if span.status == "error" and "error" in span.attrs:
        error_html = f'<span class="span-error">❌ {span.attrs["error"][:80]}</span>'

    row = f"""<div class="span-row">
        {indent}<span class="span-icon">{icon}</span>
        <span class="span-name {css_class}">{span.name}</span>
        <span class="span-duration {dur_class}">{ms}ms</span>
        {attrs}{error_html}
    </div>"""

    for child in span.children:
        row += _render_span_html(child, depth + 1)

    return row

