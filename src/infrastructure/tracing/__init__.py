"""
轻量级 Span 追踪系统

基于 contextvars 的异步安全调用链追踪，零外部依赖。

用法:
    from src.infrastructure.tracing import get_tracer, trace

    @trace("search_knowledge_base")
    async def search(query: str) -> List[Dict]:
        ...

    # 或手动：
    with get_tracer().span("my_operation", key="value") as span:
        result = await do_something()
        span.set_attr("rows", len(result))
"""

from .tracer import Tracer, Span, get_tracer, init_tracer, trace, TraceExporter

__all__ = ["Tracer", "Span", "get_tracer", "init_tracer", "trace", "TraceExporter"]
