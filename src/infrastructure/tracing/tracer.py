"""
Span 追踪核心 — contextvars + 异步安全 + 零外部依赖

设计:
  - 每个 HTTP 请求创建一棵 Trace 树
  - Span 之间通过 contextvars 自动关联父子关系
  - TraceStore 环形缓冲区保留最近 N 条完整 trace
  - 支持 console 树形输出 + JSON 持久化
"""

import asyncio
import contextvars
import json
import os
import time
from collections import deque
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# ============================================================
# 数据模型
# ============================================================

@dataclass
class Span:
    """单个 Span — 调用链上的一个节点"""
    name: str
    parent_id: str = ""
    span_id: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    attrs: Dict[str, Any] = field(default_factory=dict)
    children: List["Span"] = field(default_factory=list)
    status: str = "ok"  # ok | error

    @property
    def duration_ms(self) -> int:
        return int((self.end_time - self.start_time) * 1000)

    def set_attr(self, key: str, value: Any):
        self.attrs[key] = value

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "duration_ms": self.duration_ms,
            "attrs": self._serialize_attrs(),
            "children": [c.to_dict() for c in self.children],
            "status": self.status,
        }

    def _serialize_attrs(self) -> Dict:
        """确保 attrs 可 JSON 序列化"""
        result = {}
        for k, v in self.attrs.items():
            if isinstance(v, (str, int, float, bool, type(None))):
                result[k] = v
            else:
                result[k] = str(v)[:500]
        return result


@dataclass
class Trace:
    """一次完整请求的追踪树"""
    trace_id: str
    root_span: Span
    start_time: float = 0.0
    end_time: float = 0.0
    session_id: str = ""
    user_query: str = ""

    @property
    def duration_ms(self) -> int:
        return int((self.end_time - self.start_time) * 1000)

    @property
    def start_time_str(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.start_time))

    def to_dict(self) -> Dict:
        return {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "user_query": self.user_query[:200] if self.user_query else "",
            "start_time": self.start_time_str,
            "duration_ms": self.duration_ms,
            "root_span": self.root_span.to_dict(),
        }


# ============================================================
# contextvars — 异步安全的 Span 上下文传播
# ============================================================

_current_span: contextvars.ContextVar = contextvars.ContextVar("trace_current_span", default=None)
_current_trace: contextvars.ContextVar = contextvars.ContextVar("trace_current_trace", default=None)


# ============================================================
# TraceStore — 环形缓冲区
# ============================================================

class TraceStore:
    """内存中的 trace 存储（环形缓冲区）"""

    def __init__(self, max_traces: int = 200):
        self._traces: deque = deque(maxlen=max_traces)

    def add(self, trace: Trace):
        self._traces.append(trace)

    def list(self, n: int = 50) -> List[Trace]:
        items = list(self._traces)[-n:]
        items.reverse()
        return items

    def get(self, trace_id: str) -> Optional[Trace]:
        for t in self._traces:
            if t.trace_id == trace_id:
                return t
        return None

    @property
    def count(self) -> int:
        return len(self._traces)


# ============================================================
# Console 树形输出
# ============================================================

class ConsoleSpanRenderer:
    """将 Span 树渲染为彩色控制台输出"""

    COLORS = {
        "reset": "\033[0m",
        "dim": "\033[2m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "red": "\033[31m",
        "cyan": "\033[36m",
        "blue": "\033[34m",
        "bold": "\033[1m",
    }

    @classmethod
    def render(cls, trace: Trace) -> str:
        lines = []
        lines.append("")
        lines.append(f"{cls.COLORS['bold']}═══ Trace: {trace.trace_id} ═══{cls.COLORS['reset']}")
        if trace.session_id:
            lines.append(f"{cls.COLORS['dim']}    session: {trace.session_id}{cls.COLORS['reset']}")
        if trace.user_query:
            query_preview = trace.user_query[:80].replace("\n", " ")
            lines.append(f"{cls.COLORS['dim']}    query: {query_preview}{cls.COLORS['reset']}")
        lines.append(f"{cls.COLORS['dim']}    total: {trace.duration_ms}ms{cls.COLORS['reset']}")
        lines.append("")
        cls._render_span(trace.root_span, lines, prefix="", is_last=True)
        lines.append("")
        return "\n".join(lines)

    @classmethod
    def _render_span(cls, span: Span, lines: List[str], prefix: str, is_last: bool):
        connector = "└── " if is_last else "├── "
        duration_str = f"{span.duration_ms}ms"

        # 颜色
        if span.status == "error":
            color = cls.COLORS["red"]
        elif span.name.startswith("llm"):
            color = cls.COLORS["cyan"]
        elif span.name.startswith("tool"):
            color = cls.COLORS["yellow"]
        elif span.name.startswith("retrieve") or span.name.startswith("search"):
            color = cls.COLORS["green"]
        else:
            color = cls.COLORS["bold"]

        # 属性摘要
        attr_summary = cls._attr_summary(span.attrs)

        lines.append(
            f"{cls.COLORS['dim']}{prefix}{cls.COLORS['reset']}"
            f"{color}{connector}{span.name}{cls.COLORS['reset']}"
            f" {cls.COLORS['dim']}({duration_str}){cls.COLORS['reset']}"
            f"{' ' + attr_summary if attr_summary else ''}"
        )

        # 递归子节点
        child_prefix = prefix + ("    " if is_last else "│   ")
        for i, child in enumerate(span.children):
            cls._render_span(child, lines, child_prefix, i == len(span.children) - 1)

    @classmethod
    def _attr_summary(cls, attrs: Dict) -> str:
        """提取关键属性做简短摘要"""
        parts = []
        if "tokens" in attrs:
            parts.append(f"tokens={attrs['tokens']}")
        if "tool" in attrs:
            parts.append(f"tool={attrs['tool']}")
        if "iteration" in attrs:
            parts.append(f"iter={attrs['iteration']}")
        if "status_code" in attrs:
            parts.append(f"status={attrs['status_code']}")
        if "results" in attrs:
            parts.append(f"results={attrs['results']}")
        if "error" in attrs:
            parts.append(f"❌ {attrs['error'][:60]}")
        return ", ".join(parts)


# ============================================================
# Span 上下文管理器
# ============================================================

class SpanContext:
    """Span 的上下文管理器 — 进入时创建子节点，退出时关闭"""

    def __init__(self, name: str, tracer: "Tracer", **attrs):
        self.name = name
        self.tracer = tracer
        self.attrs = attrs
        self.span: Optional[Span] = None
        self._token = None

    def __enter__(self) -> Span:
        parent = _current_span.get()
        self.span = Span(
            name=self.name,
            parent_id=parent.span_id if parent else "",
            span_id=self.tracer._next_id(),
            start_time=time.time(),
            attrs=self.attrs,
        )

        if parent:
            parent.children.append(self.span)

        self._token = _current_span.set(self.span)

        # 第一个 span = root
        trace = _current_trace.get()
        if trace is not None and trace.root_span is None:
            trace.root_span = self.span

        return self.span

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.span:
            self.span.end_time = time.time()
            # GeneratorExit 是流式生成器被客户端主动关闭，不算错误
            if exc_type is not None and exc_type is not GeneratorExit:
                self.span.status = "error"
                self.span.set_attr("error", str(exc_val)[:200])

        if self._token is not None:
            _current_span.reset(self._token)

        return False  # 不吞异常

    async def __aenter__(self) -> Span:
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return self.__exit__(exc_type, exc_val, exc_tb)


# ============================================================
# Tracer — 主入口
# ============================================================

class Tracer:
    """轻量级 Tracer — 创建和管理 Trace"""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.store = TraceStore(max_traces=200)
        self._id_counter = 0
        self.console_renderer = ConsoleSpanRenderer()
        self._exporter: Optional["TraceExporter"] = None

    def _next_id(self) -> str:
        self._id_counter += 1
        return f"sp_{self._id_counter:04d}"

    def _next_trace_id(self) -> str:
        return f"trace_{int(time.time() * 1000):x}"

    def set_exporter(self, exporter: "TraceExporter"):
        """设置导出器（JSON 文件等）"""
        self._exporter = exporter

    def start_trace(self, session_id: str = "", user_query: str = "") -> "TraceContextManager":
        """开始一次新的 trace（应该在 HTTP 请求入口调用）"""
        return TraceContextManager(self, session_id, user_query)

    def span(self, name: str, **attrs) -> SpanContext:
        """在当前 trace 中创建一个 span"""
        if not self.enabled:
            return _NoopSpanContext()
        return SpanContext(name, self, **attrs)

    def finish_trace(self, trace: Trace):
        """完成并存储一条 trace"""
        trace.end_time = time.time()
        self.store.add(trace)

        # 控制台输出
        print(self.console_renderer.render(trace))

        # 导出
        if self._exporter:
            try:
                self._exporter.export(trace)
            except Exception as e:
                # 导出失败记录日志但不影响主流程
                import logging
                logging.getLogger("tracer").warning(f"Trace export failed: {e}")


class TraceContextManager:
    """Trace 的上下文管理器"""

    def __init__(self, tracer: Tracer, session_id: str, user_query: str):
        self.tracer = tracer
        self.session_id = session_id
        self.user_query = user_query
        self.trace: Optional[Trace] = None
        self._token = None

    def __enter__(self) -> "TraceContextManager":
        if not self.tracer.enabled:
            return self

        self.trace = Trace(
            trace_id=self.tracer._next_trace_id(),
            root_span=None,
            start_time=time.time(),
            session_id=self.session_id,
            user_query=self.user_query,
        )
        self._token = _current_trace.set(self.trace)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.trace and self._token is not None:
            if self.trace.root_span is None:
                self.trace.root_span = Span(
                    name="empty_trace",
                    span_id=self.tracer._next_id(),
                    start_time=self.trace.start_time,
                    end_time=time.time(),
                )
            self.tracer.finish_trace(self.trace)
            _current_trace.reset(self._token)
        return False

    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return self.__exit__(exc_type, exc_val, exc_tb)


class _NoopSpanContext:
    """禁用 tracing 时的空 Span"""

    def set_attr(self, key, value):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


# ============================================================
# 装饰器
# ============================================================

def trace(name: str = None, **static_attrs):
    """
    装饰器：自动包装 async 函数为 Span

    @trace("llm.chat")
    async def chat(self, messages):
        ...

    @trace("tool.execute", tool="search_kb")
    async def execute(self, **kwargs):
        ...
    """
    def decorator(func: Callable):
        span_name = name or func.__name__

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracer = get_tracer()
            if not tracer.enabled:
                return await func(*args, **kwargs)

            with tracer.span(span_name, **static_attrs) as span:
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    span.status = "error"
                    span.set_attr("error", str(e)[:200])
                    raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            tracer = get_tracer()
            if not tracer.enabled:
                return func(*args, **kwargs)

            with tracer.span(span_name, **static_attrs) as span:
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    span.status = "error"
                    span.set_attr("error", str(e)[:200])
                    raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# ============================================================
# 导出器
# ============================================================

class TraceExporter:
    """将 trace 持久化到 JSON 文件"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(self, trace: Trace):
        filename = f"trace-{trace.trace_id}.json"
        filepath = self.output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(trace.to_dict(), f, ensure_ascii=False, indent=2)


# ============================================================
# 全局单例
# ============================================================

_tracer_instance: Optional[Tracer] = None


def get_tracer() -> Tracer:
    global _tracer_instance
    if _tracer_instance is None:
        _tracer_instance = Tracer(enabled=True)
    return _tracer_instance


def init_tracer(enabled: bool = True, trace_dir: str = None) -> Tracer:
    global _tracer_instance
    _tracer_instance = Tracer(enabled=enabled)
    if trace_dir:
        _tracer_instance.set_exporter(TraceExporter(trace_dir))
    return _tracer_instance
