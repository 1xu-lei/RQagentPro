"""
请求日志中间件 — 仅记录请求日志

Trace 的创建已下沉到具体路由（chat/chat_stream），
避免 SSE 流式响应在中间件返回后才真正执行、导致 trace 上下文丢失的问题。
"""

import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


class RequestLogMiddleware(BaseHTTPMiddleware):
    """记录每个 HTTP 请求的耗时与状态码"""

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        try:
            response = await call_next(request)
        finally:
            elapsed_ms = int((time.time() - start_time) * 1000)

        # 跳过健康检查等噪声日志
        skip_paths = ("/api/health", "/api/stats", "/api/traces", "/docs", "/openapi.json", "/api/debug")
        if request.url.path in skip_paths:
            return response

        logger.info(
            f"{request.method} {request.url.path} → {response.status_code} ({elapsed_ms}ms)",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "elapsed_ms": elapsed_ms,
            },
        )

        return response
