"""
统一异常处理中间件
"""

import traceback
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """全局异常处理"""

    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            logger.error(
                f"Unhandled error: {e}",
                extra={
                    "path": str(request.url.path),
                    "method": request.method,
                    "error_type": type(e).__name__,
                },
            )
            traceback.print_exc()

            return JSONResponse(
                status_code=500,
                content={
                    "error": "internal_error",
                    "message": "服务器内部错误，请稍后重试",
                    "detail": str(e)[:200] if logger.level <= 10 else "",
                },
            )
