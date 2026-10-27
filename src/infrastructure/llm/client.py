"""
DeepSeek LLM 异步客户端

特性：
- 基于 httpx 的异步 HTTP 请求
- 自动重试（指数退避）
- 超时控制
- 支持 Function Calling
- 支持流式（SSE）输出
"""

import json
import time
import asyncio
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from src.infrastructure.llm.types import LLMResponse, StreamChunk, ToolCall
from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


class LLMClient:
    """DeepSeek API 异步客户端"""

    def __init__(
        self,
        api_key: str = "",
        api_url: str = "https://api.deepseek.com/chat/completions",
        model: str = "deepseek-v4-pro",
        temperature: float = 0.3,
        max_tokens: int = 800,
        timeout: float = 30.0,
        max_retries: int = 2,
    ):
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries

        # 内部 httpx 客户端（懒初始化）
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                limits=httpx.Limits(max_keepalive_connections=5),
            )
        return self._client

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    # ---- 核心调用 ----

    async def chat(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> LLMResponse:
        """
        发送聊天请求，返回完整响应

        Args:
            messages: 消息列表 [{"role": "...", "content": "..."}]
            tools: OpenAI Function Calling 工具定义列表
            tool_choice: "auto" | "none" | "required"
            temperature: 覆盖默认温度
            max_tokens: 覆盖默认最大 token
            stream: 是否流式（流式模式下内部收集完整响应）

        Returns:
            LLMResponse 对象
        """
        from src.infrastructure.tracing import get_tracer

        payload = self._build_payload(messages, tools, tool_choice, temperature, max_tokens, stream=False)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        with get_tracer().span("llm.chat", model=self.model) as span:
            for attempt in range(self.max_retries + 1):
                try:
                    client = await self._get_client()
                    resp = await client.post(self.api_url, headers=headers, json=payload)

                    if resp.status_code == 200:
                        data = resp.json()
                        result = self._parse_response(data)
                        # 记录 token 用量
                        usage = data.get("usage", {})
                        span.set_attr("tokens_in", usage.get("prompt_tokens", 0))
                        span.set_attr("tokens_out", usage.get("completion_tokens", 0))
                        span.set_attr("tokens_total", usage.get("total_tokens", 0))
                        span.set_attr("finish_reason", result.finish_reason)
                        if result.tool_calls:
                            span.set_attr("tool_calls_requested", ",".join(tc.name for tc in result.tool_calls))
                        span.set_attr("attempts", attempt + 1)
                        return result

                    # 可重试的错误
                    if resp.status_code in (429, 500, 502, 503):
                        if attempt < self.max_retries:
                            wait = 2 ** attempt
                            logger.warning(
                                f"LLM API {resp.status_code}, retrying in {wait}s (attempt {attempt + 1}/{self.max_retries})"
                            )
                            await asyncio.sleep(wait)
                            continue

                    raise LLMError(
                        f"API error {resp.status_code}: {resp.text[:300]}",
                        status_code=resp.status_code,
                    )

                except httpx.TimeoutException:
                    if attempt < self.max_retries:
                        logger.warning(f"LLM API timeout, retrying (attempt {attempt + 1}/{self.max_retries})")
                        await asyncio.sleep(2 ** attempt)
                        continue
                    span.set_attr("error", "timeout after retries")
                    raise LLMError("API request timeout", status_code=408)

                except httpx.RequestError as e:
                    if attempt < self.max_retries:
                        logger.warning(f"LLM request error: {e}, retrying")
                        await asyncio.sleep(2 ** attempt)
                        continue
                    span.set_attr("error", str(e)[:200])
                    raise LLMError(f"Request error: {e}")

            raise LLMError("Max retries exceeded")

    async def chat_stream(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        流式聊天请求，逐块产出 StreamChunk

        Yields:
            StreamChunk 对象（逐 token）
        """
        from src.infrastructure.tracing import get_tracer

        payload = self._build_payload(messages, tools, tool_choice, temperature, max_tokens, stream=True)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        client = await self._get_client()

        async with get_tracer().span("llm.chat", model=self.model, mode="stream") as span:
            async with client.stream("POST", self.api_url, headers=headers, json=payload) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    raise LLMError(f"Stream error {resp.status_code}: {body[:300]}", status_code=resp.status_code)

                # 解析 SSE 流
                tool_call_buffer: Dict[int, Dict] = {}  # index → accumulated tool_call

                async for line in resp.aiter_lines():
                    if not line or line.startswith(":"):
                        continue
                    if line == "data: [DONE]":
                        break
                    if not line.startswith("data: "):
                        continue

                    try:
                        data = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue

                    choice = data.get("choices", [{}])[0]
                    delta = choice.get("delta", {})
                    finish_reason = choice.get("finish_reason")

                    chunk = StreamChunk()

                    # 文本内容
                    if "content" in delta and delta["content"]:
                        chunk.content = delta["content"]

                    # DeepSeek v4 思考内容（需原样传回）
                    if "reasoning_content" in delta and delta["reasoning_content"]:
                        chunk.reasoning_content = delta["reasoning_content"]

                    # 工具调用（流式累积）
                    if "tool_calls" in delta:
                        for tc_delta in delta["tool_calls"]:
                            idx = tc_delta.get("index", 0)
                            if idx not in tool_call_buffer:
                                tool_call_buffer[idx] = {
                                    "id": tc_delta.get("id", ""),
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                }
                            buf = tool_call_buffer[idx]
                            func_delta = tc_delta.get("function", {})
                            if "name" in func_delta:
                                buf["function"]["name"] += func_delta["name"]
                            if "arguments" in func_delta:
                                buf["function"]["arguments"] += func_delta["arguments"]

                    chunk.finish_reason = finish_reason
                    if finish_reason == "tool_calls":
                        chunk.tool_calls = [
                            tc for idx, tc in sorted(tool_call_buffer.items())
                        ]

                    yield chunk

                    if finish_reason:
                        span.set_attr("finish_reason", finish_reason)
                        if finish_reason == "tool_calls":
                            span.set_attr(
                                "tool_calls_requested",
                                ",".join(tc["function"]["name"] for tc in tool_call_buffer.values()),
                            )
                        break

    # ---- 内部方法 ----

    def _build_payload(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]],
        tool_choice: str,
        temperature: Optional[float],
        max_tokens: Optional[int],
        stream: bool,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "stream": stream,
        }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

        return payload

    def _parse_response(self, data: Dict) -> LLMResponse:
        """解析非流式 API 响应"""
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        finish = choice.get("finish_reason", "stop")

        tool_calls = []
        raw_tool_calls = message.get("tool_calls", [])
        for tc in raw_tool_calls:
            func = tc.get("function", {})
            try:
                args = json.loads(func.get("arguments", "{}"))
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(
                ToolCall(id=tc.get("id", ""), name=func.get("name", ""), arguments=args)
            )

        return LLMResponse(
            content=message.get("content", "") or "",
            finish_reason=finish,
            tool_calls=tool_calls,
            usage=data.get("usage", {}),
            model=data.get("model", ""),
            raw=data,
        )


class LLMError(Exception):
    """LLM 调用异常"""

    def __init__(self, message: str, status_code: int = 0):
        super().__init__(message)
        self.status_code = status_code


# ============================================================
# 工厂函数
# ============================================================

def create_llm_client_from_config() -> LLMClient:
    """从全局配置创建 LLMClient"""
    from src.infrastructure.config import get_config

    cfg = get_config()
    return LLMClient(
        api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        api_url=cfg.get("llm.api_url", "https://api.deepseek.com/chat/completions"),
        model=cfg.get("llm.model", "deepseek-v4-pro"),
        temperature=cfg.get("llm.temperature", 0.3),
        max_tokens=cfg.get("llm.max_tokens", 800),
        timeout=cfg.get("llm.timeout", 30),
        max_retries=cfg.get("llm.max_retries", 2),
    )


import os  # noqa: E402 (import at bottom for factory function)
