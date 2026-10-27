"""
LLM 相关类型定义
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolCall:
    """工具调用请求"""
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class Message:
    """对话消息"""
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None


@dataclass
class LLMResponse:
    """LLM 响应"""
    content: str = ""
    finish_reason: str = ""  # "stop" | "tool_calls" | "length"
    tool_calls: List[ToolCall] = field(default_factory=list)
    usage: Dict[str, int] = field(default_factory=dict)  # prompt_tokens, completion_tokens
    model: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamChunk:
    """流式响应片段"""
    content: str = ""
    finish_reason: Optional[str] = None
    tool_calls: List[Dict] = field(default_factory=list)
    reasoning_content: str = ""  # DeepSeek v4 思考模式需原样传回
