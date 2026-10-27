"""
对话相关 Pydantic 数据模型
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """对话请求"""
    query: str = Field(..., description="用户问题", min_length=1, max_length=2000)
    session_id: Optional[str] = Field(None, description="会话ID，不传则创建新会话")


class ChatResponse(BaseModel):
    """对话响应"""
    session_id: str
    answer: str
    intent_code: str = ""
    intent_name: str = ""
    sub_intent: str = ""
    confidence: float = 0.0
    regulation_ref: str = ""
    source: str = ""
    is_safety: bool = False
    transfer_to_human: bool = False
    transfer_reason: str = ""
    tool_calls_count: int = 0
    tool_calls_detail: list = []
    agent_iterations: int = 0


class IntentRequest(BaseModel):
    """意图检测请求"""
    query: str = Field(..., min_length=1, max_length=500)


class IntentResponse(BaseModel):
    """意图检测响应"""
    intent_code: str = ""
    intent_name: str = ""
    sub_intent: str = ""
    confidence: float = 0.0
    is_safety: bool = False
    safety_keywords: List[str] = []
    source: str = ""


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = "ok"
    version: str = "3.0.0"
    architecture: str = "Interface → Application → Domain → Infrastructure"
    engines_ready: bool = False
    kb_entries: int = 0
    llm_connected: bool = False


class StatsResponse(BaseModel):
    """系统统计响应"""
    active_sessions: int = 0
    knowledge_base_size: int = 0
    graph_nodes: int = 0
