"""
对话接口 — 包含 SSE 流式和非流式两种模式
"""

import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.interface.schemas.chat import (
    ChatRequest, ChatResponse, IntentRequest, IntentResponse,
)
from src.application.chat_service import get_chat_service
from src.application.session_service import get_session_service
from src.infrastructure.logging import get_logger
from src.infrastructure.tracing import get_tracer

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1")


# ============================================================
# SSE 格式化工具
# ============================================================

def _format_sse(event: str, data: dict) -> str:
    """将事件格式化为标准 SSE 格式"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ============================================================
# 非流式对话
# ============================================================

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    主对话接口（非流式）

    流程：安全预检 → Agent 处理 → 响应
    """
    session_mgr = get_session_service()
    chat_service = get_chat_service()

    # 会话管理
    session = None
    if req.session_id:
        session = session_mgr.get_session(req.session_id)
        if session is None:
            session_id = session_mgr.create_session()
            session = session_mgr.get_session(session_id)
        else:
            session_id = req.session_id
    else:
        session_id = session_mgr.create_session()
        session = session_mgr.get_session(session_id)

    # 获取对话历史和上一轮意图
    history_for_agent = []
    last_intent = None
    if session:
        hist = session.get_history(last_n=3)
        history_for_agent = [
            {"role": m.get("role", "user"), "content": str(m.get("content", ""))[:500]}
            for m in hist
        ]
        last_intent = session.get_last_intent()

    # Agent 处理（创建 trace 并绑定真实 session_id / user_query）
    tracer = get_tracer()
    async with tracer.start_trace(session_id=session_id, user_query=req.query):
        result = await chat_service.chat(
            query=req.query,
            history=history_for_agent,
            session_id=session_id,
            last_intent=last_intent,
        )

    # 记录会话
    if session:
        session.add_message("user", req.query, intent=result.get("intent_name"))
        session.add_message(
            "assistant",
            result["answer"],
            type="agent_response",
            intent={
                "intent_name": result.get("intent_name"),
                "sub_intent": result.get("sub_intent"),
                "confidence": result.get("confidence"),
            },
            regulation_ref=result.get("regulation_ref"),
            source=result.get("source"),
            tool_calls=result.get("tool_calls_detail", []),
            is_safety=result.get("is_safety"),
            transfer=result.get("transfer_to_human"),
        )
        session_mgr.save_session(session_id)

        if result.get("transfer_to_human"):
            session_mgr.add_uncovered_query(
                req.query,
                {"intent_name": result.get("intent_name")},
            )

    return ChatResponse(
        session_id=session_id,
        answer=result["answer"],
        intent_code=result.get("intent_code", ""),
        intent_name=result.get("intent_name", ""),
        sub_intent=result.get("sub_intent", ""),
        confidence=result.get("confidence", 0.0),
        regulation_ref=result.get("regulation_ref", ""),
        source=result.get("source", ""),
        is_safety=result.get("is_safety", False),
        transfer_to_human=result.get("transfer_to_human", False),
        transfer_reason=result.get("transfer_reason", ""),
        tool_calls_count=result.get("tool_calls_count", 0),
        tool_calls_detail=result.get("tool_calls_detail", []),
        agent_iterations=result.get("agent_iterations", 0),
    )


# ============================================================
# SSE 流式对话
# ============================================================

@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    流式对话接口（SSE）

    事件类型：
    - safety:      安全预检命中 → 直接输出安全模板
    - tool_call:   Agent 调用工具 → 前端展示决策过程
    - tool_result: 工具返回结果
    - token:       回答文本 token → 前端打字效果
    - done:        对话完成 → 含 session_id / intent 等元信息
    - error:       发生错误
    """
    session_mgr = get_session_service()
    chat_service = get_chat_service()

    # 会话管理
    session = None
    if req.session_id:
        session = session_mgr.get_session(req.session_id)
        if session is None:
            session_id = session_mgr.create_session()
            session = session_mgr.get_session(session_id)
        else:
            session_id = req.session_id
    else:
        session_id = session_mgr.create_session()
        session = session_mgr.get_session(session_id)

    # 获取历史
    history = []
    last_intent = None
    if session:
        hist = session.get_history(last_n=3)
        history = [
            {"role": m.get("role", "user"), "content": str(m.get("content", ""))[:500]}
            for m in hist
        ]
        last_intent = session.get_last_intent()

    async def event_generator():
        full_answer = ""
        is_safety = False
        tool_calls_detail = []
        is_transfer = False
        intent_name = ""
        sub_intent = ""

        try:
            # 创建 trace：在流式生成器内部建立上下文，span 才能真正挂到本次请求上
            async with get_tracer().start_trace(session_id=session_id, user_query=req.query):
                async for event in chat_service.chat_stream(
                    query=req.query,
                    history=history,
                    session_id=session_id,
                    last_intent=last_intent,
                ):
                    event_type = event.get("type", "")

                    if event_type == "safety_alert":
                        is_safety = True
                        yield _format_sse("safety", event)

                    elif event_type == "token":
                        full_answer += event.get("content", "")
                        yield _format_sse("token", event)

                    elif event_type == "tool_call":
                        tool_calls_detail.append(event)
                        yield _format_sse("tool_call", event)

                    elif event_type == "tool_result":
                        yield _format_sse("tool_result", event)

                    elif event_type == "done":
                        if event.get("safety"):
                            is_safety = True
                            is_transfer = True

                        result = event.get("result")
                        if result:
                            intent_name = result.intent_info.get("intent_name", "")
                            sub_intent = result.intent_info.get("sub_intent", "")
                            if hasattr(result, "tool_calls_log"):
                                is_transfer = is_transfer or chat_service.has_tool_call(
                                    result, "request_human_transfer"
                                )

                        yield _format_sse("done", {
                            "session_id": session_id,
                            "is_safety": is_safety,
                            "transfer_to_human": is_transfer,
                            "intent_name": intent_name,
                            "sub_intent": sub_intent,
                            "tool_calls_count": len(tool_calls_detail),
                        })

                    elif event_type == "error":
                        yield _format_sse("error", event)
                        return

        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            yield _format_sse("error", {
                "message": f"流式输出异常: {str(e)[:100]}",
            })

        # 记录会话
        if session and full_answer:
            session.add_message("user", req.query)
            session.add_message(
                "assistant", full_answer,
                type="agent_response",
                intent={"intent_name": intent_name, "sub_intent": sub_intent},
                is_safety=is_safety,
                transfer=is_transfer,
                tool_calls=tool_calls_detail,
            )
            session_mgr.save_session(session_id)
            if is_transfer:
                session_mgr.add_uncovered_query(
                    req.query, {"intent_name": intent_name}
                )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================
# 调试接口
# ============================================================

@router.post("/intent", response_model=IntentResponse)
async def detect_intent(req: IntentRequest):
    """意图识别接口 — 调试用"""
    from src.domain.tools.intent_classify import IntentClassifyTool
    import json as jmod

    tool = IntentClassifyTool()
    result_str = await tool.execute(query=req.query)
    result = jmod.loads(result_str)

    # 安全检测
    from src.domain.safety.gateway import get_gateway
    gw = get_gateway()
    matched_kws = gw.check_safety_keywords(req.query)

    # 编码映射 — 基于规范标签体系
    from src.domain.taxonomy import get_domain_code

    intent_name = result.get("intent_name", "")
    intent_code = get_domain_code(intent_name)  # 自动将旧域名映射到规范编码

    return IntentResponse(
        intent_code=intent_code,
        intent_name=result.get("intent_name", ""),
        sub_intent=result.get("sub_intent", ""),
        confidence=result.get("confidence", 0.0),
        is_safety=len(matched_kws) > 0,
        safety_keywords=matched_kws,
        source=result.get("method", "unknown"),
    )
