"""
对话服务 — 编排完整的对话流程

流程：
1. 安全预检（关键词 + 标签双层防御）
2. Agent 自主处理（ReAct 循环）
3. 结果校验与组装
"""

import json
from typing import Any, AsyncIterator, Dict, Optional

from src.infrastructure.logging import get_logger
from src.infrastructure.tracing import get_tracer
from src.domain.safety.gateway import SafetyGateway, get_gateway
from src.domain.agent.engine import ReActAgent, AgentResult, get_agent
from src.domain.tools.init_tools import register_all_tools
from src.domain.taxonomy import get_domain_code, normalize_domain_name

logger = get_logger(__name__)


class ChatService:
    """对话编排服务"""

    def __init__(self):
        self.gateway: SafetyGateway = get_gateway()
        self.agent: Optional[ReActAgent] = None
        self._initialized = False

    def _ensure_initialized(self):
        """确保工具已注册、Agent 已初始化"""
        if self._initialized:
            return
        register_all_tools()
        self.agent = get_agent()
        self._initialized = True

    async def chat(
        self,
        query: str,
        history: list = None,
        session_id: str = None,
        last_intent: dict = None,
    ) -> Dict[str, Any]:
        """
        处理用户消息（非流式）

        Returns:
            {
                "answer": str,
                "session_id": str,
                "is_safety": bool,
                "transfer_to_human": bool,
                "intent": dict,
                "regulation_ref": str,
                "source": str,
                "tool_calls_count": int,
                "tool_calls_detail": list,
                "agent_iterations": int,
            }
        """
        self._ensure_initialized()

        query = query.strip()

        tracer = get_tracer()

        with tracer.span("chat.process") as process_span:
            # ===== Step 1: 安全预检（关键词 + 标签双层防御） =====
            intent_name = last_intent.get("intent_name", "") if last_intent else ""
            sub_intent = last_intent.get("sub_intent", "") if last_intent else ""
            with tracer.span("safety.check") as safety_span:
                is_safety, matched_kws, safety_label = self.gateway.check_safety_combined(
                    query=query,
                    intent_name=intent_name,
                    sub_intent=sub_intent,
                )
                safety_span.set_attr("is_safety", is_safety)
            if is_safety:
                template = self.gateway.get_safety_template(matched_kws)
                answer = self.gateway.format_safety_response(template)

                trigger = matched_kws or [f"label:{safety_label}"]
                logger.info(f"Safety gateway triggered: {trigger}")
                process_span.set_attr("result_source", "safety")

                return {
                    "answer": answer,
                    "session_id": session_id or "",
                    "is_safety": True,
                    "transfer_to_human": True,
                    "transfer_reason": "安全紧急事件，强制转人工/抢修",
                    "intent_code": "AQ",
                    "intent_name": "安全用气",
                    "sub_intent": "疑似漏气处置",
                    "confidence": 1.0,
                    "regulation_ref": "",
                    "source": f"safety_template({'+'.join(trigger)})",
                    "tool_calls_count": 0,
                    "tool_calls_detail": [],
                    "agent_iterations": 0,
                }

            # ===== Step 2: Agent 处理 =====
            try:
                result = await self.agent.run(
                    user_query=query,
                    history=history,
                    session_id=session_id,
                    last_intent=last_intent,
                )
            except Exception as e:
                logger.error(f"Agent run error: {e}")
                process_span.set_attr("result_source", "error")
                return {
                    "answer": f"系统处理异常，请稍后重试或拨打人工客服 0734-8192944。",
                    "session_id": session_id or "",
                    "is_safety": False,
                    "transfer_to_human": True,
                    "transfer_reason": f"系统异常: {str(e)[:100]}",
                    "intent_code": "QT",
                    "intent_name": "",
                    "sub_intent": "",
                    "confidence": 0.0,
                    "regulation_ref": "",
                    "source": "error",
                    "tool_calls_count": 0,
                    "tool_calls_detail": [],
                    "agent_iterations": 0,
                }

            # ===== Step 3: 结果组装 =====
            intent = result.intent_info
            is_transfer = self._is_transfer(result) or self._is_safety(result)
            transfer_reason = self._get_transfer_reason(result)

            # ===== Step 3.5: 输出校验 =====
            from src.domain.validator.output_guard import get_output_guard
            guard = get_output_guard()

            # 展平所有 search_knowledge_base 结果
            search_results_flat = []
            for call in result.tool_calls_log:
                if call.get("tool") == "search_knowledge_base":
                    results = call.get("result_parsed", {}).get("results", [])
                    if isinstance(results, list):
                        search_results_flat.extend(results)

            validation = guard.validate(
                result.answer,
                context={
                    "is_safety": self._is_safety(result),
                    "search_results": search_results_flat,
                },
            )
            if validation.get("issues"):
                logger.warning(
                    f"Output validation issues: {validation['issues']}",
                    extra={"answer_preview": result.answer[:100]},
                )

            process_span.set_attr("result_source", result.source)
            response = {
                "answer": result.answer,
                "session_id": session_id or "",
                "is_safety": self._is_safety(result),
                "transfer_to_human": is_transfer,
                "transfer_reason": transfer_reason,
                "intent_code": self._map_cat1_to_code(intent.get("intent_name", "")),
                "intent_name": intent.get("intent_name", ""),
                "sub_intent": intent.get("sub_intent", ""),
                "confidence": intent.get("confidence", 0.0),
                "regulation_ref": result.regulation_ref,
                "source": result.source,
                "tool_calls_count": len(result.tool_calls_log),
                "tool_calls_detail": result.tool_calls_log,
                "agent_iterations": result.iterations,
            }

            return response

    async def chat_stream(
        self,
        query: str,
        history: list = None,
        session_id: str = None,
        last_intent: dict = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        流式处理用户消息

        Yields:
            事件字典（同 Agent.run_stream 的输出格式）
        """
        self._ensure_initialized()

        query = query.strip()

        tracer = get_tracer()

        with tracer.span("chat.process") as process_span:
            # 安全预检（含标签检测）
            intent_name = last_intent.get("intent_name", "") if last_intent else ""
            sub_intent = last_intent.get("sub_intent", "") if last_intent else ""
            with tracer.span("safety.check") as safety_span:
                is_safety, matched_kws, safety_label = self.gateway.check_safety_combined(
                    query=query,
                    intent_name=intent_name,
                    sub_intent=sub_intent,
                )
                safety_span.set_attr("is_safety", is_safety)
            if is_safety:
                template = self.gateway.get_safety_template(matched_kws)
                answer = self.gateway.format_safety_response(template)
                process_span.set_attr("result_source", "safety")

                yield {
                    "type": "safety_alert",
                    "keywords": matched_kws,
                    "label_code": safety_label,
                    "template": template.get("title", ""),
                }

                for char in answer:
                    yield {"type": "token", "content": char}

                yield {"type": "done", "safety": True, "transfer": True}
                return

            # Agent 流式处理
            async for event in self.agent.run_stream(
                user_query=query,
                history=history,
                session_id=session_id,
                last_intent=last_intent,
            ):
                yield event

    # ---- 公共辅助方法 ----

    @staticmethod
    def has_tool_call(result: AgentResult, tool_name: str) -> bool:
        """检查 Agent 结果中是否调用了某个工具"""
        for call in result.tool_calls_log:
            if call.get("tool") == tool_name:
                return True
        return False

    def is_safety_result(self, result: AgentResult) -> bool:
        """检查结果是否为安全场景"""
        return self.has_tool_call(result, "check_safety") and any(
            call.get("result_parsed", {}).get("is_safety")
            for call in result.tool_calls_log
            if call.get("tool") == "check_safety"
        )

    def is_transfer_result(self, result: AgentResult) -> bool:
        """检查结果是否需要转人工"""
        return self.has_tool_call(result, "request_human_transfer")

    # ---- 私有辅助方法 ----

    def _is_safety(self, result: AgentResult) -> bool:
        for call in result.tool_calls_log:
            if call.get("tool") == "check_safety":
                parsed = call.get("result_parsed")
                if parsed and parsed.get("is_safety"):
                    return True
        return False

    def _is_transfer(self, result: AgentResult) -> bool:
        for call in result.tool_calls_log:
            if call.get("tool") == "request_human_transfer":
                return True
        return False

    def _get_transfer_reason(self, result: AgentResult) -> str:
        for call in result.tool_calls_log:
            if call.get("tool") == "request_human_transfer":
                return call.get("arguments", {}).get("reason", "Agent判定需转人工")
        return ""

    @staticmethod
    def _map_cat1_to_code(cat1: str) -> str:
        """使用规范标签体系编码"""
        return get_domain_code(cat1)


# 单例
_chat_service_instance: Optional[ChatService] = None


def get_chat_service() -> ChatService:
    global _chat_service_instance
    if _chat_service_instance is None:
        _chat_service_instance = ChatService()
    return _chat_service_instance
