"""
Agent 引擎 — 异步 ReAct (Reasoning + Acting) 循环

核心职责：
1. 接收用户消息 + 对话历史
2. 构建消息上下文（系统提示词 + 历史 + 当前问题）
3. 进入 ReAct 循环：LLM 推理 → 工具调用 → 观察 → 再推理
4. 返回最终回答 + 工具调用轨迹

特性：
- 全异步（asyncio + httpx）
- 支持流式（SSE）输出
- 可配置最大迭代次数
- 结构化工具调用日志
"""

import json
from typing import Any, AsyncIterator, Dict, List, Optional

from src.infrastructure.config import get_config
from src.infrastructure.logging import get_logger
from src.infrastructure.llm import LLMClient, LLMError, LLMResponse, StreamChunk, ToolCall
from src.domain.tools.registry import get_tool_registry
from src.domain.agent.prompt_manager import get_prompt_manager

logger = get_logger(__name__)


class AgentResult:
    """Agent 运行结果"""

    def __init__(
        self,
        answer: str = "",
        tool_calls_log: List[Dict] = None,
        iterations: int = 0,
        source: str = "agent",
        context_inherited: bool = False,
        regulation_ref: str = "",
        intent_info: Optional[Dict] = None,
    ):
        self.answer = answer
        self.tool_calls_log = tool_calls_log or []
        self.iterations = iterations
        self.source = source  # "agent" | "fallback" | "max_iterations" | "error"
        self.context_inherited = context_inherited
        self.regulation_ref = regulation_ref
        self.intent_info = intent_info or {}


class ReActAgent:
    """ReAct Agent — 异步实现"""

    def __init__(self, llm_client: LLMClient = None):
        self.max_iterations = get_config().get("agent.max_iterations", 5)
        self.history_turns = get_config().get("agent.conversation_history_turns", 3)

        # LLM 客户端
        if llm_client is None:
            import os
            cfg = get_config()
            self.llm = LLMClient(
                api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
                api_url=cfg.get("llm.api_url"),
                model=cfg.get("llm.model"),
                temperature=cfg.get("llm.temperature", 0.3),
                max_tokens=cfg.get("llm.max_tokens", 800),
                timeout=cfg.get("llm.timeout", 30),
                max_retries=cfg.get("llm.max_retries", 2),
            )
        else:
            self.llm = llm_client

        self.tool_registry = get_tool_registry()
        self.prompt_manager = get_prompt_manager()

    # ---- 主入口 ----

    async def run(
        self,
        user_query: str,
        history: List[Dict] = None,
        session_id: str = None,
        last_intent: Dict = None,
    ) -> AgentResult:
        """
        运行 Agent 处理用户问题（非流式）

        Args:
            user_query: 用户当前问题
            history: 对话历史 [{'role':'user'/'assistant', 'content':'...'}]
            session_id: 会话 ID（用于日志）
            last_intent: 上一轮意图分类结果

        Returns:
            AgentResult 对象
        """
        from src.infrastructure.tracing import get_tracer

        context_inherited = False

        # ---- 上下文意图继承检测 ----
        followup_patterns = [
            "多少钱", "收费", "价格", "费用", "地址", "在哪", "电话",
            "材料", "流程", "怎么办", "需要什么", "然后", "接着",
            "之后", "第二种", "别的", "其他", "还有", "步骤", "微信",
            "公众号", "网上", "线上", "手机", "怎么交", "怎么查",
        ]
        meta_patterns = [
            "上一个", "刚才", "之前", "前面", "刚刚", "上次", "前面那个",
            "上上个", "之前的", "刚说的", "刚才的", "前一个",
            "第一个", "第二个", "第三个", "第四个", "第五个",
            "第1个", "第2个", "第3个", "第4个", "第5个",
            "重新回答", "再回答", "回答第",
        ]
        is_meta_question = any(p in user_query for p in meta_patterns)

        if last_intent and len(user_query) <= 10 and not is_meta_question:
            if any(p in user_query for p in followup_patterns):
                context_inherited = True

        # ---- 构建消息 ----
        messages = self._build_messages(
            user_query, history, last_intent, context_inherited, is_meta_question
        )

        # ---- ReAct 循环 ----
        tool_calls_log = []

        with get_tracer().span("agent.run", session_id=session_id or "") as agent_span:
            for iteration in range(1, self.max_iterations + 1):
                with get_tracer().span("react.iteration", iteration=iteration) as iter_span:
                    try:
                        response = await self.llm.chat(
                            messages=messages,
                            tools=self.tool_registry.to_openai_schemas(),
                            tool_choice="auto",
                        )
                    except LLMError as e:
                        logger.error(f"LLM call failed at iteration {iteration}: {e}")
                        iter_span.set_attr("error", str(e)[:200])
                        agent_span.set_attr("result_source", "fallback")
                        agent_span.set_attr("total_iterations", iteration)
                        return AgentResult(
                            answer=f"系统繁忙，请稍后重试或拨打人工客服 0734-8192944。",
                            tool_calls_log=tool_calls_log,
                            iterations=iteration,
                            source="fallback",
                        )

                    # 情况1：LLM 要求调用工具
                    if response.finish_reason == "tool_calls" or response.tool_calls:
                        # 防御：finish_reason 为 tool_calls 但 tool_calls 为空（API 幻觉），跳过工具调用
                        if not response.tool_calls:
                            logger.warning(f"Iter {iteration}: finish_reason=tool_calls but tool_calls empty, skipping")
                            continue

                        # 将 assistant 消息追加到对话
                        assistant_msg = {
                            "role": "assistant",
                            "content": response.content,
                            "tool_calls": [
                                {
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.name,
                                        "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                                    },
                                }
                                for tc in response.tool_calls
                            ],
                        }
                        messages.append(assistant_msg)

                        for tc in response.tool_calls:
                            logger.info(f"Iter {iteration}: calling {tc.name}({str(tc.arguments)[:100]})")

                            # 执行工具
                            result_str = await self.tool_registry.execute(tc.name, tc.arguments)

                            # 解析结果
                            result_parsed = None
                            try:
                                result_parsed = json.loads(result_str)
                            except Exception:
                                pass

                            tool_calls_log.append({
                                "iteration": iteration,
                                "tool": tc.name,
                                "arguments": tc.arguments,
                                "result_preview": result_str[:150] + ("..." if len(result_str) > 150 else ""),
                                "result_parsed": result_parsed,
                            })

                            # 将工具结果追加到对话
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": result_str,
                            })

                        continue  # 继续循环

                    # 情况2：LLM 返回最终回答
                    if response.content:
                        logger.info(f"Final answer at iteration {iteration} ({len(response.content)} chars)")

                        # 提取意图和法规信息
                        intent_info = self._extract_intent_from_log(tool_calls_log)
                        regulation_ref = self._extract_regulation_from_log(tool_calls_log)

                        agent_span.set_attr("result_source", "agent")
                        agent_span.set_attr("total_iterations", iteration)
                        agent_span.set_attr("answer_len", len(response.content))
                        agent_span.set_attr("tools_called", len(tool_calls_log))

                        return AgentResult(
                            answer=response.content,
                            tool_calls_log=tool_calls_log,
                            iterations=iteration,
                            source="agent",
                            context_inherited=context_inherited,
                            regulation_ref=regulation_ref,
                            intent_info=intent_info,
                        )

                    # 情况3：异常空响应
                    logger.warning(f"Empty response at iteration {iteration}")
                    agent_span.set_attr("result_source", "fallback")
                    agent_span.set_attr("total_iterations", iteration)
                    return AgentResult(
                        answer="抱歉，我暂时无法处理您的问题。请拨打人工客服 0734-8192944 咨询。",
                        tool_calls_log=tool_calls_log,
                        iterations=iteration,
                        source="fallback",
                    )

            # 超出最大迭代次数
            logger.warning(f"Max iterations ({self.max_iterations}) reached")
            agent_span.set_attr("result_source", "max_iterations")
            agent_span.set_attr("total_iterations", self.max_iterations)
            return AgentResult(
                answer="抱歉，处理您的问题时遇到了复杂情况。为了更准确地帮助您，建议拨打人工客服 0734-8192944 咨询。",
                tool_calls_log=tool_calls_log,
                iterations=self.max_iterations,
                source="max_iterations",
                context_inherited=context_inherited,
            )

    async def run_stream(
        self,
        user_query: str,
        history: List[Dict] = None,
        session_id: str = None,
        last_intent: Dict = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        流式运行 Agent，逐 token 产出结果

        Yields:
            事件字典：
            - {"type": "thinking", "content": "..."}  推理思考过程
            - {"type": "tool_call", "tool": "...", "args": {...}}  工具调用
            - {"type": "tool_result", "tool": "...", "preview": "..."}  工具结果
            - {"type": "token", "content": "..."}  回答 token
            - {"type": "done", "result": AgentResult}  完成信号
            - {"type": "error", "message": "..."}  错误
        """
        from src.infrastructure.tracing import get_tracer

        # 意图继承检测（同 run 方法）
        context_inherited = False
        followup_patterns = [
            "多少钱", "收费", "价格", "费用", "地址", "在哪", "电话",
            "材料", "流程", "怎么办", "需要什么", "然后", "接着",
        ]
        meta_patterns = [
            "上一个", "刚才", "之前", "前面", "第1个", "第2个",
            "重新回答", "再回答", "回答第",
        ]
        is_meta_question = any(p in user_query for p in meta_patterns)

        if last_intent and len(user_query) <= 10 and not is_meta_question:
            if any(p in user_query for p in followup_patterns):
                context_inherited = True

        messages = self._build_messages(
            user_query, history, last_intent, context_inherited, is_meta_question
        )

        tool_calls_log = []

        with get_tracer().span("agent.run", session_id=session_id or "") as agent_span:
            for iteration in range(1, self.max_iterations + 1):
                with get_tracer().span("react.iteration", iteration=iteration) as iter_span:
                    try:
                        # 流式调用 LLM：缓冲判断→工具则执行，内容则即时推送
                        content_buf = []
                        reasoning_buf = []
                        tool_call_buf: Dict[int, Dict] = {}
                        final_finish = None

                        async for chunk in self.llm.chat_stream(
                            messages=messages,
                            tools=self.tool_registry.to_openai_schemas(),
                            tool_choice="auto",
                        ):
                            if chunk.content:
                                content_buf.append(chunk.content)
                                yield {"type": "token", "content": chunk.content}
                            if chunk.reasoning_content:
                                reasoning_buf.append(chunk.reasoning_content)
                            if chunk.finish_reason:
                                final_finish = chunk.finish_reason
                            if chunk.tool_calls:
                                # 最终 chunk 的 tool_calls 已是完整数据，直接覆盖
                                for tc in chunk.tool_calls:
                                    idx = tc.get("index", 0)
                                    tool_call_buf[idx] = {
                                        "id": tc.get("id", ""),
                                        "function": {
                                            "name": tc.get("function", {}).get("name", ""),
                                            "arguments": tc.get("function", {}).get("arguments", ""),
                                        },
                                    }

                        full_content = "".join(content_buf)

                        # 将累积的 tool_call 转成 ToolCall 对象
                        parsed_tool_calls = []
                        for tc_dict in tool_call_buf.values():
                            func = tc_dict["function"]
                            try:
                                args = json.loads(func.get("arguments", "{}"))
                            except json.JSONDecodeError:
                                args = {}
                            parsed_tool_calls.append(
                                ToolCall(id=tc_dict.get("id", ""), name=func.get("name", ""), arguments=args)
                            )

                    except LLMError as e:
                        iter_span.set_attr("error", str(e)[:200])
                        agent_span.set_attr("result_source", "fallback")
                        agent_span.set_attr("total_iterations", iteration)
                        yield {"type": "error", "message": f"LLM error: {e}"}
                        return

                    # 需要调用工具 → 流出的 token 是思考过程，已推送；继续执行工具
                    if final_finish == "tool_calls" or parsed_tool_calls:
                        # 防御：finish_reason 为 tool_calls 但 tool_calls 为空（API 幻觉），跳过工具调用
                        if not parsed_tool_calls:
                            logger.warning(f"Iter {iteration}: finish_reason=tool_calls but tool_calls empty, skipping")
                            continue

                        assistant_msg = {
                            "role": "assistant",
                            "content": full_content or None,
                            "tool_calls": [
                                {"id": tc.id, "type": "function",
                                 "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)}}
                                for tc in parsed_tool_calls
                            ],
                        }
                        reasoning = "".join(reasoning_buf)
                        if reasoning:
                            assistant_msg["reasoning_content"] = reasoning
                        messages.append(assistant_msg)

                        for tc in parsed_tool_calls:
                            yield {"type": "tool_call", "tool": tc.name, "args": tc.arguments, "iteration": iteration}
                            result_str = await self.tool_registry.execute(tc.name, tc.arguments)
                            result_parsed = None
                            try:
                                result_parsed = json.loads(result_str)
                            except Exception:
                                pass
                            tool_calls_log.append({
                                "iteration": iteration, "tool": tc.name, "arguments": tc.arguments,
                                "result_preview": result_str[:150] + ("..." if len(result_str) > 150 else ""),
                                "result_parsed": result_parsed,
                            })
                            yield {"type": "tool_result", "tool": tc.name, "preview": result_str[:200]}
                            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_str})
                        continue

                    # 最终回答 → token 已在上面实时推送
                    if full_content:
                        intent_info = self._extract_intent_from_log(tool_calls_log)
                        regulation_ref = self._extract_regulation_from_log(tool_calls_log)
                        agent_span.set_attr("result_source", "agent")
                        agent_span.set_attr("total_iterations", iteration)
                        agent_span.set_attr("answer_len", len(full_content))
                        agent_span.set_attr("tools_called", len(tool_calls_log))
                        yield {
                            "type": "done",
                            "result": AgentResult(
                                answer=full_content, tool_calls_log=tool_calls_log,
                                iterations=iteration, source="agent",
                                context_inherited=context_inherited,
                                regulation_ref=regulation_ref, intent_info=intent_info,
                            ),
                        }
                        return

                    # 空响应
                    fallback_msg = "抱歉，我暂时无法处理您的问题。请拨打人工客服 0734-8192944 咨询。"
                    for char in fallback_msg:
                        yield {"type": "token", "content": char}
                    agent_span.set_attr("result_source", "fallback")
                    agent_span.set_attr("total_iterations", iteration)
                    yield {
                        "type": "done",
                        "result": AgentResult(
                            answer=fallback_msg,
                            tool_calls_log=tool_calls_log,
                            iterations=iteration,
                            source="fallback",
                        ),
                    }
                    return

            # 超最大迭代
            max_iter_msg = "抱歉，处理您的问题时遇到了复杂情况。建议拨打人工客服 0734-8192944 咨询。"
            for char in max_iter_msg:
                yield {"type": "token", "content": char}
            agent_span.set_attr("result_source", "max_iterations")
            agent_span.set_attr("total_iterations", self.max_iterations)
            yield {
                "type": "done",
                "result": AgentResult(
                    answer=max_iter_msg,
                    tool_calls_log=tool_calls_log,
                    iterations=self.max_iterations,
                    source="max_iterations",
                ),
            }

    # ---- 消息构建 ----

    def _build_messages(
        self,
        user_query: str,
        history: List[Dict],
        last_intent: Dict,
        context_inherited: bool,
        is_meta_question: bool,
    ) -> List[Dict]:
        """构建 LLM 消息列表"""
        # 系统提示词
        system_prompt = self.prompt_manager.build_system_prompt()
        messages = [{"role": "system", "content": system_prompt}]

        # 注入对话历史
        if history:
            recent = history[-(self.history_turns * 2):]  # user+assistant pairs
            for msg in recent:
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": str(msg.get("content", ""))[:500],
                })

        # 当前用户消息
        if context_inherited:
            user_msg = (
                f"{user_query}\n\n"
                f"[系统提示：这是一个对上一轮回答的追问。"
                f"上一轮意图为「{last_intent.get('intent_name', '')}」—「{last_intent.get('sub_intent', '')}」。"
                f"请在检索时使用上轮意图域过滤。]"
            )
            messages.append({"role": "user", "content": user_msg})
        elif is_meta_question:
            messages.append({"role": "user", "content": user_query})
        elif last_intent and len(user_query) >= 4:
            hint = (
                f"{user_query}\n\n"
                f"[上轮对话意图参考：{last_intent.get('intent_name', '')} > {last_intent.get('sub_intent', '')}]"
            )
            messages.append({"role": "user", "content": hint})
        else:
            messages.append({"role": "user", "content": user_query})

        return messages

    # ---- 工具日志解析 ----

    def _extract_intent_from_log(self, tool_calls_log: List[Dict]) -> Dict:
        """从工具调用日志提取意图信息"""
        # 优先找 classify_user_intent
        for call in tool_calls_log:
            if call.get("tool") == "classify_user_intent":
                parsed = call.get("result_parsed")
                if parsed:
                    return {
                        "intent_name": parsed.get("intent_name", ""),
                        "sub_intent": parsed.get("sub_intent", ""),
                        "confidence": parsed.get("confidence", 0.7),
                        "source": "agent_classify",
                    }

        # 其次从知识库检索的 top result 提取
        for call in tool_calls_log:
            if call.get("tool") == "search_knowledge_base":
                parsed = call.get("result_parsed")
                if parsed and parsed.get("results"):
                    top = parsed["results"][0]
                    category = top.get("category", "")
                    if " > " in category:
                        cat1, cat2 = category.split(" > ", 1)
                    else:
                        cat1, cat2 = category, ""

                    vec_score = top.get("vector_score", 0.0)
                    confidence = vec_score if vec_score > 0 else 0.5

                    return {
                        "intent_name": cat1,
                        "sub_intent": cat2,
                        "confidence": round(confidence, 2),
                        "source": "agent_retrieval",
                    }

        return {
            "intent_name": "", "sub_intent": "",
            "confidence": 0.5, "source": "agent",
        }

    def _extract_regulation_from_log(self, tool_calls_log: List[Dict]) -> str:
        """从工具调用日志提取法规引用"""
        # 从 lookup_regulation 提取
        for call in tool_calls_log:
            if call.get("tool") == "lookup_regulation":
                parsed = call.get("result_parsed")
                if parsed:
                    if parsed.get("full_ref"):
                        return parsed["full_ref"]
                    code = parsed.get("code", "")
                    name = parsed.get("name", "")
                    if name:
                        return f"{name}({code})"
                    return code

        # 从检索结果的 regulation_code 提取
        for call in tool_calls_log:
            if call.get("tool") == "search_knowledge_base":
                parsed = call.get("result_parsed")
                if parsed and parsed.get("results"):
                    for r in parsed["results"]:
                        reg_code = r.get("regulation_code", "")
                        if reg_code:
                            return reg_code
        return ""

    async def close(self):
        """关闭 LLM 客户端连接"""
        await self.llm.close()


# ============================================================
# 单例
# ============================================================

_agent_instance: Optional[ReActAgent] = None


def get_agent() -> ReActAgent:
    """获取全局 Agent 单例"""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = ReActAgent()
    return _agent_instance
