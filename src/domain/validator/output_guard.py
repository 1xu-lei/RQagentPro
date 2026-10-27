"""
输出校验器 — 对 Agent 生成的回答进行质量检查

校验维度：
1. 安全信息完整性：回答是否包含安全指令的所有要素
2. 事实一致性：回答是否与检索结果矛盾
3. 法规引用准确性：引用的法规编码和名称是否匹配
4. 禁语检测：是否包含不应出现的表述
5. 长度合理性：是否过短（信息不足）或过长（冗余）

用法：
    guard = OutputGuard()
    issues = guard.validate(answer, context={"search_results": [...], "safety": False})
"""

import re
from typing import Any, Dict, List, Optional

from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


class OutputGuard:
    """Agent 输出质量校验器"""

    # 禁止出现在回答中的表述
    FORBIDDEN_PATTERNS = [
        (r"我(不|没)知道", "禁止使用'我不知道'，应引导用户转人工"),
        (r"我(猜|觉得|认为|估计)", "禁止使用主观猜测词汇"),
        (r"根据我的(知识|理解|训练)", "禁止引用自身知识来源"),
        (r"(可能|大概|也许|应该)是", "避免模糊表述，应给出确定信息"),
    ]

    # 安全模板必须包含的关键词
    SAFETY_TEMPLATE_REQUIRED = [
        ("关闭.*阀", "安全回答缺少'关闭阀门'指令"),
        ("通风", "安全回答缺少'通风'指令"),
        ("95777", "安全回答缺少抢修电话 95777"),
    ]

    # 回答质量阈值
    MIN_ANSWER_LENGTH = 10   # 最小回答长度（字符）
    MAX_ANSWER_LENGTH = 2000 # 最大回答长度（字符）
    MIN_SAFETY_LENGTH = 50   # 安全回答最小长度

    def validate(
        self,
        answer: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        校验回答质量

        Args:
            answer: Agent 生成的回答
            context: 上下文信息 {'search_results': [...], 'is_safety': bool, ...}

        Returns:
            {'valid': bool, 'issues': [...], 'warnings': [...]}
        """
        context = context or {}
        issues = []
        warnings = []

        # 1. 基础长度检查
        if len(answer) < self.MIN_ANSWER_LENGTH:
            issues.append(f"回答过短（{len(answer)}字），可能信息不完整")

        if len(answer) > self.MAX_ANSWER_LENGTH:
            warnings.append(f"回答过长（{len(answer)}字），建议精简")

        # 2. 安全场景特殊检查
        if context.get("is_safety"):
            if len(answer) < self.MIN_SAFETY_LENGTH:
                issues.append(f"安全回答过短（{len(answer)}字），必须包含完整的安全指令")

            for pattern, msg in self.SAFETY_TEMPLATE_REQUIRED:
                if not re.search(pattern, answer):
                    issues.append(msg)

        # 3. 禁语检测
        for pattern, msg in self.FORBIDDEN_PATTERNS:
            if re.search(pattern, answer):
                issues.append(msg)

        # 4. 事实一致性（如果有检索结果）
        search_results = context.get("search_results", [])
        if search_results:
            # 展平嵌套列表 [[r1,r2], [r3]] → [r1,r2,r3]
            flat_results = []
            for item in search_results:
                if isinstance(item, list):
                    flat_results.extend(item)
                elif isinstance(item, dict):
                    flat_results.append(item)

            # 检查回答是否引用了不存在的法规编码
            mentioned_regs = set(re.findall(r"FG-[A-Z]{2}-\d{3}", answer))
            actual_regs = set()
            for r in flat_results:
                if isinstance(r, dict):
                    reg_code = r.get("regulation_code", "")
                    if reg_code:
                        actual_regs.add(reg_code)

            unknown_regs = mentioned_regs - actual_regs
            if unknown_regs:
                warnings.append(f"引用了未在检索结果中的法规编码: {', '.join(unknown_regs)}")

        # 5. 联系方式完整性检查
        contact_checks = [
            ("转人工" in answer or "客服" in answer, "提到转人工时建议附上客服电话"),
        ]
        for condition, msg in contact_checks:
            if condition:
                if "0734-8192944" not in answer and "95777" not in answer:
                    warnings.append(msg)

        valid = len(issues) == 0

        if issues:
            logger.warning(f"Output validation found {len(issues)} issue(s): {issues}")
        if warnings:
            logger.info(f"Output validation found {len(warnings)} warning(s)")

        return {
            "valid": valid,
            "issues": issues,
            "warnings": warnings,
        }

    def is_critical(self, issue: str) -> bool:
        """判断是否为严重问题（需要阻断输出）"""
        critical_patterns = [
            "安全回答缺少",
            "禁止",
        ]
        return any(p in issue for p in critical_patterns)


# 单例
_guard_instance: Optional[OutputGuard] = None


def get_output_guard() -> OutputGuard:
    global _guard_instance
    if _guard_instance is None:
        _guard_instance = OutputGuard()
    return _guard_instance
