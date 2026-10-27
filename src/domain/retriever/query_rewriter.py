"""
查询改写器 — 优化用户查询以提升检索精度

改写下策略：
1. 口语化 → 标准化（如"手机上搞那个气费" → "手机缴纳燃气费"）
2. 补齐上下文信息
3. 关键词扩展（同义词）

当前实现为规则+模板版本，后续可升级为 LLM 驱动
"""

import re
from typing import Optional

from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


class QueryRewriter:
    """查询改写器 — 优化检索查询词"""

    # 口语化 → 标准化映射
    NORMALIZE_MAP = {
        "搞": "办理",
        "弄": "办理",
        "搞一下": "办理",
        "咋": "怎么",
        "啥": "什么",
        "哪儿": "哪里",
        "咋整": "怎么办",
        "整一下": "办理",
        "多少银子": "多少钱",
        "几多钱": "多少钱",
        "么子": "什么",
        "何解": "怎么办",
    }

    # 燃气领域同义词扩展
    SYNONYM_EXPAND = {
        "缴费": ["交费", "充值", "付钱", "付款", "缴纳", "交钱", "续费"],
        "报装": ["新装", "开户", "安装", "通气", "接气"],
        "过户": ["转让", "转户", "更名", "改名", "换名字"],
        "报修": ["维修", "修理", "坏了", "出问题", "故障", "不好使"],
        "停气": ["断气", "没气", "不来气", "打不着"],
        "安检": ["检查", "检测", "安全检查", "入户检查"],
        "账单": ["气费单", "缴费单", "发票", "收据", "清单"],
        "投诉": ["举报", "意见", "反映", "不满", "吐槽"],
        "漏气": ["泄漏", "泄露", "跑气", "冒气"],
    }

    # 疑问词消除（保留核心查询意图）
    QUESTION_PREFIXES = [
        r"^(请问|我想|我要|帮我|麻烦|能不能|可以|怎么样|如何)",
        r"(吗|呢|啊|呀|吧|哦)$",
    ]

    def rewrite(self, query: str, context: Optional[dict] = None) -> str:
        """
        改写用户查询为更精准的检索词

        Args:
            query: 原始用户查询
            context: 上下文信息 {'last_intent': ..., 'history': [...]}

        Returns:
            改写后的查询词
        """
        original = query
        rewritten = query.strip()

        # 1. 去除疑问前缀和后缀
        for prefix_pattern in self.QUESTION_PREFIXES:
            rewritten = re.sub(prefix_pattern, "", rewritten)

        # 2. 口语词标准化
        for slang, formal in self.NORMALIZE_MAP.items():
            rewritten = rewritten.replace(slang, formal)

        # 3. 关键词数量不足时扩展同义词
        if len(rewritten) < 10:
            for keyword, synonyms in self.SYNONYM_EXPAND.items():
                if keyword in rewritten:
                    # 添加同义词以扩大检索范围
                    expanded = " ".join(synonyms[:2])
                    rewritten = f"{rewritten} {expanded}"
                    break

        # 4. 上下文增强（如果有上一轮意图）
        if context and context.get("last_intent"):
            intent_name = context["last_intent"].get("intent_name", "")
            sub_intent = context["last_intent"].get("sub_intent", "")
            if intent_name and len(rewritten) < 15:
                rewritten = f"{intent_name} {sub_intent} {rewritten}"

        # 去除多余空白
        rewritten = re.sub(r"\s+", " ", rewritten).strip()

        if rewritten != original:
            logger.debug(f"Query rewritten: '{original[:50]}' → '{rewritten[:50]}'")

        return rewritten or original  # 保证不返回空字符串

    def rewrite_for_keywords(self, query: str) -> list:
        """
        从查询中提取关键搜索词（用于 BM25）

        Returns:
            关键词列表
        """
        # 简单的 jieba 分词 + 停用词过滤
        stopwords = {"的", "了", "是", "我", "你", "他", "她", "在", "有", "和",
                    "吗", "呢", "啊", "吧", "怎么", "什么", "如何", "哪", "这", "那",
                    "一个", "一下", "一些", "这个", "那个", "可以", "需要",
                    "请问", "我想", "我要", "帮我", "麻烦", "能不能"}

        try:
            import jieba
            words = jieba.lcut(query)
        except Exception:
            words = list(query)

        keywords = [w for w in words if w.strip() and w not in stopwords and len(w) > 1]
        return keywords


# 单例
_rewriter_instance: Optional[QueryRewriter] = None


def get_query_rewriter() -> QueryRewriter:
    global _rewriter_instance
    if _rewriter_instance is None:
        _rewriter_instance = QueryRewriter()
    return _rewriter_instance
