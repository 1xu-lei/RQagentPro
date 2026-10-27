"""
安全网关 — 双层防御（关键词 + 标签）

v3.1: 升级为关键词预检 + 标签体系驱动的双层防御
  - 第一层：关键词硬匹配（不变，保持极高召回）
  - 第二层：标签体系安全敏感检测（弥补关键词漏检）

特性：
- 安全关键词检测（从 YAML 配置文件加载）
- 标签体系安全检测（从 taxonomy 模块读取 ★ 标签）
- 否定语境识别（避免误触发）
- 4 类强制安全模板
- 信息查询过滤（电话/地址等不触发安全）
"""

from typing import Dict, List, Optional, Tuple

from src.infrastructure.config import get_config
from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


class SafetyGateway:
    """安全网关：关键词 + 标签 双层防御"""

    def __init__(self):
        self._keywords: List[str] = []
        self._negation_patterns: List[str] = []
        self._info_patterns: List[str] = []
        self._templates: Dict[str, Dict] = {}
        self._loaded = False

    def _ensure_loaded(self):
        """懒加载安全配置"""
        if self._loaded:
            return
        self._loaded = True

        try:
            cfg = get_config()
            prompt = cfg.prompt("safety")

            self._keywords = prompt.get("keywords", [])
            self._negation_patterns = prompt.get("negation_patterns", [])
            self._info_patterns = prompt.get("info_patterns", [])
            self._templates = prompt.get("templates", {})

            # 合并配置中的额外关键词
            extra_kw = cfg.get("safety.extra_keywords", [])
            if extra_kw:
                self._keywords.extend(extra_kw)
            extra_neg = cfg.get("safety.extra_negation_patterns", [])
            if extra_neg:
                self._negation_patterns.extend(extra_neg)

            logger.info(f"SafetyGateway loaded: {len(self._keywords)} keywords, "
                       f"{len(self._negation_patterns)} negation patterns, "
                       f"{len(self._templates)} templates")
        except Exception:
            # 如果配置加载失败，使用硬编码的默认值
            self._load_defaults()

    def _load_defaults(self):
        """默认安全配置（配置加载失败时的 fallback）"""
        self._keywords = [
            "漏气", "煤气泄漏", "燃气泄漏", "天然气泄漏", "闻到煤气味",
            "煤气味", "臭鸡蛋味", "燃气异味", "报警器响", "报警器报警",
            "燃气报警", "着火", "起火", "爆炸", "燃气中毒", "一氧化碳中毒",
            "中毒", "抢修", "紧急", "管破了", "管道破裂",
        ]
        self._negation_patterns = [
            "打不着", "点不着", "不着火", "打不燃", "点不燃",
            "熄火", "打不开", "开不了", "打不着火", "点不着火",
        ]
        self._info_patterns = ["电话", "多少", "是什么", "怎么查", "在哪", "号码", "热线"]

    # ---- 安全检测 ----

    def check_safety_keywords(self, query: str) -> List[str]:
        """检查安全关键词，排除否定语境 + 非紧急语境"""
        self._ensure_loaded()

        # 否定语境（"打不着火" 不是着火）
        has_negation = any(p in query for p in self._negation_patterns)

        # 非紧急语境：购买、维护、预防、保险 等
        non_emergency = [
            "在哪买", "哪里买", "怎么买", "购买", "买一个", "买哪种",
            "多少钱", "价格", "报价",
            "没电", "换电池", "更换电池", "电池没电", "电量不足",
            "保险", "投保", "保费",
            "怎么防止", "如何防止", "怎么预防", "如何预防", "预防措施",
            "怎么恢复", "如何恢复", "恢复供气",
            "怎么换", "如何换", "自己换",
        ]
        is_non_emergency = any(p in query for p in non_emergency)

        # 否定语境关键词（用户明确说"没闻到""没有""不是"等）
        negated_kw = set()
        for kw in self._keywords:
            if kw in query:
                # 检查该关键词是否被否定（如"没闻到煤气味"→抑制"煤气味"）
                for neg in ["没闻到", "没有闻到", "没有", "没味", "不是"]:
                    # 找否定词后面是否紧跟着这个关键词
                    idx = query.find(neg)
                    if idx >= 0 and kw in query[idx:idx + len(neg) + len(kw) + 5]:
                        negated_kw.add(kw)
                        break

        matched = []
        for kw in self._keywords:
            if kw in query:
                # "着火" 在否定语境中跳过（如 "打不着火"）
                if kw == "着火" and has_negation:
                    continue
                # 关键词被用户明确否定（如"没闻到煤气味"→抑制煤气味）
                if kw in negated_kw:
                    continue
                # 购买/维护/预防/保险语境中跳过安全触发
                if is_non_emergency:
                    continue
                matched.append(kw)

        if matched:
            logger.info(f"Safety keywords matched: {matched} in query: {query[:80]}")

        return matched

    # ---- 安全模板 ----

    def get_safety_template(self, matched_keywords: List[str]) -> Dict:
        """根据匹配的关键词返回最佳安全模板"""
        self._ensure_loaded()

        kw_str = " ".join(matched_keywords)

        # 按优先级匹配模板
        template_order = ["poisoning", "fire_explosion", "alarm", "pipe_damage", "gas_leak"]

        for tname in template_order:
            tpl = self._templates.get(tname, {})
            tpl_kw = tpl.get("keywords", [])
            if any(k in kw_str for k in tpl_kw):
                return tpl

        # 默认返回燃气泄漏模板
        return self._templates.get("gas_leak", self._default_gas_leak_template())

    def format_safety_response(self, template: Dict) -> str:
        """格式化安全应答文本"""
        lines = [template.get("title", "⚠️ 燃气安全紧急处置"), ""]

        for step in template.get("steps", []):
            lines.append(step)

        lines.append("")
        lines.append("【重要提醒】")
        for warning in template.get("warnings", []):
            lines.append(warning)

        if template.get("transfer_msg"):
            lines.append("")
            lines.append(f"📞 {template['transfer_msg']}")

        return "\n".join(lines)

    # ---- 标签体系安全检测（第二层防御） ----

    def check_safety_by_intent(
        self, intent_name: str = "", sub_intent: str = ""
    ) -> Tuple[bool, Optional[str]]:
        """
        根据意图分类结果检测是否为安全敏感场景

        弥补关键词匹配的盲区，例如：
        - "报警器一直响怎么办" → 关键词可能漏检，但意图分类命中 BX-02★
        - "闻到管道口有味道" → 关键词可能漏检，但意图分类命中 BX-04★

        Args:
            intent_name: 意图一级域名（规范名或旧名均可）
            sub_intent: 意图二级标签名

        Returns:
            (is_safety, matched_label_code) 如 ("BX-02", True)
        """
        if not intent_name:
            return False, None

        from src.domain.taxonomy import (
            normalize_domain_name, DOMAIN_BY_NAME, SAFETY_LABEL_CODES,
            DOMAINS, SubLabel,
        )

        std_name = normalize_domain_name(intent_name)
        domain = DOMAIN_BY_NAME.get(std_name)

        if domain is None:
            return False, None

        # 遍历该域下所有安全敏感标签，匹配二级标签名
        for sl in domain.sub_labels:
            if sl.safety_sensitive and sub_intent:
                if sub_intent == sl.name or sub_intent in sl.name or sl.name in sub_intent:
                    logger.info(
                        f"Safety label triggered: {sl.code} ({sl.name}) "
                        f"via intent [{std_name} > {sub_intent}]"
                    )
                    return True, sl.code

        return False, None

    def check_safety_combined(
        self,
        query: str,
        intent_name: str = "",
        sub_intent: str = "",
    ) -> Tuple[bool, List[str], Optional[str]]:
        """
        关键词 + 标签 联合安全检测

        Returns:
            (is_safety, matched_keywords, safety_label_code)
        """
        # 第一层：关键词检测（保持高召回）
        matched_kws = self.check_safety_keywords(query)

        # 第二层：标签检测（弥补关键词漏检）
        is_label_safety, label_code = self.check_safety_by_intent(intent_name, sub_intent)

        is_safety = len(matched_kws) > 0 or is_label_safety

        if is_safety:
            trigger = []
            if matched_kws:
                trigger.append(f"keywords: {matched_kws}")
            if is_label_safety:
                trigger.append(f"label: {label_code}")
            logger.info(f"Safety triggered by: {' + '.join(trigger)}")

        return is_safety, matched_kws, label_code

    # ---- 转人工判断 ----

    def should_transfer(
        self, intent_result: Dict, query: str
    ) -> Tuple[bool, str]:
        """
        判断是否需要转人工
        返回 (should_transfer, reason)
        """
        # 安全类强制转人工
        if intent_result.get("is_safety"):
            return True, "安全紧急事件，强制转人工"

        # 低置信度转人工（放宽阈值，减少不必要的转人工）
        conf = intent_result.get("confidence", 0)
        if conf < 0.35:
            return True, f"意图置信度过低({conf:.0%})"

        # 信息咨询类模式 — 即便含敏感词也是查信息，不转人工
        info_patterns = [
            "电话", "多少", "多久", "流程", "在哪", "怎么", "如何",
            "什么", "号码", "热线", "几点", "地址", "材料", "条件",
            "设置", "要求", "标准", "规范", "规定", "有人",
        ]
        is_info = any(p in query for p in info_patterns)

        if not is_info:
            # 情绪/投诉关键词
            emotion_kw = ["投诉", "举报", "赔偿", "起诉", "媒体", "曝光",
                         "12345", "态度差", "乱收费", "不作为", "敷衍", "推诿"]
            for kw in emotion_kw:
                if kw in query:
                    return True, f"检测到投诉/情绪关键词'{kw}'"

            # 法律相关
            legal_kw = ["法律", "律师", "法院", "诉讼", "判决", "强制执行"]
            for kw in legal_kw:
                if kw in query:
                    return True, "涉及法律事务"

        return False, ""

    def get_clarification_question(self, intent_result: Dict) -> str:
        """生成澄清追问（置信度 0.5-0.8 时）"""
        cat1 = intent_result.get("intent_name", "")
        cat2 = intent_result.get("sub_intent", "")

        if cat2:
            return (
                f"您是想咨询【{cat1} - {cat2}】相关的问题吗？"
                f"请再详细描述一下您的需求，我可以给您更准确的解答。"
            )
        elif cat1:
            return (
                f"您的问题可能和【{cat1}】相关。"
                f"请问具体是关于{cat1}的哪个方面呢？"
                f"比如报装、缴费、安检还是其他？"
            )
        else:
            return (
                "抱歉，我没有完全理解您的问题。能否再详细描述一下？"
                "或者您可以直接拨打人工客服 0734-8192944 咨询。"
            )

    # ---- Fallback ----

    @staticmethod
    def _default_gas_leak_template() -> Dict:
        return {
            "title": "⚠️ 燃气安全紧急处置",
            "steps": [
                "① 立即关闭燃气总阀门",
                "② 打开所有门窗，保持通风",
                "③ 切勿开关任何电器",
                "④ 不要使用明火、打火机",
                "⑤ 迅速撤离到室外安全区域",
                "⑥ 在室外拨打24小时抢修专线：95777",
            ],
            "warnings": ["🚫 禁止开关电器", "🚫 禁止使用明火"],
            "transfer": True,
            "transfer_msg": "已为您优先接入人工客服和抢修调度。",
        }


# 单例
_gateway_instance: Optional[SafetyGateway] = None


def get_gateway() -> SafetyGateway:
    global _gateway_instance
    if _gateway_instance is None:
        _gateway_instance = SafetyGateway()
    return _gateway_instance
