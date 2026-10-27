"""
意图分类工具 — 关键词投票 → BGE质心 → RAG条目 → 加权集成

架构: 四层 fallback，逐级提升准确率
  关键词投票 (60%查询, 89%准确) → BGE语义 (补充) → RAG条目 (补充) → 加权集成

标签体系对齐 v2: 所有 intent_name 输出自动标准化为规范域名
"""

import json, os, re
from pathlib import Path
from typing import Optional
from src.domain.tools.base import BaseTool
from src.domain.taxonomy import normalize_domain_name


class IntentClassifyTool(BaseTool):
    name = "classify_user_intent"
    description = "识别用户问题的意图分类。仅在 search_knowledge_base 无结果时使用。"
    priority = 70

    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "用户原始问题文本"},
            "context_intent": {"type": "object", "description": "可选，上一轮对话的意图分类结果"},
        },
        "required": ["query"],
    }

    @staticmethod
    def _std(name: str) -> str:
        """标准化域名（旧名 → 规范名）"""
        return normalize_domain_name(name)

    # ---- 领域模式补位（已对齐规范域名） ----
    DOMAIN_PATTERNS = [
        # 投诉语境优先（防止业务名词抢走投诉意图）
        ("投诉", "投诉建议", "服务态度投诉"),
        ("举报", "投诉建议", "服务态度投诉"),
        ("态度差", "投诉建议", "服务态度投诉"),
        ("态度不好", "投诉建议", "服务态度投诉"),
        ("乱收费", "投诉建议", "乱收费投诉"),
        ("敷衍", "投诉建议", "服务态度投诉"),
        ("不作为", "投诉建议", "服务态度投诉"),
        ("没人来", "投诉建议", "抢修不及时投诉"),
        ("上门.*慢", "投诉建议", "抢修不及时投诉"),
        ("太慢", "投诉建议", "抢修不及时投诉"),
        # 其他模式
        ("户主", "账户信息", "户号查询"),
        ("抄表", "抄表计量", "抄表周期查询"),
        ("报修", "报修抢修", "抢修进度/时效"),
        ("抢修", "报修抢修", "抢修进度/时效"),
        ("入户安检", "营业服务", "上门服务预约"),
        ("初装费", "价格政策", "收费项目明细"),
        ("气价", "价格政策", "居民气价标准"),
        ("手机.*缴费", "缴费充值", "线上缴费操作"),
        ("漏气.*注意", "安全用气", "泄漏应急处置"),
    ]

    def __init__(self):
        self._kb = None
        self._keyword_index = None
        self._centroids = None        # BGE质心: {cat1: np.ndarray}
        self._centroid_labels = None

    # ---- 数据加载 ----

    @property
    def kb(self):
        if self._kb is None:
            from src.infrastructure.storage.kb_store import get_kb_store
            self._kb = get_kb_store().get_all()
        return self._kb

    @property
    def keyword_index(self):
        if self._keyword_index is None:
            self._keyword_index = {}
            for item in self.kb:
                for kw in item.get("关键词", "").split():
                    if kw not in self._keyword_index:
                        self._keyword_index[kw] = []
                    self._keyword_index[kw].append(item["id"])
        return self._keyword_index

    # ---- BGE 质心构建 ----

    def _build_centroids(self):
        """预计算每个类别的 BGE 质心（含离群提纯 + LLM 增强数据）"""
        if self._centroids is not None:
            return

        import numpy as np
        from src.infrastructure.embedding import get_embedder
        embedder = get_embedder()
        embedder.load()

        # 收集每个一级域的文本（标准化域名，合并新旧命名）
        cat_texts = {}
        for item in self.kb:
            cat1 = self._std(item["一级域"])
            text = f"{cat1} {item['二级标签']} {item['标准问']}"
            cat_texts.setdefault(cat1, []).append(text)

        # 加载 LLM 增强数据（标准化域名以合并新旧命名）
        aug_path = Path(__file__).parent.parent.parent.parent / "data" / "augmented_queries.json"
        aug_count = 0
        if aug_path.exists():
            with open(aug_path, "r", encoding="utf-8") as f:
                aug_data = json.load(f)
            for cat1, queries in aug_data.items():
                std_cat = self._std(cat1)  # 标准化旧域名
                cat_texts.setdefault(std_cat, [])
                for q in queries:
                    cat_texts[std_cat].append(f"{std_cat} {q}")
                aug_count += len(queries)
            print(f"[BGE-Classify] Loaded {aug_count} augmented queries")

        # 构建质心（含离群提纯 + 逆频率权重）
        self._centroids = {}
        self._centroid_weights = {}  # 类别权重（抑制大类偏差）
        total_removed = 0
        for cat1, texts in cat_texts.items():
            if not texts:
                continue
            embeddings = embedder.encode(texts, normalize=True, show_progress=False)
            raw_centroid = np.mean(embeddings, axis=0)

            # 离群提纯（仅对样本充足的类别）
            if len(texts) >= 8:
                sims = np.dot(embeddings, raw_centroid)
                threshold = np.mean(sims) - 2.0 * np.std(sims)
                keep_mask = sims >= threshold
                if np.sum(keep_mask) >= 3:
                    removed = len(texts) - np.sum(keep_mask)
                    if removed > 0:
                        embeddings = embeddings[keep_mask]
                        total_removed += removed

            self._centroids[cat1] = np.mean(embeddings, axis=0)
            # 逆频率权重：大类降权，小类升权 (sqrt 抑制极端差异)
            self._centroid_weights[cat1] = 1.0 / np.sqrt(len(texts))

        # 权重归一化到 [0.5, 1.5] 范围
        min_w = min(self._centroid_weights.values())
        max_w = max(self._centroid_weights.values())
        for cat in self._centroid_weights:
            normalized = 0.5 + (self._centroid_weights[cat] - min_w) / (max_w - min_w)
            self._centroid_weights[cat] = normalized

        self._centroid_labels = list(self._centroids.keys())
        print(f"[BGE-Classify] {len(self._centroids)} centroids built "
              f"({total_removed} outliers removed, dim={next(iter(self._centroids.values())).shape[0]})")
        # 打印权重供调试
        top_w = sorted(self._centroid_weights.items(), key=lambda x: -x[1])[:3]
        low_w = sorted(self._centroid_weights.items(), key=lambda x: x[1])[:3]
        print(f"  Weights: highest={[(c,round(w,2)) for c,w in top_w]}, "
              f"lowest={[(c,round(w,2)) for c,w in low_w]}")

    # ---- BGE 语义分类 ----

    def _bge_classify(self, query: str) -> dict:
        """BGE 余弦相似度分类，返回所有类别的分数"""
        import numpy as np
        from src.infrastructure.embedding import get_embedder
        try:
            self._build_centroids()
            embedder = get_embedder()
            q_vec = embedder.encode([query], normalize=True)[0]

            all_scores = {}
            for cat1, centroid in self._centroids.items():
                all_scores[cat1] = float(np.dot(q_vec, centroid))

            best_cat = max(all_scores, key=all_scores.get)

            # 最常见二级标签
            best_cat2 = ""
            cat2_count = {}
            for item in self.kb:
                if item["一级域"] == best_cat:
                    cat2_count[item["二级标签"]] = cat2_count.get(item["二级标签"], 0) + 1
            if cat2_count:
                best_cat2 = max(cat2_count, key=cat2_count.get)

            return {
                "intent_name": self._std(best_cat), "sub_intent": best_cat2,
                "confidence": round(min(0.95, max(0.35, all_scores[best_cat])), 2),
                "_all_scores": all_scores,
            }
        except Exception as e:
            return {"intent_name": "增值与其他", "sub_intent": "", "confidence": 0.30, "_all_scores": {}}

    # ---- RAG 条目级投票 ----

    async def _rag_vote(self, query: str) -> dict:
        """检索最相似的具体条目，按类别投票"""
        from src.domain.retriever.hybrid import get_retriever
        try:
            retriever = get_retriever()
            retriever.load()
            results = await retriever.hybrid_search(query, top_k=5)
            cat_votes = {}
            for r in results:
                cat1 = self._std(r["entry"]["一级域"])
                cat_votes[cat1] = cat_votes.get(cat1, 0) + r.get("score", 0.5)
            if not cat_votes:
                return {"best": "", "scores": {}}
            best = max(cat_votes, key=cat_votes.get)
            total = max(sum(cat_votes.values()), 0.001)
            return {"best": best, "scores": {k: v/total for k, v in cat_votes.items()}}
        except Exception:
            return {"best": "", "scores": {}}

    # ---- 加权集成 ----

    async def _ensemble_classify(self, query: str, kw_result: dict) -> dict:
        """三信号加权：关键词(0.30) + BGE质心(0.50) + RAG条目(0.20)"""
        kw_conf = kw_result.get("confidence", 0.3)
        kw_cat = kw_result.get("intent_name", "")

        # BGE
        bge_result = self._bge_classify(query)
        bge_scores = bge_result.get("_all_scores", {})
        bge_cat = bge_result.get("intent_name", "")

        # RAG（仅在关键词和BGE不一致时启用）
        rag_result = {"best": "", "scores": {}}
        if kw_cat != bge_cat and kw_conf < 0.70:
            rag_result = await self._rag_vote(query)

        # 加权求和（应用类别权重抑制大类偏差）
        all_cats = set(list(bge_scores.keys()) + [kw_cat])
        if rag_result["best"]:
            all_cats.update(rag_result["scores"].keys())

        final_scores = {}
        for cat in all_cats:
            score = 0.0
            if cat == kw_cat:
                score += 0.30 * kw_conf
            # BGE 分数乘以类别权重（大类降权，小类升权）
            bge_w = self._centroid_weights.get(cat, 1.0) if hasattr(self, '_centroid_weights') and self._centroid_weights else 1.0
            score += 0.50 * bge_scores.get(cat, 0.0) * bge_w
            score += 0.20 * rag_result["scores"].get(cat, 0.0)
            final_scores[cat] = score

        best_cat = max(final_scores, key=final_scores.get)

        # 二级标签
        best_cat2 = ""
        cat2_count = {}
        for item in self.kb:
            if item["一级域"] == best_cat:
                cat2_count[item["二级标签"]] = cat2_count.get(item["二级标签"], 0) + 1
        if cat2_count:
            best_cat2 = max(cat2_count, key=cat2_count.get)

        # 置信度 = BGE余弦（天然校准）
        bge_conf = bge_scores.get(best_cat, bge_scores.get(bge_cat, 0.35))
        confidence = round(min(0.95, max(0.35, bge_conf)), 2)

        # 方法标记
        methods_used = [m for m in [
            "kw" if best_cat == kw_cat else "",
            "bge" if best_cat == bge_cat else "",
            "rag" if best_cat == rag_result.get("best", "") else "",
        ] if m]

        return {
            "method": f"ensemble({'+'.join(methods_used)})",
            "intent_name": self._std(best_cat), "sub_intent": best_cat2,
            "confidence": confidence,
        }

    # ---- 主入口 ----

    async def execute(self, query: str, context_intent: Optional[dict] = None) -> str:
        # Step 0: 上下文意图继承（短追问）
        if context_intent and len(query) <= 15:
            followup = ["多少钱","收费","价格","费用","地址","在哪","电话","材料","流程",
                        "怎么办","需要什么","然后","接着","之后","第二种","别的","其他",
                        "还有","步骤","微信","公众号","网上","线上","手机","怎么交","怎么查",
                        "多久","什么条件","哪个","哪家"]
            if any(p in query for p in followup):
                return json.dumps({
                    "method": "context_inherit",
                    "intent_name": self._std(context_intent.get("intent_name", "")),
                    "sub_intent": context_intent.get("sub_intent", ""),
                    "confidence": 0.85,
                }, ensure_ascii=False)

        # Step 1: 投诉语境前置检测（防止业务名词抢走投诉意图）
        complaint_signal = 0
        complaint_kw = {
            "态度差": 3, "态度不好": 3, "态度太": 3, "态度恶劣": 3,
            "乱收费": 3, "多收费": 3, "收费不合理": 3, "收费太贵": 3,
            "敷衍": 3, "推诿": 3, "不作为": 3, "不管": 2,
            "没人来": 3, "没人管": 3, "不到人": 2, "不给处理": 3,
            "举报": 3, "投诉": 3, "起诉": 3, "曝光": 2, "媒体": 2,
            "表扬": 3, "感谢": 2, "谢谢": 1,
            "太慢": 3, "太差": 3, "太贵": 3, "不行": 1,
            "合理吗": 2, "不合理": 3, "多收": 3, "乱收": 3,
            "不履职": 3, "渎职": 3, "刁难": 3, "违规收费": 3,
            "强制要求": 3, "怎么处理": 1, "意见": 1,
        }
        for kw, weight in complaint_kw.items():
            if kw in query:
                complaint_signal += weight

        # Step 2: jieba 关键词投票
        try:
            import jieba
            words = list(jieba.lcut(query))
            # 子词补丁：防复合词过度合并导致关键词漏匹配
            extra = []
            for w in words:
                for prefix in ["燃气","缴费","报装","安检","抄表"]:
                    if prefix in w and w != prefix:
                        extra.append(prefix)
            words.extend(extra)
        except Exception:
            words = list(query)

        keyword_scores = {}
        for word in words:
            if word in self.keyword_index:
                for item_id in self.keyword_index[word]:
                    keyword_scores[item_id] = keyword_scores.get(item_id, 0) + 1

        if keyword_scores:
            cat1_votes, cat2_votes = {}, {}
            top_items = sorted(keyword_scores.items(), key=lambda x: -x[1])[:5]
            top_ids = [i for i, s in top_items]
            for item in self.kb:
                if item["id"] in top_ids:
                    cat1_votes[item["一级域"]] = cat1_votes.get(item["一级域"], 0) + 1
                    cat2_votes[item["二级标签"]] = cat2_votes.get(item["二级标签"], 0) + 1

            # 投诉信号加权（对抗嵌入的业务名词）
            if complaint_signal >= 3:
                cat1_votes["投诉建议"] = cat1_votes.get("投诉建议", 0) + complaint_signal

            if cat1_votes:
                best_cat1 = max(cat1_votes, key=cat1_votes.get)
                best_cat2 = max(cat2_votes, key=cat2_votes.get) if cat2_votes else ""
                total_votes = sum(cat1_votes.values())
                kw_conf = min(0.90, max(0.35, cat1_votes[best_cat1] / max(total_votes, 1)))

                # 高置信 → 直接返回关键词结果
                if kw_conf >= 0.55:
                    return json.dumps({
                        "method": "keyword_match",
                        "intent_name": self._std(best_cat1), "sub_intent": best_cat2,
                        "confidence": kw_conf,
                    }, ensure_ascii=False)

                # 低置信 → 走加权集成
                kw_result = {"intent_name": self._std(best_cat1), "sub_intent": best_cat2, "confidence": kw_conf}
                return json.dumps(await self._ensemble_classify(query, kw_result), ensure_ascii=False)

        # Step 2: 无关键词匹配 → 加权集成
        ensemble_result = await self._ensemble_classify(query, {"intent_name": "", "confidence": 0.0})
        if ensemble_result["confidence"] >= 0.40:
            return json.dumps(ensemble_result, ensure_ascii=False)

        # Step 3: 领域模式兜底
        for pattern, cat1, cat2 in self.DOMAIN_PATTERNS:
            if re.search(pattern, query):
                return json.dumps({
                    "method": "domain_pattern",
                    "intent_name": self._std(cat1), "sub_intent": cat2,
                    "confidence": 0.65,
                }, ensure_ascii=False)

        # Step 4: 上下文 fallback
        if context_intent and context_intent.get("intent_name"):
            return json.dumps({
                "method": "context_fallback",
                "intent_name": self._std(context_intent.get("intent_name", "")),
                "sub_intent": context_intent.get("sub_intent", ""),
                "confidence": 0.45,
            }, ensure_ascii=False)

        # 兜底
        return json.dumps({
            "method": "fast_fallback", "intent_name": "增值与其他", "sub_intent": "",
            "confidence": 0.30,
        }, ensure_ascii=False)
