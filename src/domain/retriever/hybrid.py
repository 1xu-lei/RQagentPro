"""
混合检索引擎 — FAISS 向量检索 + BM25 关键词检索 + RRF 融合

优化：
  - 向量检索和 BM25 并行执行（asyncio + 线程池）
  - jieba 启动预加载
"""

import asyncio
import concurrent.futures
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.infrastructure.logging import get_logger

logger = get_logger(__name__)

# 共享线程池（用于并行执行向量检索和关键词检索）
_search_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="search")

# 路径常量
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INDEX_DIR = DATA_DIR / "indexes"
FAISS_PATH = INDEX_DIR / "faiss.index"
BM25_PATH = INDEX_DIR / "bm25.pkl"


class HybridRetriever:
    """混合检索器：FAISS + BM25 + RRF 融合 + 意图域加权"""

    def __init__(self):
        self.kb: List[Dict] = []
        self.doc_texts: List[str] = []
        self.faiss_index = None
        self.bm25 = None
        self._loaded = False

    def load(self):
        """加载知识库和索引"""
        if self._loaded:
            return

        # jieba 预加载 — 消除首次调用的延迟
        try:
            import jieba
            jieba.initialize()
            # 预热：切分一条测试文本，触发词典加载和缓存初始化
            _ = list(jieba.cut("衡阳燃气开户报装缴费查询"))
            logger.info("jieba tokenizer preloaded")
        except Exception as e:
            logger.debug(f"jieba preload skipped: {e}")

        # 从 SQLite 加载知识库（自动从 JSON 迁移）
        from src.infrastructure.storage.kb_store import get_kb_store
        store = get_kb_store()
        self.kb = store.get_all()
        logger.info(f"Knowledge base loaded: {len(self.kb)} entries (SQLite)")

        # 构建文档文本
        self.doc_texts = []
        for item in self.kb:
            text = f"{item['一级域']} {item['二级标签']} {item['标准问']} {item['标签定义']}"
            self.doc_texts.append(text)

        # 加载 FAISS 索引（检查维度一致性，不匹配则重建）
        from src.infrastructure.embedding import get_embedder
        embedder = get_embedder()
        embedder.load()
        embedder.load_pca()  # 尝试加载已保存的 PCA

        if FAISS_PATH.exists() and embedder.dim > 0:
            try:
                import faiss
                self.faiss_index = faiss.read_index(str(FAISS_PATH))
                if self.faiss_index.d != embedder.dim:
                    logger.warning(
                        f"FAISS dim mismatch: index={self.faiss_index.d} vs embedder={embedder.dim}, rebuilding..."
                    )
                    self.faiss_index = None
                else:
                    logger.info(f"FAISS index loaded: {self.faiss_index.ntotal} vectors @ {self.faiss_index.d}d")
            except Exception as e:
                logger.warning(f"FAISS index load failed: {e}")
        if self.faiss_index is None:
            self._build_faiss()

        # 加载 BM25 索引（不存在则构建）
        if BM25_PATH.exists():
            try:
                with open(BM25_PATH, "rb") as f:
                    self.bm25 = pickle.load(f)
                logger.info("BM25 index loaded")
            except Exception as e:
                logger.warning(f"BM25 index load failed: {e}")
        if self.bm25 is None:
            self._build_bm25()

        self._loaded = True

    def _build_faiss(self):
        """构建 FAISS 向量索引（自动适配 PCA 降维后的维度）"""
        import faiss
        from src.infrastructure.embedding import get_embedder
        embedder = get_embedder()
        embedder.load()

        # 检查并加载 PCA（如果已保存）
        if not embedder.load_pca():
            # 首次运行：拟合 PCA
            logger.info("PCA not found, fitting on knowledge base...")
            embedder.fit_pca(self.doc_texts, target_dim=256)

        dim = embedder.dim
        logger.info(f"Building FAISS index (dim={dim})...")
        embeddings = embedder.encode(self.doc_texts, show_progress=True, normalize=True)
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(embeddings.astype(np.float32))
        FAISS_PATH.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.faiss_index, str(FAISS_PATH))
        logger.info(f"FAISS index built: {self.faiss_index.ntotal} vectors @ {dim}d")

    def _build_bm25(self):
        """构建 BM25 关键词索引"""
        from rank_bm25 import BM25Okapi
        logger.info("Building BM25 index...")
        try:
            import jieba
            tokenized = [list(jieba.cut(text)) for text in self.doc_texts]
        except Exception:
            tokenized = [text.split() for text in self.doc_texts]
        self.bm25 = BM25Okapi(tokenized)
        BM25_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(BM25_PATH, "wb") as f:
            pickle.dump(self.bm25, f)
        logger.info(f"BM25 index built: {len(self.doc_texts)} documents")

    # ---- 检索 ----

    def vector_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """FAISS 向量语义检索（自动应用 PCA 降维）"""
        if self.faiss_index is None:
            return []

        from src.infrastructure.embedding import get_embedder
        embedder = get_embedder()
        # encode() 自动应用 PCA（如果已拟合）
        q_embedding = embedder.encode([query], normalize=True).astype(np.float32)

        scores, indices = self.faiss_index.search(q_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self.kb):
                results.append({
                    "entry": self.kb[idx],
                    "score": float(score),
                    "source": "vector",
                })
        return results

    def keyword_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """BM25 关键词检索"""
        if self.bm25 is None:
            return []

        try:
            import jieba
            tokenized = list(jieba.cut(query))
        except Exception:
            tokenized = query.split()

        scores = self.bm25.get_scores(tokenized)

        max_score = max(scores) if max(scores) > 0 else 1
        norm_scores = scores / max_score

        top_indices = np.argsort(norm_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if norm_scores[idx] > 0:
                results.append({
                    "entry": self.kb[idx],
                    "score": float(norm_scores[idx]),
                    "source": "bm25",
                })
        return results

    async def hybrid_search(
        self,
        query: str,
        intent: Optional[Dict] = None,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        混合检索：向量 + BM25 + RRF 融合 + 意图域加权

        Args:
            query: 检索查询词
            intent: 意图信息 {'intent_name': '...', 'sub_intent': '...', 'confidence': 0.8}
            top_k: 返回结果数

        Returns:
            排序后的检索结果列表
        """
        self.load()

        from src.infrastructure.tracing import get_tracer
        tracer = get_tracer()

        # 并行执行向量检索和关键词检索（各自挂 span；gather 子任务会继承当前 span 上下文）
        async def _run_vector():
            with tracer.span("retrieve.faiss") as sp:
                loop = asyncio.get_running_loop()
                res = await loop.run_in_executor(_search_executor, self.vector_search, query, top_k * 2)
                sp.set_attr("hits", len(res))
                return res

        async def _run_keyword():
            with tracer.span("retrieve.bm25") as sp:
                loop = asyncio.get_running_loop()
                res = await loop.run_in_executor(_search_executor, self.keyword_search, query, top_k * 2)
                sp.set_attr("hits", len(res))
                return res

        try:
            vector_results, keyword_results = await asyncio.gather(_run_vector(), _run_keyword())
        except RuntimeError:
            # 无事件循环（同步调用场景）
            vector_results = self.vector_search(query, top_k=top_k * 2)
            keyword_results = self.keyword_search(query, top_k=top_k * 2)

        # RRF 融合
        with tracer.span("retrieve.rrf") as rrf_span:
            scores: Dict = {}
            vector_scores: Dict = {}

            for rank, r in enumerate(vector_results):
                eid = r["entry"]["id"]
                scores[eid] = scores.get(eid, 0) + 1.0 / (rank + 60)
                scores[f"{eid}_entry"] = r
                if eid not in vector_scores or r["score"] > vector_scores[eid]:
                    vector_scores[eid] = r["score"]

            for rank, r in enumerate(keyword_results):
                eid = r["entry"]["id"]
                scores[eid] = scores.get(eid, 0) + 1.0 / (rank + 60)
                scores[f"{eid}_entry"] = r

            # 意图域加权
            if intent and intent.get("confidence", 0) >= 0.5:
                intent_cat1 = intent.get("intent_name", "")
                intent_cat2 = intent.get("sub_intent", "")
                for key in list(scores.keys()):
                    if isinstance(key, str) and key.endswith("_entry"):
                        eid = int(key.replace("_entry", ""))
                        entry = scores[key]["entry"]
                        if entry["一级域"] == intent_cat1:
                            scores[eid] = scores.get(eid, 0) * 1.3
                        if intent_cat2 and entry["二级标签"] == intent_cat2:
                            scores[eid] = scores.get(eid, 0) * 1.5

            # 排序取 top-k
            entry_ids = [k for k in scores if isinstance(k, int)]
            entry_ids.sort(key=lambda x: scores[x], reverse=True)

            results = []
            for eid in entry_ids[:top_k]:
                entry_key = f"{eid}_entry"
                if entry_key in scores:
                    results.append({
                        "entry": scores[entry_key]["entry"],
                        "score": scores[eid],
                        "vector_score": round(vector_scores.get(eid, 0.0), 4),
                        "source": "hybrid",
                    })

            rrf_span.set_attr("fused", len(entry_ids))
            rrf_span.set_attr("returned", len(results))

        logger.debug(f"Hybrid search '{query[:50]}': {len(results)} results, "
                    f"top_score={results[0]['score']:.3f}" if results else "no results")

        return results


# 单例
_retriever_instance: Optional[HybridRetriever] = None


def get_retriever() -> HybridRetriever:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = HybridRetriever()
    return _retriever_instance
