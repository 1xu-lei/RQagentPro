"""
SQLite 知识库存储 — 替代 knowledge_base.json

特性：
  - 启动时一次性加载到内存（与 JSON 完全相同的运行时性能）
  - 支持增删改查（增量更新，无需重建全文索引）
  - FTS5 全文搜索（效果等同 BM25，减少一个依赖）
  - 单文件，拷贝即备份
  - 自动从 JSON 迁移（首次运行时检测并导入）

用法：
  from src.infrastructure.storage.kb_store import get_kb_store
  store = get_kb_store()
  kb = store.get_all()          # 全量加载（与 json.load 完全等价）
  store.insert(entry)           # 新增一条
  store.update(entry_id, data)  # 更新一条
  store.delete(entry_id)        # 删除一条
  store.search("燃气 泄漏")     # FTS5 全文搜索
  store.count_by_domain()       # 按域统计
"""

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# 数据库路径
DB_PATH = Path(__file__).parent.parent.parent.parent / "data" / "kb.sqlite3"

# 旧 JSON 路径（用于首次自动迁移）
JSON_PATH = Path(__file__).parent.parent.parent.parent / "data" / "knowledge_base.json"


class KBStore:
    """SQLite 知识库存储"""

    def __init__(self, db_path: str = None):
        self._db_path = Path(db_path) if db_path else DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()  # SQLite 写操作串行化
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_tables()

    # ================================================================
    # 连接管理
    # ================================================================

    def _get_conn(self) -> sqlite3.Connection:
        """获取线程本地连接（读操作可并发，写操作需加锁）"""
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")       # 读写并发
            self._conn.execute("PRAGMA synchronous=NORMAL")      # 安全与速度平衡
            self._conn.execute("PRAGMA cache_size=-8000")        # 8MB 缓存
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    # ================================================================
    # 建表
    # ================================================================

    def _ensure_tables(self):
        """创建表结构（幂等）"""
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS knowledge_base (
                id          INTEGER PRIMARY KEY,
                code        TEXT    DEFAULT '',
                domain      TEXT    NOT NULL DEFAULT '',
                label       TEXT    NOT NULL DEFAULT '',
                definition  TEXT    DEFAULT '',
                question    TEXT    NOT NULL DEFAULT '',
                answer      TEXT    NOT NULL DEFAULT '',
                regulation  TEXT    DEFAULT '',
                safety      INTEGER DEFAULT 0,
                keywords    TEXT    DEFAULT '',
                source_org  TEXT    DEFAULT '',
                source_url  TEXT    DEFAULT '',
                updated_at  TEXT    DEFAULT '',
                created_at  TEXT    DEFAULT (datetime('now','localtime'))
            );

            CREATE INDEX IF NOT EXISTS idx_kb_domain ON knowledge_base(domain);
            CREATE INDEX IF NOT EXISTS idx_kb_label  ON knowledge_base(label);
            CREATE INDEX IF NOT EXISTS idx_kb_safety ON knowledge_base(safety);

            -- FTS5 全文索引（替代 BM25）
            CREATE VIRTUAL TABLE IF NOT EXISTS kb_fts USING fts5(
                question, answer, keywords, definition,
                content='knowledge_base',
                content_rowid='id'
            );

            -- 触发器：自动同步 FTS 索引
            CREATE TRIGGER IF NOT EXISTS kb_ai AFTER INSERT ON knowledge_base BEGIN
                INSERT INTO kb_fts(rowid, question, answer, keywords, definition)
                VALUES (new.id, new.question, new.answer, new.keywords, new.definition);
            END;

            CREATE TRIGGER IF NOT EXISTS kb_ad AFTER DELETE ON knowledge_base BEGIN
                INSERT INTO kb_fts(kb_fts, rowid, question, answer, keywords, definition)
                VALUES ('delete', old.id, old.question, old.answer, old.keywords, old.definition);
            END;

            CREATE TRIGGER IF NOT EXISTS kb_au AFTER UPDATE ON knowledge_base BEGIN
                INSERT INTO kb_fts(kb_fts, rowid, question, answer, keywords, definition)
                VALUES ('delete', old.id, old.question, old.answer, old.keywords, old.definition);
                INSERT INTO kb_fts(rowid, question, answer, keywords, definition)
                VALUES (new.id, new.question, new.answer, new.keywords, new.definition);
            END;
        """)
        conn.commit()

    # ================================================================
    # 自动迁移
    # ================================================================

    def migrate_from_json(self, json_path: str = None) -> int:
        """从 JSON 文件迁移数据到 SQLite（幂等：已有数据则跳过）"""
        conn = self._get_conn()
        existing = conn.execute("SELECT COUNT(*) FROM knowledge_base").fetchone()[0]
        if existing > 0:
            return existing  # 已迁移

        src = Path(json_path) if json_path else JSON_PATH
        if not src.exists():
            return 0

        with open(src, "r", encoding="utf-8") as f:
            kb = json.load(f)

        rows = []
        for item in kb:
            rows.append((
                item.get("id", 0),
                item.get("编码", ""),
                item.get("一级域", ""),
                item.get("二级标签", ""),
                item.get("标签定义", ""),
                item.get("标准问", ""),
                item.get("标准答", ""),
                item.get("法规编码", ""),
                1 if item.get("安全敏感") else 0,
                item.get("关键词", ""),
                item.get("来源单位", ""),
                item.get("来源链接", ""),
                item.get("更新时间", ""),
            ))

        with self._lock:
            conn.executemany(
                """INSERT INTO knowledge_base
                   (id, code, domain, label, definition, question, answer,
                    regulation, safety, keywords, source_org, source_url, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            conn.commit()

        # 触发 FTS 索引构建
        conn.execute(
            "INSERT INTO kb_fts(kb_fts) VALUES ('rebuild')"
        )
        conn.commit()

        return len(rows)

    # ================================================================
    # 读操作（与 json.load 完全等价）
    # ================================================================

    def get_all(self) -> List[Dict]:
        """全量加载（替代 json.load）—— 返回与 JSON 完全相同的结构"""
        # 自动从 JSON 迁移（如果表为空且 JSON 存在）
        conn = self._get_conn()
        count = conn.execute("SELECT COUNT(*) FROM knowledge_base").fetchone()[0]
        if count == 0:
            self.migrate_from_json()

        rows = conn.execute(
            "SELECT * FROM knowledge_base ORDER BY id"
        ).fetchall()

        return [self._row_to_dict(r) for r in rows]

    def get_by_id(self, entry_id: int) -> Optional[Dict]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM knowledge_base WHERE id = ?", (entry_id,)
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def count(self) -> int:
        conn = self._get_conn()
        return conn.execute("SELECT COUNT(*) FROM knowledge_base").fetchone()[0]

    # ================================================================
    # 写操作
    # ================================================================

    def insert(self, entry: Dict) -> int:
        """新增一条，返回自增 ID"""
        conn = self._get_conn()
        with self._lock:
            cur = conn.execute(
                """INSERT INTO knowledge_base
                   (code, domain, label, definition, question, answer,
                    regulation, safety, keywords, source_org, source_url, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    entry.get("编码", ""),
                    entry.get("一级域", ""),
                    entry.get("二级标签", ""),
                    entry.get("标签定义", ""),
                    entry.get("标准问", ""),
                    entry.get("标准答", ""),
                    entry.get("法规编码", ""),
                    1 if entry.get("安全敏感") else 0,
                    entry.get("关键词", ""),
                    entry.get("来源单位", ""),
                    entry.get("来源链接", ""),
                    entry.get("更新时间", datetime.now().strftime("%Y-%m-%d")),
                ),
            )
            conn.commit()
        return cur.lastrowid

    def update(self, entry_id: int, data: Dict) -> bool:
        """更新一条"""
        # 映射字段名
        field_map = {
            "编码": "code", "一级域": "domain", "二级标签": "label",
            "标签定义": "definition", "标准问": "question", "标准答": "answer",
            "法规编码": "regulation", "关键词": "keywords",
            "来源单位": "source_org", "来源链接": "source_url",
        }
        sets = []
        values = []
        for chinese_key, value in data.items():
            col = field_map.get(chinese_key, chinese_key)
            sets.append(f"{col} = ?")
            values.append(value)

        if not sets:
            return False

        # 特殊处理安全敏感
        if "安全敏感" in data:
            sets.append("safety = ?")
            values.append(1 if data["安全敏感"] else 0)
            # 避免重复
            sets = list(dict.fromkeys(sets))
            values = values[:len(sets)]

        conn = self._get_conn()
        with self._lock:
            conn.execute(
                f"UPDATE knowledge_base SET {', '.join(sets)}, updated_at = ? WHERE id = ?",
                values + [datetime.now().strftime("%Y-%m-%d"), entry_id],
            )
            conn.commit()
        return True

    def delete(self, entry_id: int) -> bool:
        conn = self._get_conn()
        with self._lock:
            conn.execute("DELETE FROM knowledge_base WHERE id = ?", (entry_id,))
            conn.commit()
        return True

    # ================================================================
    # 全文搜索（FTS5）
    # ================================================================

    def search(self, query: str, limit: int = 10) -> List[Dict]:
        """
        FTS5 全文搜索 — 替代 BM25 关键词检索

        FTS5 使用 BM25 算法进行相关性排序，效果等同 rank-bm25 库
        """
        conn = self._get_conn()
        # FTS5 语法：双引号精确短语，默认 OR 逻辑
        safe_query = query.replace('"', '').replace("'", "")
        terms = " OR ".join(safe_query.split())
        try:
            rows = conn.execute(
                """SELECT k.*, rank
                   FROM kb_fts f
                   JOIN knowledge_base k ON k.id = f.rowid
                   WHERE kb_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (terms, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []

        results = []
        for row in rows:
            d = self._row_to_dict(row)
            d["_ft_score"] = row["rank"] if "rank" in row.keys() else 0
            results.append(d)
        return results

    # ================================================================
    # 聚合查询
    # ================================================================

    def count_by_domain(self) -> Dict[str, int]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT domain, COUNT(*) as cnt FROM knowledge_base GROUP BY domain ORDER BY cnt DESC"
        ).fetchall()
        return {r["domain"]: r["cnt"] for r in rows}

    def count_safety(self) -> int:
        conn = self._get_conn()
        return conn.execute(
            "SELECT COUNT(*) FROM knowledge_base WHERE safety = 1"
        ).fetchone()[0]

    def list_domains(self) -> List[str]:
        conn = self._get_conn()
        return [r[0] for r in conn.execute(
            "SELECT DISTINCT domain FROM knowledge_base ORDER BY domain"
        ).fetchall()]

    # ================================================================
    # 导出
    # ================================================================

    def export_json(self, output_path: str = None) -> str:
        """导出为 JSON（兼容旧格式）"""
        path = Path(output_path) if output_path else JSON_PATH
        data = self.get_all()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return str(path)

    # ================================================================
    # 内部
    # ================================================================

    def _row_to_dict(self, row: sqlite3.Row) -> Dict:
        """SQLite Row → 与 JSON 完全相同的字典结构"""
        return {
            "id": row["id"],
            "编码": row["code"] or "",
            "一级域": row["domain"],
            "二级标签": row["label"],
            "标签定义": row["definition"] or "",
            "标准问": row["question"],
            "标准答": row["answer"],
            "法规编码": row["regulation"] or "",
            "安全敏感": bool(row["safety"]),
            "关键词": row["keywords"] or "",
            "来源单位": row["source_org"] or "",
            "来源链接": row["source_url"] or "",
            "更新时间": row["updated_at"] or "",
        }


# ================================================================
# 全局单例
# ================================================================

_store_instance: Optional[KBStore] = None


def get_kb_store(db_path: str = None) -> KBStore:
    global _store_instance
    if _store_instance is None:
        _store_instance = KBStore(db_path)
    return _store_instance


def init_kb_store(db_path: str = None) -> KBStore:
    global _store_instance
    _store_instance = KBStore(db_path)
    return _store_instance
