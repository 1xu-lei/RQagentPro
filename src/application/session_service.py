"""
会话服务 — 会话生命周期管理

特性：
- 内存缓存 + 磁盘持久化
- 会话自动过期（可配置 TTL）
- 最大会话数限制
- 未覆盖问题记录
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from src.infrastructure.config import get_config
from src.infrastructure.logging import get_logger
from src.infrastructure.storage import FileStore

logger = get_logger(__name__)


class Session:
    """单个会话"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = datetime.now()
        self.last_active = datetime.now()
        self.messages: List[Dict] = []
        self.context: Dict = {}
        self.closed = False

    def add_message(self, role: str, content: str, **meta):
        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            **meta,
        }
        self.messages.append(msg)
        self.last_active = datetime.now()
        return msg

    def get_history(self, last_n: int = 3) -> List[Dict]:
        """获取最近 N 轮对话"""
        return self.messages[-(last_n * 2):]

    def get_last_intent(self) -> Optional[Dict]:
        """获取最近一次意图"""
        for msg in reversed(self.messages):
            if "intent" in msg and msg["intent"]:
                return msg["intent"]
        return None

    def is_expired(self, ttl_minutes: int) -> bool:
        """检查会话是否过期"""
        return datetime.now() - self.last_active > timedelta(minutes=ttl_minutes)

    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat(),
            "messages": self.messages,
            "context": self.context,
            "closed": self.closed,
        }


class SessionService:
    """会话管理器"""

    def __init__(self):
        cfg = get_config()
        self.ttl_minutes = cfg.get("session.ttl_minutes", 60)
        self.max_sessions = cfg.get("session.max_sessions", 1000)
        self.persist_to_disk = cfg.get("session.persist_to_disk", True)

        self.sessions: Dict[str, Session] = {}

        # 存储
        log_dir = cfg.resolve_path("paths.log_dir")
        self.store = FileStore(str(log_dir))

    def create_session(self) -> str:
        """创建新会话"""
        sid = f"sess_{int(time.time() * 1000)}"

        # 清理过期会话（腾空间）
        if len(self.sessions) >= self.max_sessions:
            self._evict_expired()

        self.sessions[sid] = Session(sid)
        logger.debug(f"Session created: {sid}")
        return sid

    def get_session(self, session_id: str) -> Optional[Session]:
        """获取会话（内存 → 磁盘）"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            if session.is_expired(self.ttl_minutes):
                logger.debug(f"Session expired: {session_id}")
                self.close_session(session_id)
                return None
            return session

        # 从磁盘恢复
        if self.persist_to_disk:
            return self._restore_from_disk(session_id)

        return None

    def save_session(self, session_id: str):
        """保存会话到磁盘"""
        if not self.persist_to_disk:
            return
        if session_id not in self.sessions:
            return
        session = self.sessions[session_id]
        self.store.write_json(f"{session_id}.json", session.to_dict())

    def close_session(self, session_id: str):
        """关闭会话"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.closed = True
            self.save_session(session_id)
            del self.sessions[session_id]
            logger.debug(f"Session closed: {session_id}")

    def delete_session(self, session_id: str) -> bool:
        """删除会话（内存 + 磁盘）"""
        if session_id in self.sessions:
            del self.sessions[session_id]
        return self.store.delete(f"{session_id}.json")

    def list_sessions(self, limit: int = 20) -> List[Dict]:
        """列出所有会话"""
        sessions = []

        # 内存中的会话
        for session in self.sessions.values():
            sessions.append({
                "session_id": session.session_id,
                "created_at": session.created_at.isoformat(),
                "message_count": len(session.messages),
                "first_message": self._get_first_message(session),
                "closed": session.closed,
            })

        # 磁盘中的会话
        for fname in self.store.list_files("sess_*.json"):
            sid = fname.replace(".json", "")
            if sid not in self.sessions:
                data = self.store.read_json(fname)
                if data:
                    sessions.append({
                        "session_id": data.get("session_id", sid),
                        "created_at": data.get("created_at", ""),
                        "message_count": len(data.get("messages", [])),
                        "first_message": self._get_first_message_from_data(
                            data.get("messages", [])
                        ),
                        "closed": data.get("closed", False),
                    })

        sessions.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return sessions[:limit]

    def add_uncovered_query(self, query: str, intent: dict):
        """记录未覆盖问题"""
        record = {
            "query": query,
            "intent": intent,
            "timestamp": datetime.now().isoformat(),
        }
        self.store.append_jsonl("uncovered_queries.jsonl", record)

    # ---- 内部方法 ----

    def _evict_expired(self):
        """清理过期会话"""
        expired = [
            sid for sid, s in self.sessions.items()
            if s.is_expired(self.ttl_minutes)
        ]
        for sid in expired:
            del self.sessions[sid]
        logger.debug(f"Evicted {len(expired)} expired sessions")

    def _restore_from_disk(self, session_id: str) -> Optional[Session]:
        """从磁盘恢复会话"""
        data = self.store.read_json(f"{session_id}.json")
        if not data:
            return None

        session = Session(session_id)
        session.created_at = datetime.fromisoformat(data.get("created_at", ""))
        session.messages = data.get("messages", [])
        session.context = data.get("context", {})
        session.closed = data.get("closed", False)
        session.last_active = datetime.now()

        # 检查过期
        if session.is_expired(self.ttl_minutes):
            return None

        self.sessions[session_id] = session
        return session

    @staticmethod
    def _get_first_message(session: Session) -> str:
        for msg in session.messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                return content[:40] if len(content) > 40 else content
        return "（空对话）"

    @staticmethod
    def _get_first_message_from_data(messages: List[Dict]) -> str:
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                return content[:40] if len(content) > 40 else content
        return "（空对话）"


# 单例
_session_service_instance: Optional[SessionService] = None


def get_session_service() -> SessionService:
    global _session_service_instance
    if _session_service_instance is None:
        _session_service_instance = SessionService()
    return _session_service_instance
