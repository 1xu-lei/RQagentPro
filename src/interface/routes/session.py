"""
会话管理接口
"""

from fastapi import APIRouter, HTTPException
from src.application.session_service import get_session_service

router = APIRouter(prefix="/api/v1")

session_mgr = get_session_service()


@router.get("/sessions")
async def list_sessions():
    """获取最近会话列表"""
    return {"sessions": session_mgr.list_sessions(limit=50)}


@router.get("/history/{session_id}")
async def get_history(session_id: str):
    """获取会话详情"""
    session = session_mgr.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.to_dict()


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """删除会话"""
    ok = session_mgr.delete_session(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted", "session_id": session_id}
