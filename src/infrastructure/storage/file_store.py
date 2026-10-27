"""
文件存储 — JSON 文件的读写封装

提供带错误处理、编码保证、原子写入的文件操作
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional


class FileStore:
    """JSON 文件存储"""

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ---- 读 ----

    def read_json(self, filename: str) -> Optional[Any]:
        """读取 JSON 文件，失败返回 None"""
        filepath = self._resolve(filename)
        if not filepath.exists():
            return None
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"[FileStore] Read error {filepath}: {e}")
            return None

    def read_json_list(self, filename: str) -> List:
        """读取 JSON 数组文件，失败返回空列表"""
        data = self.read_json(filename)
        if isinstance(data, list):
            return data
        return []

    def read_json_dict(self, filename: str) -> Dict:
        """读取 JSON 对象文件，失败返回空字典"""
        data = self.read_json(filename)
        if isinstance(data, dict):
            return data
        return {}

    # ---- 写 ----

    def write_json(self, filename: str, data: Any, atomic: bool = True):
        """写入 JSON 文件，默认原子写入（先写临时文件再 rename）"""
        filepath = self._resolve(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        json_str = json.dumps(data, ensure_ascii=False, indent=2)

        if atomic:
            # 原子写入：先写临时文件，再重命名
            tmp = tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                suffix=".json",
                dir=str(filepath.parent),
                delete=False,
            )
            try:
                tmp.write(json_str)
                tmp.flush()
                os.fsync(tmp.fileno())
                tmp.close()
                os.replace(tmp.name, str(filepath))
            except Exception:
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)
                raise
        else:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(json_str)

    def append_jsonl(self, filename: str, record: Dict):
        """追加一行到 JSONL 文件"""
        filepath = self._resolve(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ---- 工具 ----

    def exists(self, filename: str) -> bool:
        return self._resolve(filename).exists()

    def delete(self, filename: str) -> bool:
        filepath = self._resolve(filename)
        if filepath.exists():
            filepath.unlink()
            return True
        return False

    def list_files(self, pattern: str = "*") -> List[str]:
        """列出匹配模式的文件"""
        return [str(p.name) for p in self.base_dir.glob(pattern)]

    def _resolve(self, filename: str) -> Path:
        """解析文件路径（含路径遍历防护）"""
        # 禁止路径遍历：拒绝包含路径分隔符、空文件名、".." 等危险模式
        if not filename or ".." in filename or "/" in filename or "\\" in filename:
            raise ValueError(f"Invalid filename: {filename!r}")
        resolved = (self.base_dir / filename).resolve()
        # 确保解析后的路径仍在 base_dir 之下
        if not str(resolved).startswith(str(self.base_dir.resolve())):
            raise ValueError(f"Path traversal detected: {filename!r}")
        return resolved
