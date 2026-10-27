"""
结构化日志系统

支持两种输出模式：
- structured: JSON 格式，适合生产环境日志收集
- console: 彩色控制台，适合开发调试

用法：
    from src.infrastructure.logging import get_logger
    logger = get_logger(__name__)
    logger.info("Agent started", extra={"session_id": "sess_123"})
"""

import logging
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Optional


class StructuredFormatter(logging.Formatter):
    """JSON 结构化日志格式化器"""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }

        # 合并 extra 字段
        if hasattr(record, "extra_data") and record.extra_data:
            log_entry.update(record.extra_data)

        # 异常信息
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = {
                "type": type(record.exc_info[1]).__name__,
                "message": str(record.exc_info[1]),
            }

        return json.dumps(log_entry, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    """彩色控制台格式化器"""

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",
        "DIM": "\033[2m",
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        reset = self.COLORS["RESET"]
        dim = self.COLORS["DIM"]

        # 时间
        ts = datetime.now().strftime("%H:%M:%S")
        # 级别
        level = f"{color}{record.levelname:<8}{reset}"
        # 来源
        source = f"{dim}{record.name}:{record.lineno}{reset}"
        # 消息
        msg = record.getMessage()

        return f"{dim}{ts}{reset} {level} {source}  {msg}"


class LoggerFactory:
    """日志工厂 — 根据配置创建 logger"""

    _initialized = False
    _log_level = logging.INFO
    _log_format = "console"
    _log_dir = None

    @classmethod
    def setup(
        cls,
        level: str = "INFO",
        fmt: str = "console",
        log_dir: Optional[str] = None,
    ):
        """初始化日志系统"""
        cls._log_level = getattr(logging, level.upper(), logging.INFO)
        cls._log_format = fmt
        cls._log_dir = Path(log_dir) if log_dir else None
        cls._initialized = True

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """获取指定名称的 logger"""
        if not cls._initialized:
            cls.setup()

        logger = logging.getLogger(name)
        logger.setLevel(cls._log_level)

        # 避免重复添加 handler
        if logger.handlers:
            return logger

        # 控制台 handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(cls._log_level)
        if cls._log_format == "structured":
            console_handler.setFormatter(StructuredFormatter())
        else:
            console_handler.setFormatter(ConsoleFormatter())
        logger.addHandler(console_handler)

        # 文件 handler（仅 structured 模式写入文件）
        if cls._log_dir and cls._log_format == "structured":
            cls._log_dir.mkdir(parents=True, exist_ok=True)
            today = datetime.now().strftime("%Y-%m-%d")
            log_file = cls._log_dir / f"rqagent-{today}.log"

            file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(StructuredFormatter())
            logger.addHandler(file_handler)

        logger.propagate = False
        return logger


def get_logger(name: str) -> logging.Logger:
    """便捷函数：获取 logger"""
    return LoggerFactory.get_logger(name)


def setup_logging(config: dict = None):
    """从配置字典初始化日志系统"""
    if config:
        LoggerFactory.setup(
            level=config.get("level", "INFO"),
            fmt=config.get("format", "console"),
            log_dir=config.get("log_dir"),
        )
    else:
        LoggerFactory.setup()
