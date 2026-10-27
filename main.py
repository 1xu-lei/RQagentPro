"""
RQagentPro — 衡阳燃气AI客服系统 v3.1
分层架构: 接口层 → 应用层 → 领域层 → 基础设施层

启动方式:
    python main.py
"""
import sys
import os
import asyncio
from pathlib import Path

# 将项目根目录加入 path
PROJECT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_DIR))

# Windows 终端 UTF-8 编码支持（解决 emoji 打印报错）
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_env():
    """加载 .env 文件"""
    env_file = PROJECT_DIR / ".env"
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())
        print("[OK] .env loaded")
    else:
        print(f"[!] 未找到 .env 文件: {env_file}")
        print(f"    请复制 {PROJECT_DIR / '.env.example'} → {env_file}")
        print(f"    编辑 {env_file}，填入 DEEPSEEK_API_KEY=sk-xxx")


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════╗
║       衡阳燃气AI客服智能体系 v3.1 (分层架构版)            ║
║       RQagentPro — Hengyang Gas AI Customer Service      ║
║                                                          ║
║       架构: Interface → Application → Domain → Infra      ║
║       引擎: DeepSeek Function Calling (async)             ║
║       检索: FAISS + BM25 (RRF融合) + Reranker            ║
║       前端: Vue 3 + Naive UI                             ║
║                                                          ║
║       v2.0 → v3.1: 分层解耦 + 流式输出 + 结构化日志       ║
╚══════════════════════════════════════════════════════════╝
""")


def check_prerequisites() -> bool:
    """检查启动前置条件"""
    all_ok = True

    # 检查 .env / API Key
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print(f"[!] DEEPSEEK_API_KEY 未设置，Agent 将无法工作")
        print(f"    1. 复制 {PROJECT_DIR / '.env.example'} → {PROJECT_DIR / '.env'}")
        print(f"    2. 编辑 {PROJECT_DIR / '.env'}，写入 DEEPSEEK_API_KEY=sk-xxx")
        print(f"    3. 或在终端执行: set DEEPSEEK_API_KEY=sk-xxx")
        all_ok = False

    # 检查知识库（已迁移至 SQLite）
    kb_path = PROJECT_DIR / "data" / "kb.sqlite3"
    if not kb_path.exists():
        print(f"[!] 知识库文件不存在: {kb_path}")
        print(f"    请将 kb.sqlite3 放入 {PROJECT_DIR / 'data'}\\ 目录")
        all_ok = False

    # 检查嵌入模型
    model_dir = PROJECT_DIR / "models" / "bge-base-zh-v1.5"
    if not model_dir.exists():
        print(f"[!] BGE 嵌入模型目录不存在: {model_dir}")
        print(f"    请从旧项目复制模型到 {model_dir}")
        print(f"    或下载 BAAI/bge-base-zh-v1.5 放入该目录")
        all_ok = False

    return all_ok


def find_available_port(start=8001, end=8010):
    """扫描可用端口"""
    import socket
    for port in range(start, end + 1):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", port))
            sock.close()
            return port
        except OSError:
            sock.close()
    return None


async def warmup():
    """预热：加载模型、建立连接池、验证连通性"""
    from src.infrastructure.config import get_config
    from src.infrastructure.logging import get_logger, setup_logging

    cfg = get_config()

    # 初始化日志
    setup_logging(cfg.get("logging", {}))
    logger = get_logger("main")

    logger.info("Warming up system components...")

    # 1. 预热嵌入模型
    try:
        from src.infrastructure.embedding import get_embedder
        embedder = get_embedder()
        embedder.load()
        embedder.encode(["预热测试"])
        logger.info("Embedding model warmed up")
    except Exception as e:
        logger.error(f"Embedding model warmup failed: {e}")

    # 2. 预热 LLM 连接
    client = None
    try:
        from src.infrastructure.llm import LLMClient
        client = LLMClient(
            api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
            api_url=cfg.get("llm.api_url"),
            model=cfg.get("llm.model"),
            timeout=cfg.get("llm.timeout", 20),
        )
        resp = await client.chat(
            messages=[{"role": "user", "content": "OK"}],
            max_tokens=5,
            temperature=0,
        )
        logger.info(f"LLM connectivity verified (model: {resp.model})")
    except Exception as e:
        logger.warning(f"LLM warmup failed (will retry on first request): {e}")
    finally:
        # 确保 httpx 连接在事件循环关闭前清理，避免 Windows Proactor 析构报错
        if client is not None:
            try:
                await client.close()
            except Exception:
                pass

    # 3. 预热检索引擎
    try:
        from src.domain.retriever.hybrid import get_retriever
        retriever = get_retriever()
        logger.info(f"Retriever warmed up ({len(retriever.kb)} entries)")
    except Exception as e:
        logger.warning(f"Retriever warmup failed: {e}")

    logger.info("Warmup complete")


def main():
    """主入口"""
    load_env()
    print_banner()

    if not check_prerequisites():
        print("\n[!] 前置条件未满足，请按上述提示处理后重新启动")
        sys.exit(1)

    # 端口检测
    port = find_available_port(8001, 8010)
    if port is None:
        print("[!] 端口 8001-8010 全部被占用，请手动释放后重试")
        sys.exit(1)

    # 将端口号传递给 API 层（startup 预热完成后打印访问链接）
    os.environ["RQAGENT_PORT"] = str(port)

    # ★ 同步预热：在启动 uvicorn 前完成模型加载等耗时初始化
    # 这样 startup 事件中可以直接使用已预热的组件
    print("\n[预热] 正在加载模型和连接...")
    try:
        asyncio.run(warmup())
    except Exception as e:
        print(f"[WARNING] 预热未完全成功（服务仍可启动）: {e}")

    import uvicorn

    uvicorn.run(
        "src.interface.api:create_app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
        factory=True,
    )


if __name__ == "__main__":
    main()
