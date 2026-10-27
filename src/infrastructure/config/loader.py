"""
配置加载器 — 基于 YAML 的统一配置管理

支持：
- 主配置文件 (config.yaml)
- Prompt 模板 (prompts/*.yaml)
- 环境变量覆盖（DEEPSEEK_API_KEY 等敏感信息）
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


class ConfigLoader:
    """YAML 配置加载器，支持嵌套键的 '.' 分隔访问"""

    def __init__(self, config_dir: str = None):
        if config_dir is None:
            # 默认从项目根目录的 config/ 加载
            config_dir = Path(__file__).parent.parent.parent.parent / "config"
        self.config_dir = Path(config_dir)
        self._config: Dict[str, Any] = {}
        self._prompts: Dict[str, Dict] = {}

    # ---- 加载 ----

    def load(self) -> "ConfigLoader":
        """加载主配置和所有 prompt 模板"""
        self._load_main_config()
        self._load_prompts()
        self._apply_env_overrides()
        return self

    def _load_main_config(self):
        config_path = self.config_dir / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)
        print(f"[Config] Loaded: {config_path}")

    def _load_prompts(self):
        prompt_dir = self.config_dir / "prompts"
        if not prompt_dir.exists():
            return

        for yaml_file in prompt_dir.glob("*.yaml"):
            name = yaml_file.stem  # 文件名（不含扩展名）
            with open(yaml_file, "r", encoding="utf-8") as f:
                self._prompts[name] = yaml.safe_load(f)
            print(f"[Config] Prompt loaded: {name}")

    def _apply_env_overrides(self):
        """环境变量覆盖配置文件中的值"""
        overrides = {
            "llm.api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
            "server.port": os.environ.get("PORT"),
            "app.debug": os.environ.get("DEBUG"),
        }
        for key, value in overrides.items():
            if value is not None:
                # 转换类型
                if value.lower() in ("true", "false"):
                    value = value.lower() == "true"
                elif value.isdigit():
                    value = int(value)
                self.set(key, value)

    # ---- 访问 ----

    def get(self, key: str, default: Any = None) -> Any:
        """按 '.' 分隔路径获取配置值，如 'llm.model'"""
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value

    def set(self, key: str, value: Any):
        """设置配置值"""
        keys = key.split(".")
        target = self._config
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
        target[keys[-1]] = value

    def all(self) -> Dict[str, Any]:
        """返回完整配置"""
        return self._config.copy()

    # ---- Prompt 访问 ----

    def prompt(self, name: str = "system") -> Dict:
        """获取指定 prompt 模板"""
        return self._prompts.get(name, {})

    def all_prompts(self) -> Dict[str, Dict]:
        return self._prompts.copy()

    # ---- 路径解析 ----

    def resolve_path(self, config_key: str) -> Path:
        """将配置中的相对路径解析为绝对路径"""
        relative = self.get(config_key, "")
        if not relative:
            return Path()
        project_root = Path(__file__).parent.parent.parent.parent
        return project_root / relative


# ============================================================
# 单例
# ============================================================

_config_instance: Optional[ConfigLoader] = None


def get_config() -> ConfigLoader:
    """获取全局配置单例"""
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigLoader().load()
    return _config_instance


def init_config(config_dir: str = None) -> ConfigLoader:
    """初始化配置（可用于指定自定义目录）"""
    global _config_instance
    _config_instance = ConfigLoader(config_dir).load()
    return _config_instance
