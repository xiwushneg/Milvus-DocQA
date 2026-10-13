"""全局配置加载模块。

- 默认读取项目根目录 config/config.yaml
- 支持环境变量覆盖关键项（DOCQA_MILVUS_URI / DOCQA_EMBEDDING_PROVIDER /
  DOCQA_LLM_BASE_URL / DOCQA_LLM_MODEL）
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = os.environ.get(
    "DOCQA_CONFIG", str(PROJECT_ROOT / "config" / "config.yaml")
)


def _load_dotenv() -> None:
    """加载项目根目录 .env（存在时），不覆盖已设置的环境变量。"""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(PROJECT_ROOT / ".env", override=False)

# 环境变量覆盖表: (section, key) -> 环境变量名
_ENV_OVERRIDES = {
    ("milvus", "uri"): "DOCQA_MILVUS_URI",
    ("embedding", "provider"): "DOCQA_EMBEDDING_PROVIDER",
    ("llm", "base_url"): "DOCQA_LLM_BASE_URL",
    ("llm", "model"): "DOCQA_LLM_MODEL",
}


def load_config(path: str | None = None) -> dict[str, Any]:
    _load_dotenv()
    cfg_path = Path(path or DEFAULT_CONFIG_PATH)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    for (section, key), env_name in _ENV_OVERRIDES.items():
        val = os.environ.get(env_name)
        if val:
            cfg.setdefault(section, {})[key] = val
    return cfg
