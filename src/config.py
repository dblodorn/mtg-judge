"""Config loading — YAML profile + environment overrides."""

import os
from pathlib import Path

import yaml

DEFAULT_CONFIG = "config/default.yaml"

# Env var → config key. Env wins over YAML so `LLM_MODEL=qwen2.5:7b make run` works.
ENV_OVERRIDES = {
    "LLM_MODEL": "llm_model",
    "EMBED_MODEL": "embed_model",
    "OLLAMA_HOST": "ollama_host",
}


def load_config(path: str | None = None) -> dict:
    """Load a config profile. `path` > $MTG_JUDGE_CONFIG > config/default.yaml.

    A non-default profile (e.g. config/pi.yaml) is layered on top of the default,
    so it only needs to list the keys it changes.
    """
    path = path or os.environ.get("MTG_JUDGE_CONFIG") or DEFAULT_CONFIG
    config = yaml.safe_load(Path(DEFAULT_CONFIG).read_text()) or {}
    if Path(path) != Path(DEFAULT_CONFIG):
        config.update(yaml.safe_load(Path(path).read_text()) or {})

    for env, key in ENV_OVERRIDES.items():
        if os.environ.get(env):
            config[key] = os.environ[env]
    if os.environ.get("LLM_THINK"):
        config["think"] = os.environ["LLM_THINK"].lower() in ("1", "true", "yes")
    return config
