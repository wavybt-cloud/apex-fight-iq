"""Configuration loading with reproducibility metadata.

The resolved config dict is hashed into every run record so any historical
prediction can be tied back to the exact parameters that produced it.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent.parent   # nflquant/ project dir
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "config" / "default.yaml"


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    cfg["_meta"] = {
        "config_path": str(cfg_path),
        "config_hash": config_hash(cfg),
        "git_commit": _git_commit(),
    }
    return cfg


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def config_hash(cfg: dict) -> str:
    clean = {k: v for k, v in cfg.items() if k != "_meta"}
    return hashlib.sha256(json.dumps(clean, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PACKAGE_ROOT,
            capture_output=True, text=True, timeout=10,
        ).stdout.strip() or None
    except Exception:
        return None


def cache_dir(cfg: dict) -> Path:
    d = Path(cfg["data"]["cache_dir"])
    if not d.is_absolute():
        d = PACKAGE_ROOT / d
    d.mkdir(parents=True, exist_ok=True)
    return d


def ca_bundle(cfg: dict) -> str | bool:
    """TLS verification argument for requests: agent-proxy CA bundle if present."""
    env = os.environ.get("NFLQUANT_CA_BUNDLE")
    if env and Path(env).exists():
        return env
    p = cfg.get("data", {}).get("ca_bundle")
    if p and Path(p).exists():
        return p
    return True
