from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path = "configs/default.yaml", output_dir: str | None = None) -> dict:
    """Load the YAML config. Relative paths are resolved against the repo root."""
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if output_dir is not None:
        cfg["output_dir"] = output_dir
    return cfg


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path
