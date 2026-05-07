"""Configuration utilities."""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv


def load_yaml(path: str | Path) -> dict:
    """Load YAML config."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_env(path: str | Path = "configs/teacher.env") -> None:
    """Load environment variables."""
    if Path(path).exists():
        load_dotenv(path)


def ensure_dir(path: str | Path) -> Path:
    """Ensure directory exists."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path