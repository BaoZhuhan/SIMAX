"""IO utilities for SIMAX."""

import json
import yaml
import logging
from pathlib import Path
from typing import Any, Union


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Setup a logger with the given name."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def load_yaml(path: Union[str, Path]) -> dict:
    """Load a YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: dict, path: Union[str, Path]) -> None:
    """Save data to a YAML file."""
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def load_json(path: Union[str, Path]) -> Any:
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> None:
    """Save data to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure a directory exists."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
