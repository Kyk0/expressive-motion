"""Load pipeline settings and dotted-key overrides.

Defaults come from ``configs/pipeline.json``; callers may select another file
or override individual values without modifying source code.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "pipeline.json"


def _strip_comments(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _strip_comments(v) for k, v in node.items() if not k.startswith("_")}
    if isinstance(node, list):
        return [_strip_comments(v) for v in node]
    return node


def _coerce(text: str) -> Any:
    lowered = text.strip().lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("null", "none"):
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


class PipelineConfig:
    def __init__(self, data: dict):
        self._data = data

    @classmethod
    def load(cls, path: Path | None = None, overrides: list[str] | None = None) -> "PipelineConfig":
        source = Path(path) if path else DEFAULT_CONFIG
        if not source.is_file():
            raise FileNotFoundError(f"Pipeline config not found: {source}")

        data = _strip_comments(json.loads(source.read_text(encoding="utf-8")))
        config = cls(data)

        for override in overrides or []:
            if "=" not in override:
                raise ValueError(f"Override must be key=value, got {override!r}")
            key, value = override.split("=", 1)
            config.set(key.strip(), _coerce(value))

        return config

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node = self._data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def section(self, name: str) -> dict:
        value = self.get(name, {})
        return value if isinstance(value, dict) else {}

    def as_dict(self) -> dict:
        return copy.deepcopy(self._data)
