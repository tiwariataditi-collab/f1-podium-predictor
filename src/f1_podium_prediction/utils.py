from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib


def ensure_parent(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def save_json(data: dict[str, Any], path: str | Path) -> None:
    ensure_parent(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_object(obj: Any, path: str | Path) -> None:
    joblib.dump(obj, ensure_parent(path))


def load_object(path: str | Path) -> Any:
    return joblib.load(path)
