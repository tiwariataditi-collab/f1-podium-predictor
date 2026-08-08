from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any


def save_object(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("wb") as file:
        pickle.dump(obj, file)


def load_object(path: str | Path) -> Any:
    path = Path(path)

    with path.open("rb") as file:
        return pickle.load(file)


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=4,
        )