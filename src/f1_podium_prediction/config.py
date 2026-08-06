from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

def load_config(path : str | Path) -> dict[str, Any]:
    with Path(path).open('r', encoding = 'utf-8') as file:
        config = yaml.safe_load(file)
    if not config:
        raise ValueError(f"Config file is empty or invalid: {path}")
    return config

def ensure_parent(path : str | Path) -> None:
    Path(path).parent.mkdir(parents = True, exist_ok = True)

def ensure_dir(path : str | Path) -> None:
    Path(path).mkdir(parents = True, exist_ok = True)

        