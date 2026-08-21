from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_FILES = [
    "races.csv",
    "results.csv",
    "drivers.csv",
    "constructors.csv",
    "qualifying.csv",
    "circuits.csv",
]


def read_raw_csv(
    raw_dir: str | Path,
    filename: str,
) -> pd.DataFrame:
    """Read one raw F1 CSV file."""

    path = Path(raw_dir) / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {path}"
        )

    return pd.read_csv(path)


def validate_raw_files(
    raw_dir: str | Path,
) -> None:
    """Check that all required raw datasets exist."""

    raw_dir = Path(raw_dir)

    missing = [
        filename
        for filename in REQUIRED_FILES
        if not (raw_dir / filename).exists()
    ]

    if missing:
        raise FileNotFoundError(
            f"Missing raw datasets in {raw_dir}: {missing}"
        )