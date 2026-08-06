from __future__ import annotations

import pandas as pd

from src.f1_podium_prediction.configuration import DataConfig, load_config
from src.f1_podium_prediction.logger import get_logger


class DataIngestion:
    def __init__(self, data_config: DataConfig) -> None:
        self.data_config = data_config
        self.logger = get_logger(__name__)

    def _read(self, file_name: str) -> pd.DataFrame:
        path = self.data_config.raw_dir / file_name
        self.logger.info("Reading %s", path)

        return pd.read_csv(path)

    def load_raw_tables(self) -> dict[str, pd.DataFrame]:
        tables = {
            "races": self._read("races.csv"),
            "results": self._read("results.csv"),
            "drivers": self._read("drivers.csv"),
            "constructors": self._read("constructors.csv"),
            "qualifying": self._read("qualifying.csv"),
            "circuits": self._read("circuits.csv"),
        }

        self.logger.info("Successfully loaded all raw datasets.")

        return tables


if __name__ == "__main__":
    print("=" * 60)
    print("DATA INGESTION TEST")
    print("=" * 60)

    try:
        # Load configuration
        config = load_config()

        # Initialize ingestion
        ingestion = DataIngestion(config.data)

        # Load datasets
        tables = ingestion.load_raw_tables()

        print("\n Data Ingestion Successful!\n")

        print(f"{'Dataset':<20}{'Rows':>10}{'Columns':>12}")
        print("-" * 45)

        for name, df in tables.items():
            print(f"{name:<20}{df.shape[0]:>10}{df.shape[1]:>12}")

        print("\n  All datasets loaded successfully!")

    except Exception as e:
        print("\n Data Ingestion Failed!")
        print(e)