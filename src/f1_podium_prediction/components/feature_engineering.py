from __future__ import annotations

import numpy as np
import pandas as pd

from src.f1_podium_prediction.configuration import FeatureConfig
from src.f1_podium_prediction.logger import get_logger


ID_COLUMNS = ["raceId", "driverId", "constructorId", "date"]


class FeatureEngineer:
    def __init__(self, feature_config: FeatureConfig) -> None:
        self.feature_config = feature_config
        self.logger = get_logger(__name__)

    def build_model_frame(self, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
        races = tables["races"][
            ["raceId", "year", "round", "circuitId", "name", "date"]
        ].rename(
            columns={"name": "race_name"}
        )

        results = tables["results"][
            ["raceId", "driverId", "constructorId", "grid", "positionOrder", "points"]
        ]

        drivers = tables["drivers"][
            ["driverId", "driverRef", "forename", "surname", "nationality"]
        ].rename(
            columns={
                "nationality": "driver_nationality"
            }
        )

        constructors = tables["constructors"][
            ["constructorId", "constructorRef", "name", "nationality"]
        ].rename(
            columns={
                "name": "constructor_name",
                "nationality": "constructor_nationality",
            }
        )

        circuits = tables["circuits"][
            ["circuitId", "circuitRef", "country"]
        ].rename(columns={"country": "circuit_country"})

        qualifying = (
            tables["qualifying"][
                ["raceId", "driverId", "constructorId", "position"]
            ]
            .rename(columns={"position": "qualifying_position"})
            .sort_values(["raceId", "driverId", "qualifying_position"])
            .drop_duplicates(
                ["raceId", "driverId", "constructorId"],
                keep="first",
            )
        )

        frame = (
            results.merge(races, on="raceId", how="left")
            .merge(drivers, on="driverId", how="left")
            .merge(constructors, on="constructorId", how="left")
            .merge(circuits, on="circuitId", how="left")
            .merge(
                qualifying,
                on=["raceId", "driverId", "constructorId"],
                how="left",
            )
        )

        return self._prepare_features(frame)

    def _prepare_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()

        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")

        frame["driver_name"] = (
            frame["forename"].fillna("")
            + " "
            + frame["surname"].fillna("")
        ).str.strip()

        
        frame["grid"] = (
            pd.to_numeric(frame["grid"], errors="coerce")
            .replace(0, np.nan)
        )

        frame["qualifying_position"] = pd.to_numeric(
            frame["qualifying_position"],
            errors="coerce",
        )

        frame["qualifying_position"] = frame["qualifying_position"].fillna(
            frame["grid"]
        )

        frame["positionOrder"] = pd.to_numeric(
            frame["positionOrder"],
            errors="coerce",
        )

        frame["points"] = (
            pd.to_numeric(frame["points"], errors="coerce")
            .fillna(0.0)
        )

        frame[self.feature_config.target] = (
            frame["positionOrder"] <= 3
        ).astype(int)

        frame = frame.sort_values(
            ["date", "raceId", "positionOrder"],
            kind="mergesort",
        )

        for entity in ["driver", "constructor"]:
            key = f"{entity}Id"

            group = frame.groupby(key, sort=False)

            starts = group.cumcount()

            podiums_before = (
                group[self.feature_config.target].cumsum()
                - frame[self.feature_config.target]
            )

            points_before = (
                group["points"].cumsum()
                - frame["points"]
            )

            frame[f"{entity}_prior_starts"] = starts

            frame[f"{entity}_prior_podium_rate"] = np.where(
                starts > 0,
                podiums_before / starts,
                0.0,
            )

            frame[f"{entity}_prior_avg_points"] = np.where(
                starts > 0,
                points_before / starts,
                0.0,
            )

        for column in self.feature_config.numeric_features:
            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            )

        for column in self.feature_config.categorical_features:
            frame[column] = (
                frame[column]
                .fillna("unknown")
                .astype(str)
            )

        required_columns = (
            ID_COLUMNS
            + [self.feature_config.target]
            + self.feature_config.model_features
        )

        missing = [col for col in required_columns if col not in frame.columns]

        if missing:
            raise ValueError(f"Missing columns: {missing}")

        self.logger.info(
            "Feature frame created with shape %s",
            frame[required_columns].shape,
        )

        return frame[required_columns]

if __name__ == "__main__":

    from src.f1_podium_prediction.configuration import load_config
    from src.f1_podium_prediction.components.data_ingestion import DataIngestion

    print("=" * 20)
    print("FEATURE ENGINEERING TEST")
    print("=" * 20)

    try:
        config = load_config()

        ingestion = DataIngestion(config.data)
        tables = ingestion.load_raw_tables()

        print("\n Raw datasets loaded successfully!")

        # Feature Engineering
        engineer = FeatureEngineer(config.features)

        processed_df = engineer.build_model_frame(tables)

        config.data.processed_dir.mkdir(parents=True, exist_ok=True)

        output_path = config.data.processed_dir / "processed_f1_data.csv"

        processed_df.to_csv(output_path, index=False)

        print("\n Feature Engineering Successful!")

        print(f"\nProcessed Dataset Shape : {processed_df.shape}")

        print(f"\nDataset Saved At : {output_path}")

        print("\nFirst 5 Rows:\n")
        print(processed_df.head())

    except Exception as e:
        print("\n  Feature Engineering Failed!")
        print(e)