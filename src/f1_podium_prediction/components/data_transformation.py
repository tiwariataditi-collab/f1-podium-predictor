from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.f1_podium_prediction.configuration import (
    DataConfig,
    FeatureConfig,
    load_config,
)
from src.f1_podium_prediction.logger import get_logger


class DataTransformer:
    def __init__(
        self,
        data_config: DataConfig,
        feature_config: FeatureConfig,
    ) -> None:
        self.data_config = data_config
        self.feature_config = feature_config
        self.logger = get_logger(__name__)

    def get_preprocessor(self) -> ColumnTransformer:
        self.logger.info("Creating preprocessing pipeline...")

        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore"),
                ),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    numeric_pipeline,
                    self.feature_config.numeric_features,
                ),
                (
                    "categorical",
                    categorical_pipeline,
                    self.feature_config.categorical_features,
                ),
            ]
        )

        self.logger.info("Preprocessor created successfully.")

        return preprocessor

    def split_and_save(
        self,
        frame: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:

        self.logger.info("Starting train-test split...")

        if "date" not in frame.columns:
            raise ValueError("'date' column not found in dataframe.")

        split_year = pd.to_datetime(
            frame["date"],
            errors="coerce",
        ).dt.year

        train = frame[
            split_year < self.feature_config.test_start_year
        ].copy()

        test = frame[
            split_year >= self.feature_config.test_start_year
        ].copy()

        self.data_config.processed_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        train.to_csv(
            self.data_config.train_path,
            index=False,
        )

        test.to_csv(
            self.data_config.test_path,
            index=False,
        )

        self.logger.info(
            "Train Shape : %s",
            train.shape,
        )

        self.logger.info(
            "Test Shape : %s",
            test.shape,
        )

        self.logger.info("Train/Test datasets saved successfully.")

        return train, test


if __name__ == "__main__":

    print("=" * 20)
    print("DATA TRANSFORMATION TEST")
    print("=" * 20)

    try:

        config = load_config()

        transformer = DataTransformer(
            config.data,
            config.features,
        )

        file_path = (
            config.data.processed_dir
            / "processed_f1_data.csv"
        )

        df = pd.read_csv(file_path)

        train, test = transformer.split_and_save(df)

        print()

        print("Train Shape :", train.shape)

        print("Test Shape  :", test.shape)

        print()

        print("Data Transformation Successful!")

    except Exception as e:

        print()

        print(" Data Transformation Failed!")

        print(e)