from __future__ import annotations

import pandas as pd

from src.f1_podium_prediction.configuration import DataConfig, FeatureConfig
from src.f1_podium_prediction.exception import DataValidationError
from src.f1_podium_prediction.logger import get_logger


class DataValidator:
    def __init__(
        self,
        data_config: DataConfig,
        feature_config: FeatureConfig,
    ) -> None:
        self.data_config = data_config
        self.feature_config = feature_config
        self.logger = get_logger(__name__)

    def validate_raw_files(self) -> None:
        missing = [
            name
            for name in self.data_config.required_files
            if not (self.data_config.raw_dir / name).exists()
        ]

        if missing:
            raise DataValidationError(
                f"Missing raw files in {self.data_config.raw_dir}: {missing}. "
                "Place the required CSV files inside data/raw/."
            )

        self.logger.info("Raw data validation passed.")

    @staticmethod
    def validate_columns(
        frame: pd.DataFrame,
        required_columns: list[str],
        dataset_name: str,
    ) -> None:
        missing = [
            column
            for column in required_columns
            if column not in frame.columns
        ]

        if missing:
            raise DataValidationError(
                f"{dataset_name} is missing required columns: {missing}"
            )

    def validate_inference_frame(
        self,
        frame: pd.DataFrame,
    ) -> None:
        self.validate_columns(
            frame,
            self.feature_config.model_features,
            "Inference data",
        )

        self.logger.info(
            "Inference frame validation passed."
        )


if __name__ == "__main__":
    print("=" * 60)
    print("DATA VALIDATION TEST")
    print("=" * 60)

    try:
        from src.f1_podium_prediction.configuration import load_config

        config = load_config()

        validator = DataValidator(
            config.data,
            config.features,
        )

        # Check raw CSV files
        validator.validate_raw_files()

        print("\nRaw files validation: PASSED")

        # Check processed train/test files
        for file_path, name in [
            (config.data.train_path, "Train"),
            (config.data.test_path, "Test"),
        ]:
            if not file_path.exists():
                raise DataValidationError(
                    f"{name} file not found: {file_path}"
                )

            df = pd.read_csv(file_path)

            validator.validate_columns(
                df,
                config.features.model_features,
                name,
            )

            print(
                f"{name} validation: PASSED | Shape: {df.shape}"
            )

        print("\n" + "=" * 60)
        print("DATA VALIDATION SUCCESSFUL!")
        print("=" * 60)

    except Exception as e:
        print("\nData Validation Failed!")
        print(f"{type(e).__name__}: {e}")