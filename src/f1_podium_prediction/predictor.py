from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.f1_podium_prediction.components.data_validation import DataValidator
from src.f1_podium_prediction.configuration import AppConfig, load_config
from src.f1_podium_prediction.exception import ModelNotFoundError
from src.f1_podium_prediction.logger import get_logger
from src.f1_podium_prediction.utils import load_object


class PodiumPredictor:
    """Loads the trained model and makes F1 podium predictions."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or load_config()
        self.logger = get_logger(__name__)

        # Use the main model path first. If it does not exist,
        # fall back to the legacy model path.
        model_path = self.config.artifacts.model_path

        if not model_path.exists():
            legacy_path = self.config.artifacts.legacy_model_path

            if legacy_path.exists():
                model_path = legacy_path

        if not model_path.exists():
            raise ModelNotFoundError(
                f"Trained model was not found at: {model_path}"
            )

        self.logger.info("Loading trained model from %s", model_path)

        self.artifact = load_object(model_path)

        self.model = self.artifact["model"]
        self.features = self.artifact["features"]

        self.best_model_name = self.artifact.get(
            "best_model_name",
            "unknown",
        )

        self.validator = DataValidator(
            self.config.data,
            self.config.features,
        )

        self.logger.info(
            "Loaded model: %s",
            self.best_model_name,
        )

    def predict_frame(
        self,
        frame: pd.DataFrame,
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        """Make predictions for a DataFrame."""

        if not 0 <= threshold <= 1:
            raise ValueError(
                "Threshold must be between 0 and 1."
            )

        # Check whether all required model features are present.
        self.validator.validate_inference_frame(frame)

        result = frame.copy()
        X = result[self.features]

        # Probability of finishing on the podium.
        probabilities = self.model.predict_proba(X)[:, 1]

        result["podium_probability"] = probabilities

        result["predicted_is_podium"] = (
            probabilities >= threshold
        ).astype(int)

        # Show drivers with the highest podium probability first.
        result = result.sort_values(
            "podium_probability",
            ascending=False,
        ).reset_index(drop=True)

        self.logger.info(
            "Predictions generated for %d records.",
            len(result),
        )

        return result

    def predict_csv(
        self,
        input_path: str | Path,
        output_path: str | Path,
        threshold: float = 0.5,
    ) -> Path:
        """Read prediction data from a CSV and save the results."""

        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(
                f"Input file not found: {input_path}"
            )

        self.logger.info(
            "Reading prediction data from %s",
            input_path,
        )

        frame = pd.read_csv(input_path)

        predictions = self.predict_frame(
            frame,
            threshold=threshold,
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        predictions.to_csv(
            output_path,
            index=False,
        )

        self.logger.info(
            "Predictions saved to %s",
            output_path,
        )

        return output_path

    def predict_single(
        self,
        payload: dict[str, Any],
        threshold: float = 0.5,
    ) -> dict[str, Any]:
        """Make a prediction for a single driver/race input."""

        frame = pd.DataFrame([payload])

        prediction = self.predict_frame(
            frame,
            threshold=threshold,
        )

        return prediction.iloc[0].to_dict()