from __future__ import annotations

import argparse
from pathlib import Path

from src.f1_podium_prediction.configuration import load_config
from src.f1_podium_prediction.logger import get_logger
from src.f1_podium_prediction.predictor import PodiumPredictor


def run_prediction(
    config_path: str,
    input_path: str,
    output_path: str,
    threshold: float,
) -> None:
    config = load_config(config_path)

    logger = get_logger(
        __name__,
        config.logging.log_file,
    )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between 0 and 1.")

    input_file = Path(input_path)

    if not input_file.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_file}"
        )

    logger.info("STARTING F1 PODIUM PREDICTION PIPELINE")
    logger.info("Loading trained model...")

    predictor = PodiumPredictor(config)

    logger.info("Using model: %s", predictor.best_model_name)
    logger.info("Reading input data from %s", input_file)

    output = predictor.predict_csv(
        input_path=input_file,
        output_path=output_path,
        threshold=threshold,
    )

    logger.info("PREDICTION PIPELINE COMPLETED SUCCESSFULLY")

    print("\nPREDICTION COMPLETED SUCCESSFULLY")
    print(f"Model Used  : {predictor.best_model_name}")
    print(f"Output File : {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run batch prediction."
    )

    parser.add_argument(
        "--config",
        default="configs/config.yml",
    )

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument(
        "--output",
        default="outputs/predictions.csv",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_prediction(
        args.config,
        args.input,
        args.output,
        args.threshold,
    )