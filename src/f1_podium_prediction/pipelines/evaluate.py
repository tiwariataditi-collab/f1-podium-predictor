from __future__ import annotations

import argparse

import pandas as pd

from src.f1_podium_prediction.components.model_evaluation import ModelEvaluator
from src.f1_podium_prediction.configuration import load_config
from src.f1_podium_prediction.logger import get_logger
from src.f1_podium_prediction.predictor import PodiumPredictor


def run_evaluation(
    config_path: str = "configs/config.yml",
) -> dict:
    """Evaluate the saved F1 podium prediction model."""

    config = load_config(config_path)

    logger = get_logger(
        __name__,
        config.logging.log_file,
    )

    logger.info("STARTING F1 PODIUM MODEL EVALUATION")

    # Load the saved model through PodiumPredictor
    logger.info("Loading trained model...")

    predictor = PodiumPredictor(config)

    logger.info(
        "Loaded model: %s",
        predictor.best_model_name,
    )

    # Check that the test dataset exists
    if not config.data.test_path.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {config.data.test_path}. "
            "Please run the training pipeline first."
        )

    logger.info(
        "Loading test dataset from %s",
        config.data.test_path,
    )

    test = pd.read_csv(
        config.data.test_path,
    )

    logger.info(
        "Test dataset shape: %s",
        test.shape,
    )

    # Evaluate the model
    logger.info("Evaluating model...")

    evaluator = ModelEvaluator(config)

    metrics = evaluator.evaluate(
        predictor.model,
        test,
    )

    logger.info("MODEL EVALUATION COMPLETED SUCCESSFULLY")

    # Display final results
    print("\nMODEL EVALUATION COMPLETED SUCCESSFULLY")

    print(
        f"\nModel Used         : "
        f"{predictor.best_model_name}"
    )

    print(
        f"ROC-AUC            : "
        f"{metrics['roc_auc']:.4f}"
    )

    print(
        f"Average Precision  : "
        f"{metrics['average_precision']:.4f}"
    )

    print(
        f"\nMetrics Saved At   : "
        f"{config.artifacts.metrics_path}"
    )

    print(
        f"Figures Saved At   : "
        f"{config.artifacts.figures_dir}"
    )

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the trained F1 podium prediction model."
    )

    parser.add_argument(
        "--config",
        default="configs/config.yml",
        help="Path to the configuration file.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_evaluation(
        config_path=args.config,
    )