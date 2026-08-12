from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import seaborn as sns

from src.f1_podium_prediction.components.data_ingestion import DataIngestion
from src.f1_podium_prediction.components.data_validation import DataValidator
from src.f1_podium_prediction.components.feature_engineering import FeatureEngineer
from src.f1_podium_prediction.configuration import load_config


def run_eda(config_path: str = "configs/config.yml") -> None:
    # Load configuration
    config = load_config(config_path)

    # Validate raw datasets
    validator = DataValidator(
        config.data,
        config.features,
    )
    validator.validate_raw_files()

    # Load raw datasets
    ingestion = DataIngestion(config.data)
    tables = ingestion.load_raw_tables()

    # Feature engineering
    engineer = FeatureEngineer(config.features)
    frame = engineer.build_model_frame(tables)

    # Create figures directory
    config.artifacts.figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 1. Target Distribution

    target_counts = (
        frame[config.features.target]
        .value_counts()
        .sort_index()
    )

    target_counts.index = target_counts.index.map(
        {
            0: "Not Podium",
            1: "Podium",
        }
    )

    plt.figure(figsize=(6, 4))

    sns.barplot(
        x=target_counts.index,
        y=target_counts.values,
    )

    plt.title("Podium Target Distribution")
    plt.xlabel("Result")
    plt.ylabel("Number of Drivers")
    plt.tight_layout()

    plt.savefig(
        config.artifacts.figures_dir
        / "target_distribution.png",
        dpi=160,
    )

    plt.close()

    # --------------------------------------------------
    # 2. Grid Position vs Podium Rate
    # --------------------------------------------------

    grid_summary = (
        frame[
            frame["grid"].between(1, 25)
        ]
        .groupby("grid")[config.features.target]
        .mean()
    )

    plt.figure(figsize=(9, 4))
    grid_summary.plot(marker="o",)
    plt.title("Grid Position vs Podium Rate")
    plt.xlabel("Grid Position")
    plt.ylabel("Podium Rate")
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        config.artifacts.figures_dir
        / "grid_vs_podium_rate.png",
        dpi=160,)
    plt.close()

    print("EDA COMPLETED SUCCESSFULLY")
   

    print(f"\nEDA figures saved at:"
        f"\n{config.artifacts.figures_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate EDA charts for F1 Podium Prediction."
    )

    parser.add_argument(
        "--config",
        default="configs/config.yml",
        help="Path to configuration file.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_eda(args.config)