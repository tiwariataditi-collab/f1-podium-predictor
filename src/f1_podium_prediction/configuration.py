from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.f1_podium_prediction.exception import DataValidationError


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    random_state: int


@dataclass(frozen=True)
class DataConfig:
    raw_dir: Path
    processed_dir: Path
    train_path: Path
    test_path: Path
    required_files: list[str]


@dataclass(frozen=True)
class FeatureConfig:
    target: str
    test_start_year: int
    numeric_features: list[str]
    categorical_features: list[str]

    @property
    def model_features(self) -> list[str]:
        return self.numeric_features + self.categorical_features


@dataclass(frozen=True)
class TrainingConfig:
    scoring: str
    cv: int
    tune: bool
    models: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class ArtifactConfig:
    model_dir: Path
    model_path: Path
    legacy_model_path: Path
    metrics_path: Path
    model_comparison_path: Path
    feature_importance_path: Path
    figures_dir: Path
    prediction_output_path: Path


@dataclass(frozen=True)
class LoggingConfig:
    log_dir: Path
    log_file: Path


@dataclass(frozen=True)
class AppConfig:
    project: ProjectConfig
    data: DataConfig
    features: FeatureConfig
    training: TrainingConfig
    artifacts: ArtifactConfig
    logging: LoggingConfig


class ConfigurationManager:
    def __init__(self, config_path: str | Path = "configs/config.yml") -> None:
        self.config_path = Path(config_path)
        self._config = self._load_yaml(self.config_path)

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise DataValidationError(f"Config file is empty or invalid: {path}")
        return config

    def get_config(self) -> AppConfig:
        cfg = self._config
        return AppConfig(
            project=ProjectConfig(**cfg["project"]),
            data=DataConfig(
                raw_dir=Path(cfg["data"]["raw_dir"]),
                processed_dir=Path(cfg["data"]["processed_dir"]),
                train_path=Path(cfg["data"]["train_path"]),
                test_path=Path(cfg["data"]["test_path"]),
                required_files=list(cfg["data"]["required_files"]),
            ),
            features=FeatureConfig(
                target=cfg["features"]["target"],
                test_start_year=int(cfg["features"]["test_start_year"]),
                numeric_features=list(cfg["features"]["numeric_features"]),
                categorical_features=list(cfg["features"]["categorical_features"]),
            ),
            training=TrainingConfig(
                scoring=cfg["training"]["scoring"],
                cv=int(cfg["training"]["cv"]),
                tune=bool(cfg["training"]["tune"]),
                models=dict(cfg["training"]["models"]),
            ),
            artifacts=ArtifactConfig(
                model_dir=Path(cfg["artifacts"]["model_dir"]),
                model_path=Path(cfg["artifacts"]["model_path"]),
                legacy_model_path=Path(cfg["artifacts"]["legacy_model_path"]),
                metrics_path=Path(cfg["artifacts"]["metrics_path"]),
                model_comparison_path=Path(cfg["artifacts"]["model_comparison_path"]),
                feature_importance_path=Path(cfg["artifacts"]["feature_importance_path"]),
                figures_dir=Path(cfg["artifacts"]["figures_dir"]),
                prediction_output_path=Path(cfg["artifacts"]["prediction_output_path"]),
            ),
            logging=LoggingConfig(
                log_dir=Path(cfg["logging"]["log_dir"]),
                log_file=Path(cfg["logging"]["log_file"]),
            ),
        )


def load_config(path: str | Path = "configs/config.yml") -> AppConfig:
    return ConfigurationManager(path).get_config()
