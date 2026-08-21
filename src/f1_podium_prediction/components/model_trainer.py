from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.f1_podium_prediction.configuration import AppConfig
from src.f1_podium_prediction.logger import get_logger
from src.f1_podium_prediction.utils import save_object


@dataclass
class TrainedModelResult:
    best_model_name: str
    best_score: float
    best_pipeline: Pipeline
    comparison: pd.DataFrame


class ModelTrainer:
    """Train multiple models, compare their performance, and save the best one."""

    def __init__(self, config: AppConfig, preprocessor) -> None:
        self.config = config
        self.preprocessor = preprocessor
        self.logger = get_logger(__name__)

    def _candidate_models(self) -> dict[str, tuple[Any, dict[str, list[Any]]]]:
        """Return the models and hyperparameters to evaluate."""

        random_state = self.config.project.random_state

        models = {
            "logistic_regression": (
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
                {
                    "model__C": [0.2, 1.0, 3.0],
                },
            ),
            "random_forest": (
                RandomForestClassifier(
                    n_estimators=350,
                    min_samples_leaf=4,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=random_state,
                ),
                {
                    "model__max_depth": [8, 14, None],
                    "model__min_samples_leaf": [3, 6],
                },
            ),
            "gradient_boosting": (
                GradientBoostingClassifier(
                    random_state=random_state,
                ),
                {
                    "model__learning_rate": [0.04, 0.08],
                    "model__max_depth": [2, 3],
                },
            ),
            "xgboost": (
                XGBClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=3,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=random_state,
                    n_jobs=-1,
                ),
                {
                    "model__n_estimators": [200, 300],
                    "model__max_depth": [3, 4],
                    "model__learning_rate": [0.05, 0.1],
                },
            ),
        }

        return models

    def train_and_select(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
    ) -> TrainedModelResult:
        """Train all enabled models, compare them, and select the best model."""

        feature_config = self.config.features

        X_train = train[feature_config.model_features]
        y_train = train[feature_config.target]

        X_test = test[feature_config.model_features]
        y_test = test[feature_config.target]

        results = []
        trained_models = {}

        for model_name, (estimator, param_grid) in self._candidate_models().items():

            # Check whether this model is enabled in the configuration
            model_config = self.config.training.models.get(model_name, {})

            if not model_config.get("enabled", True):
                self.logger.info("Skipping disabled model: %s", model_name)
                continue

            self.logger.info("Starting training for model: %s", model_name)

            pipeline = Pipeline(
                steps=[
                    ("preprocessor", self.preprocessor),
                    ("model", estimator),
                ]
            )

            # Perform hyperparameter tuning if enabled
            if self.config.training.tune:
                self.logger.info(
                    "Performing hyperparameter tuning for: %s",
                    model_name,
                )

                grid_search = GridSearchCV(
                    estimator=pipeline,
                    param_grid=param_grid,
                    scoring=self.config.training.scoring,
                    cv=self.config.training.cv,
                    n_jobs=-1,
                    verbose=1,
                )

                grid_search.fit(X_train, y_train)

                model = grid_search.best_estimator_

                self.logger.info(
                    "Best parameters for %s: %s",
                    model_name,
                    grid_search.best_params_,
                )

            else:
                self.logger.info(
                    "Training %s without hyperparameter tuning",
                    model_name,
                )

                model = pipeline.fit(X_train, y_train)

            # Get probability predictions for the positive class
            y_prob = model.predict_proba(X_test)[:, 1]

            # Convert probabilities into class predictions
            y_pred = (y_prob >= 0.5).astype(int)

            # Calculate evaluation metrics
            roc_auc = roc_auc_score(y_test, y_prob)
            avg_precision = average_precision_score(y_test, y_prob)

            precision = precision_score(
                y_test,
                y_pred,
                zero_division=0,
            )

            recall = recall_score(
                y_test,
                y_pred,
                zero_division=0,
            )

            f1 = f1_score(
                y_test,
                y_pred,
                zero_division=0,
            )

            # Store model performance
            results.append(
                {
                    "model": model_name,
                    "roc_auc": roc_auc,
                    "average_precision": avg_precision,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )

            trained_models[model_name] = model

            self.logger.info(
                "%s | ROC-AUC: %.4f | AP: %.4f | Precision: %.4f | "
                "Recall: %.4f | F1: %.4f",
                model_name,
                roc_auc,
                avg_precision,
                precision,
                recall,
                f1,
            )

        if not results:
            raise ValueError(
                "No models were trained. Please check the model configuration."
            )

        # Create a comparison table and rank models by Average Precision
        comparison = pd.DataFrame(results)

        comparison = (
                comparison
                .sort_values(
                    by="f1",
                    ascending=False,
                )
                .reset_index(drop=True)
            )

        # Select the best-performing model
        best_model_name = comparison.loc[0, "model"]
        best_model = trained_models[best_model_name]
        best_score = float(comparison.loc[0,"f1",])

        self.logger.info("Best Model: %s", best_model_name)
        self.logger.info("Best F1 Score: %.4f", best_score)

        # Create the model artifact
        artifact = {
            "model": best_model,
            "features": feature_config.model_features,
            "target": feature_config.target,
            "best_model_name": best_model_name,
            "comparison": comparison.to_dict(orient="records"),
        }

        # Save the model in both current and legacy locations
        save_object(artifact, self.config.artifacts.model_path)
        save_object(artifact, self.config.artifacts.legacy_model_path)

        # Save model comparison results
        comparison_path = self.config.artifacts.model_comparison_path
        comparison_path.parent.mkdir(parents=True, exist_ok=True)

        comparison.to_csv(comparison_path, index=False)

        self.logger.info(
            "Best model saved to: %s",
            self.config.artifacts.model_path,
        )

        self.logger.info(
            "Model comparison saved to: %s",
            comparison_path,
        )

        return TrainedModelResult(
            best_model_name=best_model_name,
            best_score=best_score,
            best_pipeline=best_model,
            comparison=comparison,
        )