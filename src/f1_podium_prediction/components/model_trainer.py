from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
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

    def __init__(
        self,
        config: AppConfig,
        preprocessor,
    ) -> None:

        self.config = config
        self.preprocessor = preprocessor
        self.logger = get_logger(__name__)

    # =========================================================
    # CANDIDATE MODELS
    # =========================================================

    def _candidate_models(
        self,
    ) -> dict[str, tuple[Any, dict[str, list[Any]]]]:

        random_state = self.config.project.random_state

        return {

            "logistic_regression": (

                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=random_state,
                ),

                {
                    "model__C": [
                        0.2,
                        1.0,
                        3.0,
                    ]
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
                    "model__max_depth": [
                        8,
                        14,
                        None,
                    ],

                    "model__min_samples_leaf": [
                        3,
                        6,
                    ],
                },
            ),

            "gradient_boosting": (

                GradientBoostingClassifier(
                    random_state=random_state,
                ),

                {
                    "model__learning_rate": [
                        0.04,
                        0.08,
                    ],

                    "model__max_depth": [
                        2,
                        3,
                    ],
                },
            ),
        }

    # =========================================================
    # TRAIN + SELECT BEST MODEL
    # =========================================================

    def train_and_select(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
    ) -> TrainedModelResult:

        feature_cfg = self.config.features

        # -----------------------------------------------------
        # Split features and target
        # -----------------------------------------------------

        X_train = train[
            feature_cfg.model_features
        ]

        y_train = train[
            feature_cfg.target
        ]

        X_test = test[
            feature_cfg.model_features
        ]

        y_test = test[
            feature_cfg.target
        ]

        rows = []

        trained_models = {}

        # -----------------------------------------------------
        # Train each candidate model
        # -----------------------------------------------------

        for (
            model_name,
            (estimator, param_grid),
        ) in self._candidate_models().items():

            # Check config
            model_config = self.config.training.models.get(
                model_name,
                {},
            )

            if not model_config.get(
                "enabled",
                True,
            ):
                self.logger.info(
                    "Skipping disabled model: %s",
                    model_name,
                )

                continue

            self.logger.info(
                "Starting model: %s",
                model_name,
            )

            # -------------------------------------------------
            # Create pipeline
            # -------------------------------------------------

            pipeline = Pipeline(
                steps=[
                    (
                        "preprocessor",
                        self.preprocessor,
                    ),
                    (
                        "model",
                        estimator,
                    ),
                ]
            )

            # -------------------------------------------------
            # Hyperparameter tuning
            # -------------------------------------------------

            if self.config.training.tune:

                self.logger.info(
                    "Hyperparameter tuning: %s",
                    model_name,
                )

                search = GridSearchCV(
                    estimator=pipeline,
                    param_grid=param_grid,
                    scoring=self.config.training.scoring,
                    cv=self.config.training.cv,
                    n_jobs=-1,
                    verbose=1,
                )

                search.fit(
                    X_train,
                    y_train,
                )

                model = search.best_estimator_

                self.logger.info(
                    "Best parameters for %s: %s",
                    model_name,
                    search.best_params_,
                )

            # -------------------------------------------------
            # Normal training
            # -------------------------------------------------

            else:

                self.logger.info(
                    "Training without tuning: %s",
                    model_name,
                )

                model = pipeline.fit(
                    X_train,
                    y_train,
                )

            # -------------------------------------------------
            # Predictions
            # -------------------------------------------------

            y_prob = model.predict_proba(
                X_test
            )[:, 1]

            y_pred = (
                y_prob >= 0.5
            ).astype(int)

            # -------------------------------------------------
            # Metrics
            # -------------------------------------------------

            roc_auc = roc_auc_score(
                y_test,
                y_prob,
            )

            avg_precision = average_precision_score(
                y_test,
                y_prob,
            )

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

            # -------------------------------------------------
            # Store results
            # -------------------------------------------------

            rows.append(
                {
                    "model": model_name,
                    "roc_auc": roc_auc,
                    "average_precision": avg_precision,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )

            trained_models[
                model_name
            ] = model

            self.logger.info(
                "%s | ROC-AUC=%.4f | AP=%.4f | Precision=%.4f | Recall=%.4f | F1=%.4f",
                model_name,
                roc_auc,
                avg_precision,
                precision,
                recall,
                f1,
            )

        # =====================================================
        # CHECK TRAINING RESULTS
        # =====================================================

        if not rows:

            raise ValueError(
                "No models were trained. "
                "Check training.models in config.yml."
            )

        # -----------------------------------------------------
        # Create comparison DataFrame
        # -----------------------------------------------------

        comparison = pd.DataFrame(
            rows
        )

        required_columns = [
            "model",
            "roc_auc",
            "average_precision",
            "precision",
            "recall",
            "f1",
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in comparison.columns
        ]

        if missing_columns:

            raise ValueError(
                f"Missing metric columns: {missing_columns}"
            )

        # =====================================================
        # SELECT BEST MODEL
        # =====================================================

        comparison = (
            comparison
            .sort_values(
                by="average_precision",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

        best_model_name = comparison.loc[
            0,
            "model",
        ]

        best_model = trained_models[
            best_model_name
        ]

        best_score = float(
            comparison.loc[
                0,
                "average_precision",
            ]
        )

        self.logger.info(
            "============================================"
        )

        self.logger.info(
            "Best Model: %s",
            best_model_name,
        )

        self.logger.info(
            "Best Average Precision: %.4f",
            best_score,
        )

        self.logger.info(
            "============================================"
        )

        # =====================================================
        # SAVE BEST MODEL
        # =====================================================

        artifact = {
            "model": best_model,
            "features": feature_cfg.model_features,
            "target": feature_cfg.target,
            "best_model_name": best_model_name,
            "comparison": comparison.to_dict(
                orient="records"
            ),
        }

        save_object(
            artifact,
            self.config.artifacts.model_path,
        )

        save_object(
            artifact,
            self.config.artifacts.legacy_model_path,
        )

        self.logger.info(
            "Best model saved to: %s",
            self.config.artifacts.model_path,
        )

        # =====================================================
        # SAVE MODEL COMPARISON
        # =====================================================

        self.config.artifacts.model_comparison_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        comparison.to_csv(
            self.config.artifacts.model_comparison_path,
            index=False,
        )

        self.logger.info(
            "Model comparison saved to: %s",
            self.config.artifacts.model_comparison_path,
        )

        return TrainedModelResult(
            best_model_name=best_model_name,
            best_score=best_score,
            best_pipeline=best_model,
            comparison=comparison,
        )


# =============================================================
# TESTING
# =============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("MODEL TRAINER TEST")
    print("=" * 60)

    try:

        from src.f1_podium_prediction.configuration import (
            load_config,
        )

        from src.f1_podium_prediction.components.data_transformation import (
            DataTransformer,
        )

        # -----------------------------------------------------
        # Load configuration
        # -----------------------------------------------------

        config = load_config()

        print("\nConfiguration loaded successfully.")

        # -----------------------------------------------------
        # Check train/test files
        # -----------------------------------------------------

        if not config.data.train_path.exists():

            raise FileNotFoundError(
                f"Train file not found: "
                f"{config.data.train_path}"
            )

        if not config.data.test_path.exists():

            raise FileNotFoundError(
                f"Test file not found: "
                f"{config.data.test_path}"
            )

        # -----------------------------------------------------
        # Load train/test
        # -----------------------------------------------------

        train = pd.read_csv(
            config.data.train_path
        )

        test = pd.read_csv(
            config.data.test_path
        )

        print(
            f"\nTrain Shape: {train.shape}"
        )

        print(
            f"Test Shape : {test.shape}"
        )

        # -----------------------------------------------------
        # Create preprocessor
        # -----------------------------------------------------

        transformer = DataTransformer(
            config.data,
            config.features,
        )

        preprocessor = (
            transformer.get_preprocessor()
        )

        # -----------------------------------------------------
        # Create trainer
        # -----------------------------------------------------

        trainer = ModelTrainer(
            config,
            preprocessor,
        )

        # -----------------------------------------------------
        # Train models
        # -----------------------------------------------------

        result = trainer.train_and_select(
            train,
            test,
        )

        # -----------------------------------------------------
        # Display result
        # -----------------------------------------------------

        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)

        print(
            result.comparison.to_string(
                index=False
            )
        )

        print(
            "\nBest Model :",
            result.best_model_name,
        )

        print(
            "Best Score :",
            round(
                result.best_score,
                4,
            ),
        )

        print(
            "\nModel Training Successful!"
        )

    except Exception as e:

        print(
            "\nModel Training Failed!"
        )

        print(
            f"{type(e).__name__}: {e}"
        )