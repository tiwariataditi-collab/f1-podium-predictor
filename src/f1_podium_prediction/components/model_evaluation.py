from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

from src.f1_podium_prediction.configuration import AppConfig, load_config
from src.f1_podium_prediction.logger import get_logger
from src.f1_podium_prediction.utils import load_object, save_json


class ModelEvaluator:

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.logger = get_logger(__name__)

    # =========================================================
    # MAIN EVALUATION
    # =========================================================

    def evaluate(
        self,
        model,
        test: pd.DataFrame,
    ) -> dict:

        self.logger.info(
            "Starting model evaluation..."
        )

        x_test = test[
            self.config.features.model_features
        ]

        y_test = test[
            self.config.features.target
        ]

        # -----------------------------------------------------
        # Predictions
        # -----------------------------------------------------

        proba = model.predict_proba(
            x_test
        )[:, 1]

        pred = (
            proba >= 0.5
        ).astype(int)

        # -----------------------------------------------------
        # Metrics
        # -----------------------------------------------------

        metrics = {
            "roc_auc": float(
                roc_auc_score(
                    y_test,
                    proba,
                )
            ),

            "average_precision": float(
                average_precision_score(
                    y_test,
                    proba,
                )
            ),

            "classification_report": classification_report(
                y_test,
                pred,
                output_dict=True,
            ),

            "confusion_matrix": (
                confusion_matrix(
                    y_test,
                    pred,
                ).tolist()
            ),
        }

        # -----------------------------------------------------
        # Save metrics
        # -----------------------------------------------------

        self.config.artifacts.metrics_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        save_json(
            metrics,
            self.config.artifacts.metrics_path,
        )

        self.logger.info(
            "Metrics saved to: %s",
            self.config.artifacts.metrics_path,
        )

        # -----------------------------------------------------
        # Precision-Recall curve
        # -----------------------------------------------------

        self._save_pr_curve(
            y_test,
            proba,
        )

        # -----------------------------------------------------
        # Feature importance
        # -----------------------------------------------------

        self._save_feature_importance(
            model,
            x_test,
            y_test,
        )

        # -----------------------------------------------------
        # SHAP
        # -----------------------------------------------------

        self._save_shap_summary_if_possible(
            model,
            x_test,
        )

        self.logger.info(
            "Model evaluation completed successfully."
        )

        return metrics

    # =========================================================
    # PRECISION-RECALL CURVE
    # =========================================================

    def _save_pr_curve(
        self,
        y_true,
        proba,
    ) -> None:

        self.config.artifacts.figures_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        precision, recall, _ = precision_recall_curve(
            y_true,
            proba,
        )

        plt.figure(
            figsize=(7, 5)
        )

        plt.plot(
            recall,
            precision,
            linewidth=2.5,
        )

        plt.xlabel(
            "Recall"
        )

        plt.ylabel(
            "Precision"
        )

        plt.title(
            "Podium Prediction Precision-Recall Curve"
        )

        plt.tight_layout()

        output_path = (
            self.config.artifacts.figures_dir
            / "precision_recall_curve.png"
        )

        plt.savefig(
            output_path,
            dpi=160,
        )

        plt.close()

        self.logger.info(
            "PR curve saved to: %s",
            output_path,
        )

    # =========================================================
    # FEATURE IMPORTANCE
    # =========================================================

    def _save_feature_importance(
        self,
        model,
        x_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> None:

        try:

            self.logger.info(
                "Calculating permutation feature importance..."
            )

            result = permutation_importance(
                model,
                x_test,
                y_test,
                n_repeats=3,
                random_state=self.config.project.random_state,
                scoring=self.config.training.scoring,
                n_jobs=-1,
            )

            importance = pd.DataFrame(
                {
                    "feature": x_test.columns,
                    "importance_mean": result.importances_mean,
                    "importance_std": result.importances_std,
                }
            ).sort_values(
                "importance_mean",
                ascending=False,
            )

            self.config.artifacts.feature_importance_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            importance.to_csv(
                self.config.artifacts.feature_importance_path,
                index=False,
            )

            self.logger.info(
                "Feature importance saved to: %s",
                self.config.artifacts.feature_importance_path,
            )

        except Exception as exc:

            self.logger.warning(
                "Feature importance skipped: %s",
                exc,
            )

    # =========================================================
    # SHAP
    # =========================================================

    def _save_shap_summary_if_possible(
        self,
        model,
        x_test: pd.DataFrame,
    ) -> None:

        try:

            import shap

            estimator = model.named_steps[
                "model"
            ]

            # SHAP only for tree-based models
            if not hasattr(
                estimator,
                "feature_importances_",
            ):
                self.logger.info(
                    "SHAP skipped: model is not tree-based."
                )
                return

            sample = x_test.sample(
                min(200, len(x_test)),
                random_state=self.config.project.random_state,
            )

            preprocessor = model.named_steps[
                "preprocessor"
            ]

            transformed = preprocessor.transform(
                sample
            )

            explainer = shap.TreeExplainer(
                estimator
            )

            values = explainer.shap_values(
                transformed
            )

            if isinstance(values, list):

                shap_values = values[1]

            else:

                shap_values = values

            shap.summary_plot(
                shap_values,
                transformed,
                show=False,
                max_display=15,
            )

            self.config.artifacts.figures_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            output_path = (
                self.config.artifacts.figures_dir
                / "shap_summary.png"
            )

            plt.tight_layout()

            plt.savefig(
                output_path,
                dpi=160,
                bbox_inches="tight",
            )

            plt.close()

            self.logger.info(
                "SHAP summary saved to: %s",
                output_path,
            )

        except ImportError:

            self.logger.warning(
                "SHAP is not installed. Skipping SHAP analysis."
            )

        except Exception as exc:

            self.logger.warning(
                "SHAP skipped: %s",
                exc,
            )


# =============================================================
# TESTING
# =============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("MODEL EVALUATION TEST")
    print("=" * 60)

    try:

        # -----------------------------------------------------
        # Load configuration
        # -----------------------------------------------------

        config = load_config()

        print(
            "\nConfiguration loaded successfully."
        )

        # -----------------------------------------------------
        # Check files
        # -----------------------------------------------------

        if not config.data.test_path.exists():

            raise FileNotFoundError(
                f"Test file not found: "
                f"{config.data.test_path}"
            )

        if not config.artifacts.model_path.exists():

            raise FileNotFoundError(
                f"Trained model not found: "
                f"{config.artifacts.model_path}"
            )

        # -----------------------------------------------------
        # Load test data
        # -----------------------------------------------------

        test = pd.read_csv(
            config.data.test_path
        )

        print(
            f"Test Shape: {test.shape}"
        )

        # -----------------------------------------------------
        # Load trained model
        # -----------------------------------------------------

        artifact = load_object(
            config.artifacts.model_path
        )

        model = artifact["model"]

        print(
            "Best Model:",
            artifact["best_model_name"],
        )

        # -----------------------------------------------------
        # Evaluate
        # -----------------------------------------------------

        evaluator = ModelEvaluator(
            config
        )

        metrics = evaluator.evaluate(
            model,
            test,
        )

        # -----------------------------------------------------
        # Display important metrics
        # -----------------------------------------------------

        print(
            "\n" + "=" * 60
        )

        print(
            "EVALUATION RESULTS"
        )

        print(
            "=" * 60
        )

        print(
            "ROC-AUC :",
            round(
                metrics["roc_auc"],
                4,
            ),
        )

        print(
            "Average Precision :",
            round(
                metrics["average_precision"],
                4,
            ),
        )

        print(
            "\nConfusion Matrix:"
        )

        print(
            metrics["confusion_matrix"]
        )

        print(
            "\nModel Evaluation Successful!"
        )

    except Exception as e:

        print(
            "\nModel Evaluation Failed!"
        )

        print(
            f"{type(e).__name__}: {e}"
        )