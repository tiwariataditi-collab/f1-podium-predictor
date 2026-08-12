from __future__ import annotations

import argparse

from src.f1_podium_prediction.components.data_ingestion import DataIngestion
from src.f1_podium_prediction.components.data_transformation import DataTransformer
from src.f1_podium_prediction.components.data_validation import DataValidator
from src.f1_podium_prediction.components.feature_engineering import FeatureEngineer
from src.f1_podium_prediction.components.model_evaluation import ModelEvaluator
from src.f1_podium_prediction.components.model_trainer import ModelTrainer
from src.f1_podium_prediction.configuration import load_config
from src.f1_podium_prediction.logger import get_logger


def run_training(
    config_path: str = "configs/config.yml",):

     # 1. Load Configuration
    
    config = load_config(config_path)

    logger = get_logger(__name__,
        config.logging.log_file,
    )

    logger.info("STARTING F1 PODIUM TRAINING PIPELINE")
  

   # 2. Validate Raw Data
    
    logger.info("Step 1: Validating raw datasets...")

    validator = DataValidator(config.data,
        config.features,
    )

    validator.validate_raw_files()

    logger.info("Raw data validation successful.")

    # 3. Data Ingestion
    
    logger.info("Step 2: Loading raw datasets...")

    ingestion = DataIngestion(config.data)

    tables = ingestion.load_raw_tables()

    logger.info("Raw datasets loaded successfully.")

    # 4. Feature Engineering

    logger.info("Step 3: Building feature dataset...")

    engineer = FeatureEngineer(config.features)

    feature_frame = engineer.build_model_frame(tables)

    logger.info("Feature dataset shape: %s",
        feature_frame.shape,
    )

    # 5. Train-Test Split
    
    logger.info("Step 4: Creating train/test datasets...")

    transformer = DataTransformer(config.data,
        config.features,
    )

    train, test = transformer.split_and_save(feature_frame)

    logger.info("Train shape: %s",
        train.shape,
    )

    logger.info("Test shape: %s",
        test.shape,
    )

    # 6. Create Preprocessor

    logger.info("Step 5: Creating preprocessing pipeline...")

    preprocessor = transformer.get_preprocessor()

    # 7. Train Models
    
    logger.info("Step 6: Training and selecting best model...")

    trainer = ModelTrainer(config,
        preprocessor,
    )

    result = trainer.train_and_select(train,test,)

    logger.info("Best model: %s",
        result.best_model_name,
    )

    logger.info("Best Average Precision: %.4f",
        result.best_score,
    )

    # 8. Model Evaluation
   
    logger.info("Step 7: Evaluating best model...")

    evaluator = ModelEvaluator(config)

    metrics = evaluator.evaluate(result.best_pipeline,
        test,
    )

    logger.info("ROC-AUC: %.4f",
        metrics["roc_auc"],)

    logger.info("Average Precision: %.4f",
        metrics["average_precision"],
    )

   
    # 9. Final Output
    
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY")

    print(f"\nBest Model           : "
        f"{result.best_model_name}"
    )

    print(f"Best Average Precision : "
        f"{result.best_score:.4f}"
    )

    print(f"ROC-AUC              : "
        f"{metrics['roc_auc']:.4f}"
    )

    print(f"Average Precision    : "
        f"{metrics['average_precision']:.4f}"
    )

    print(f"Model Saved At       : "
        f"{config.artifacts.model_path}"
    )

    print(f"Metrics Saved At     : "
        f"{config.artifacts.metrics_path}"
    )

    print( f"Comparison Saved At  : "
        f"{config.artifacts.model_comparison_path}"
    )

    


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Train F1 podium prediction models."
    )

    parser.add_argument(
        "--config",
        default="configs/config.yml",
        help="Path to configuration file.",
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    run_training(
        args.config
    )