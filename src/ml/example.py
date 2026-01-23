"""
Example script demonstrating Bitcoin price prediction ML stack.

This script shows:
1. Training a model on synthetic data
2. Making predictions
3. Evaluating performance
4. Saving and loading models
"""

import logging
from pathlib import Path

from ml.price_predictor import BitcoinPricePredictor, PredictorConfig
from ml.features import FeatureConfig
from ml.trainer import ModelTrainer, TrainerConfig
from ml.data_loader import DataLoader
from ml.metrics import MetricsCalculator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def example_training():
    """Example: Train a model on synthetic data."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Training Model on Synthetic Data")
    print("="*60 + "\n")

    # Configure trainer
    trainer_config = TrainerConfig(
        data_source="synthetic",
        model_dir="models/example_bitcoin_predictor",
        auto_retrain=False,
    )

    # Create trainer
    trainer = ModelTrainer(trainer_config=trainer_config)

    # Train
    logger.info("Starting training...")
    metrics = trainer.train_once()

    # Print results
    print("\nTraining Results:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")

    if 'total_return' in metrics:
        print(f"\n  Total Return: {metrics['total_return']:.2%}")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")

    return trainer


def example_prediction(trainer: ModelTrainer):
    """Example: Make predictions on new data."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Making Predictions")
    print("="*60 + "\n")

    # Generate some test data
    data_loader = DataLoader()
    test_df = data_loader.generate_synthetic_data(n_samples=500)

    # Get predictor
    predictor = trainer.predictor

    # Make prediction on latest data
    result = predictor.predict_single(test_df)

    print("Prediction Result:")
    print(f"  Direction:   {result['direction']}")
    print(f"  Probability: {result['probability']:.2%}")
    print(f"  Confidence:  {result['confidence']:.2%}")
    print(f"\nIndividual Model Predictions:")
    print(f"  LightGBM: {result['individual_predictions']['lightgbm']:.4f}")
    print(f"  XGBoost:  {result['individual_predictions']['xgboost']:.4f}")
    print(f"  CatBoost: {result['individual_predictions']['catboost']:.4f}")


def example_feature_importance(trainer: ModelTrainer):
    """Example: Analyze feature importance."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Feature Importance Analysis")
    print("="*60 + "\n")

    # Get top 20 features
    top_features = trainer.get_feature_importance(top_n=20)

    print("Top 20 Most Important Features:")
    for i, (feature, importance) in enumerate(top_features.items(), 1):
        print(f"  {i:2d}. {feature:30s} {importance:.6f}")


def example_save_load():
    """Example: Save and load models."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Saving and Loading Models")
    print("="*60 + "\n")

    model_dir = "models/example_bitcoin_predictor"

    # Check if model exists
    if not Path(model_dir).exists():
        print(f"Model not found at {model_dir}")
        print("Run example_training() first!")
        return

    # Load model
    logger.info(f"Loading model from {model_dir}...")
    predictor = BitcoinPricePredictor()
    predictor.load_model(model_dir)

    # Get model info
    info = predictor.get_model_info()

    print("Loaded Model Info:")
    print(f"  Trained:           {info['is_trained']}")
    print(f"  Last Training:     {info['last_training_time']}")
    print(f"  Number of Features: {info['n_features']}")
    print(f"  Ensemble Weights:")
    print(f"    LightGBM: {info['ensemble_weights']['lightgbm']:.2f}")
    print(f"    XGBoost:  {info['ensemble_weights']['xgboost']:.2f}")
    print(f"    CatBoost: {info['ensemble_weights']['catboost']:.2f}")


def example_custom_config():
    """Example: Train with custom configuration."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Training with Custom Configuration")
    print("="*60 + "\n")

    # Custom predictor config
    predictor_config = PredictorConfig(
        prediction_horizon=5,
        lgb_weight=0.4,
        xgb_weight=0.4,
        catboost_weight=0.2,
        threshold=0.55,  # Higher threshold for more conservative predictions
    )

    # Custom feature config
    feature_config = FeatureConfig(
        rsi_periods=[9, 14, 28],
        ma_periods=[10, 20, 50, 100],
        ema_periods=[12, 26, 50],
    )

    # Custom trainer config
    trainer_config = TrainerConfig(
        data_source="synthetic",
        model_dir="models/custom_bitcoin_predictor",
        train_size=0.75,
        val_size=0.15,
        test_size=0.10,
    )

    # Create trainer
    trainer = ModelTrainer(
        trainer_config=trainer_config,
        predictor_config=predictor_config,
        feature_config=feature_config,
    )

    # Train
    logger.info("Training with custom configuration...")
    metrics = trainer.train_once()

    print("\nCustom Model Results:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")


def main():
    """Run all examples."""
    try:
        # Example 1: Training
        trainer = example_training()

        # Example 2: Predictions
        example_prediction(trainer)

        # Example 3: Feature importance
        example_feature_importance(trainer)

        # Example 4: Save/Load
        example_save_load()

        # Example 5: Custom config
        # example_custom_config()  # Uncomment to run

        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60 + "\n")

    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
