"""
NYC Delivery Promise Engine - ETA Model Training Module

This module trains Random Forest regression models for predicting trip duration (ETA).
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import logging
from typing import Tuple, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ETAModel:
    """ETA prediction model using Random Forest regression."""
    
    def __init__(self, **kwargs):
        """
        Initialize ETA model.
        
        Args:
            **kwargs: Parameters for RandomForestRegressor
        """
        default_params = {
            'n_estimators': 100,
            'max_depth': 15,
            'min_samples_split': 100,
            'random_state': 42,
            'n_jobs': -1
        }
        default_params.update(kwargs)
        
        self.model = RandomForestRegressor(**default_params)
        self.feature_cols = None
        self.is_trained = False
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare feature matrix for modeling.
        
        Args:
            df: DataFrame with engineered features
            
        Returns:
            Feature matrix
        """
        # Define feature columns for modeling
        numeric_features = [
            'trip_distance', 'pickup_hour', 'pickup_dow',
            'tavg', 'prcp', 'wspd'
        ]
        
        categorical_features = [
            'pickup_borough_encoded', 'dropoff_borough_encoded',
            'weather_severity_encoded'
        ]
        
        boolean_features = [
            'is_weekend', 'is_rush_hour', 'is_wet', 'is_holiday',
            'same_borough', 'airport_trip'
        ]
        
        # Combine all features
        self.feature_cols = numeric_features + categorical_features + boolean_features
        
        # Filter to available columns
        available_cols = [col for col in self.feature_cols if col in df.columns]
        self.feature_cols = available_cols
        
        return df[self.feature_cols].copy()
    
    def train(self, df_train: pd.DataFrame, target_col: str = 'trip_duration_minutes') -> Dict[str, Any]:
        """
        Train the ETA model.
        
        Args:
            df_train: Training DataFrame
            target_col: Target column name
            
        Returns:
            Training metrics
        """
        logger.info("🚀 Training ETA regression model...")
        
        # Prepare features and target
        X_train = self.prepare_features(df_train)
        y_train = df_train[target_col].copy()
        
        logger.info(f"📊 Training data: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
        logger.info(f"🎯 Target range: {y_train.min():.1f} to {y_train.max():.1f} minutes")
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Calculate training metrics
        y_pred_train = self.model.predict(X_train)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        
        logger.info(f"✅ Model training completed!")
        logger.info(f"📈 Training MAE: {train_mae:.2f} minutes")
        logger.info(f"📈 Training RMSE: {train_rmse:.2f} minutes")
        
        return {
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'n_features': len(self.feature_cols),
            'n_samples': len(X_train)
        }
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make ETA predictions.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Predicted trip durations
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X = self.prepare_features(df)
        return self.model.predict(X)
    
    def evaluate(self, df_test: pd.DataFrame, target_col: str = 'trip_duration_minutes') -> Dict[str, Any]:
        """
        Evaluate model performance on test data.
        
        Args:
            df_test: Test DataFrame
            target_col: Target column name
            
        Returns:
            Evaluation metrics
        """
        logger.info("📊 Evaluating ETA model...")
        
        # Make predictions
        y_test = df_test[target_col].copy()
        y_pred = self.predict(df_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # P90 coverage analysis
        residuals = y_test - y_pred
        p90_adjustment = np.percentile(residuals, 90)
        y_pred_p90 = y_pred + p90_adjustment
        p90_coverage = (y_test <= y_pred_p90).mean()
        
        # Prediction quality
        pred_mean = np.mean(y_pred)
        actual_mean = np.mean(y_test)
        pred_median = np.median(y_pred)
        actual_median = np.median(y_test)
        
        metrics = {
            'test_mae': mae,
            'test_rmse': rmse,
            'p90_coverage': p90_coverage,
            'p90_adjustment': p90_adjustment,
            'pred_mean': pred_mean,
            'actual_mean': actual_mean,
            'pred_median': pred_median,
            'actual_median': actual_median
        }
        
        logger.info(f"🎯 Test Results:")
        logger.info(f"   MAE: {mae:.2f} minutes")
        logger.info(f"   RMSE: {rmse:.2f} minutes")
        logger.info(f"   P90 Coverage: {p90_coverage:.1%}")
        
        return metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance rankings.
        
        Returns:
            DataFrame with feature importance
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, filepath: str):
        """
        Save trained model to file.
        
        Args:
            filepath: Path to save model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'feature_cols': self.feature_cols,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"💾 Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str):
        """
        Load trained model from file.
        
        Args:
            filepath: Path to model file
            
        Returns:
            Loaded ETAModel instance
        """
        model_data = joblib.load(filepath)
        
        eta_model = cls()
        eta_model.model = model_data['model']
        eta_model.feature_cols = model_data['feature_cols']
        eta_model.is_trained = model_data['is_trained']
        
        logger.info(f"📂 Model loaded from {filepath}")
        return eta_model


def train_eta_model(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    model_params: Dict = None,
    save_path: str = None
) -> Tuple[ETAModel, Dict]:
    """
    Complete ETA model training pipeline.
    
    Args:
        df_train: Training data
        df_test: Test data
        model_params: Model hyperparameters
        save_path: Path to save trained model
        
    Returns:
        Tuple of (trained_model, metrics)
    """
    # Initialize model
    params = model_params or {}
    eta_model = ETAModel(**params)
    
    # Train model
    train_metrics = eta_model.train(df_train)
    
    # Evaluate model
    test_metrics = eta_model.evaluate(df_test)
    
    # Combine metrics
    all_metrics = {**train_metrics, **test_metrics}
    
    # Feature importance
    importance_df = eta_model.get_feature_importance()
    logger.info(f"\n🔍 Top 5 Important Features:")
    for _, row in importance_df.head().iterrows():
        logger.info(f"   {row['feature']:20s}: {row['importance']:.3f}")
    
    # Save model if path provided
    if save_path:
        eta_model.save_model(save_path)
    
    return eta_model, all_metrics


if __name__ == "__main__":
    # Example usage for testing
    import os
    import sys
    
    # Add parent directory to path for imports
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from build_features import create_ml_features
    from data_download import load_taxi_data, load_supporting_data
    
    logger.info("🧪 Testing ETA model training...")
    
    try:
        # Load sample data
        data_dir = "../DATA"
        df_train = load_taxi_data(2025, [5, 6], data_dir).sample(10000)  # Sample for testing
        df_test = load_taxi_data(2025, [7], data_dir).sample(5000)
        weather, zones, holidays = load_supporting_data(data_dir)
        
        # Feature engineering
        df_train, df_test, _ = create_ml_features(df_train, df_test, weather, zones, holidays)
        
        # Train model
        model, metrics = train_eta_model(df_train, df_test)
        
        logger.info(f"✅ ETA model training test completed!")
        logger.info(f"📊 Final MAE: {metrics['test_mae']:.2f} minutes")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
