"""
NYC Delivery Promise Engine - Model Evaluation Module

This module provides comprehensive evaluation metrics and analysis tools
for delivery promise optimization models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, roc_auc_score, precision_recall_curve
import json
from datetime import datetime
import logging
from typing import Dict, Any, Tuple, List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation for delivery promise optimization."""
    
    def __init__(self):
        self.eta_metrics = {}
        self.delay_metrics = {}
        self.business_metrics = {}
    
    def evaluate_eta_model(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        dataset_name: str = "test"
    ) -> Dict[str, Any]:
        """
        Evaluate ETA regression model performance.
        
        Args:
            y_true: Actual trip durations
            y_pred: Predicted trip durations
            dataset_name: Name of dataset being evaluated
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"📊 Evaluating ETA model on {dataset_name} data...")
        
        # Basic regression metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # Residual analysis
        residuals = y_true - y_pred
        residual_std = np.std(residuals)
        residual_mean = np.mean(residuals)
        
        # P90 coverage (important for delivery promises)
        p90_adjustment = np.percentile(residuals, 90)
        y_pred_p90 = y_pred + p90_adjustment
        p90_coverage = np.mean(y_true <= y_pred_p90)
        
        # Percentile accuracy
        actual_p50 = np.percentile(y_true, 50)
        actual_p90 = np.percentile(y_true, 90)
        pred_p50 = np.percentile(y_pred, 50)
        pred_p90 = np.percentile(y_pred, 90)
        
        metrics = {
            f'{dataset_name}_mae': mae,
            f'{dataset_name}_rmse': rmse,
            f'{dataset_name}_mape': mape,
            f'{dataset_name}_residual_mean': residual_mean,
            f'{dataset_name}_residual_std': residual_std,
            f'{dataset_name}_p90_coverage': p90_coverage,
            f'{dataset_name}_p90_adjustment': p90_adjustment,
            f'{dataset_name}_actual_p50': actual_p50,
            f'{dataset_name}_actual_p90': actual_p90,
            f'{dataset_name}_pred_p50': pred_p50,
            f'{dataset_name}_pred_p90': pred_p90,
            f'{dataset_name}_n_samples': len(y_true)
        }
        
        self.eta_metrics.update(metrics)
        
        logger.info(f"✅ ETA Evaluation Results ({dataset_name}):")
        logger.info(f"   MAE: {mae:.2f} minutes")
        logger.info(f"   RMSE: {rmse:.2f} minutes")
        logger.info(f"   P90 Coverage: {p90_coverage:.1%}")
        
        return metrics
    
    def evaluate_delay_model(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        dataset_name: str = "test"
    ) -> Dict[str, Any]:
        """
        Evaluate delay classification model performance.
        
        Args:
            y_true: True delay labels (0/1)
            y_pred_proba: Predicted delay probabilities
            dataset_name: Name of dataset being evaluated
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"🎯 Evaluating delay model on {dataset_name} data...")
        
        # ROC-AUC
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        # Precision-Recall analysis
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        # Find optimal threshold
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_precision = precision[optimal_idx]
        optimal_recall = recall[optimal_idx]
        optimal_f1 = f1_scores[optimal_idx]
        
        # Class balance
        positive_rate = np.mean(y_true)
        
        metrics = {
            f'{dataset_name}_roc_auc': roc_auc,
            f'{dataset_name}_optimal_threshold': optimal_threshold,
            f'{dataset_name}_optimal_precision': optimal_precision,
            f'{dataset_name}_optimal_recall': optimal_recall,
            f'{dataset_name}_optimal_f1': optimal_f1,
            f'{dataset_name}_positive_rate': positive_rate,
            f'{dataset_name}_n_samples': len(y_true)
        }
        
        self.delay_metrics.update(metrics)
        
        logger.info(f"✅ Delay Evaluation Results ({dataset_name}):")
        logger.info(f"   ROC-AUC: {roc_auc:.3f}")
        logger.info(f"   Optimal F1: {optimal_f1:.3f}")
        logger.info(f"   Positive Rate: {positive_rate:.1%}")
        
        return metrics
    
    def analyze_promise_strategy(
        self,
        trip_durations: np.ndarray,
        strategy_name: str = "current"
    ) -> Dict[str, Any]:
        """
        Analyze delivery promise strategy performance.
        
        Args:
            trip_durations: Array of trip durations
            strategy_name: Name of strategy being analyzed
            
        Returns:
            Dictionary of business metrics
        """
        logger.info(f"📈 Analyzing {strategy_name} promise strategy...")
        
        # Key percentiles for promise strategies
        p50 = np.percentile(trip_durations, 50)
        p90 = np.percentile(trip_durations, 90)
        p95 = np.percentile(trip_durations, 95)
        mean_duration = np.mean(trip_durations)
        
        # P50 vs P90 strategy comparison
        p50_late_rate = np.mean(trip_durations > p50)
        p90_late_rate = np.mean(trip_durations > p90)
        
        # Buffer analysis
        p50_buffer = 0  # By definition
        p90_buffer = p90 - p50
        
        # Customer satisfaction proxy
        p50_customer_wait = p50
        p90_customer_wait = p90
        wait_time_increase = p90_customer_wait - p50_customer_wait
        
        metrics = {
            f'{strategy_name}_p50_minutes': p50,
            f'{strategy_name}_p90_minutes': p90,
            f'{strategy_name}_p95_minutes': p95,
            f'{strategy_name}_mean_minutes': mean_duration,
            f'{strategy_name}_p50_late_rate': p50_late_rate,
            f'{strategy_name}_p90_late_rate': p90_late_rate,
            f'{strategy_name}_p50_buffer': p50_buffer,
            f'{strategy_name}_p90_buffer': p90_buffer,
            f'{strategy_name}_wait_time_increase': wait_time_increase,
            f'{strategy_name}_late_rate_improvement': p50_late_rate - p90_late_rate,
            f'{strategy_name}_n_trips': len(trip_durations)
        }
        
        self.business_metrics.update(metrics)
        
        logger.info(f"✅ Promise Strategy Analysis ({strategy_name}):")
        logger.info(f"   P50 Strategy: {p50:.1f} min promise, {p50_late_rate:.1%} late")
        logger.info(f"   P90 Strategy: {p90:.1f} min promise, {p90_late_rate:.1%} late")
        logger.info(f"   Trade-off: +{wait_time_increase:.1f} min wait for {(p50_late_rate-p90_late_rate)*100:.0f}pp fewer late deliveries")
        
        return metrics
    
    def borough_analysis(
        self,
        df: pd.DataFrame,
        duration_col: str = 'trip_duration_minutes'
    ) -> Dict[str, Any]:
        """
        Analyze performance by borough.
        
        Args:
            df: DataFrame with borough and duration data
            duration_col: Name of duration column
            
        Returns:
            Borough performance metrics
        """
        logger.info("🏙️ Analyzing borough performance...")
        
        borough_stats = df.groupby('pickup_borough')[duration_col].agg([
            'count', 'mean', 'median',
            lambda x: x.quantile(0.9),
            lambda x: x.quantile(0.95)
        ]).round(1)
        
        borough_stats.columns = ['trips', 'avg_duration', 'median_duration', 'p90_duration', 'p95_duration']
        borough_stats = borough_stats.sort_values('avg_duration', ascending=False)
        
        # Calculate performance metrics
        best_borough = borough_stats['avg_duration'].idxmin()
        worst_borough = borough_stats['avg_duration'].idxmax()
        performance_range = borough_stats['avg_duration'].max() - borough_stats['avg_duration'].min()
        
        metrics = {
            'best_performing_borough': best_borough,
            'worst_performing_borough': worst_borough,
            'borough_performance_range_minutes': performance_range,
            'borough_stats': borough_stats.to_dict('index')
        }
        
        self.business_metrics.update(metrics)
        
        logger.info(f"✅ Borough Analysis:")
        logger.info(f"   Best: {best_borough} ({borough_stats.loc[best_borough, 'avg_duration']:.1f} min avg)")
        logger.info(f"   Worst: {worst_borough} ({borough_stats.loc[worst_borough, 'avg_duration']:.1f} min avg)")
        logger.info(f"   Range: {performance_range:.1f} minutes")
        
        return metrics
    
    def generate_evaluation_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.
        
        Returns:
            Complete evaluation report
        """
        logger.info("📋 Generating evaluation report...")
        
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'eta_model_metrics': self.eta_metrics,
            'delay_model_metrics': self.delay_metrics,
            'business_metrics': self.business_metrics
        }
        
        # Summary statistics
        if 'test_mae' in self.eta_metrics:
            report['summary'] = {
                'eta_performance': {
                    'mae_minutes': self.eta_metrics['test_mae'],
                    'p90_coverage': self.eta_metrics['test_p90_coverage'],
                    'quality_grade': 'Excellent' if self.eta_metrics['test_mae'] < 5 else 'Good'
                }
            }
        
        if 'test_roc_auc' in self.delay_metrics:
            report['summary']['delay_performance'] = {
                'roc_auc': self.delay_metrics['test_roc_auc'],
                'f1_score': self.delay_metrics['test_optimal_f1'],
                'quality_grade': 'Excellent' if self.delay_metrics['test_roc_auc'] > 0.8 else 'Good'
            }
        
        logger.info("✅ Evaluation report generated")
        return report
    
    def save_report(self, filepath: str):
        """
        Save evaluation report to JSON file.
        
        Args:
            filepath: Path to save report
        """
        report = self.generate_evaluation_report()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"💾 Evaluation report saved to {filepath}")
    
    def create_performance_plots(self, y_true: np.ndarray, y_pred: np.ndarray, save_dir: str = None):
        """
        Create performance visualization plots.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            save_dir: Directory to save plots
        """
        logger.info("📊 Creating performance plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Prediction vs Actual scatter
        axes[0, 0].scatter(y_true, y_pred, alpha=0.5)
        axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Duration (minutes)')
        axes[0, 0].set_ylabel('Predicted Duration (minutes)')
        axes[0, 0].set_title('Prediction vs Actual')
        
        # Plot 2: Residuals
        residuals = y_true - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.5)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Duration (minutes)')
        axes[0, 1].set_ylabel('Residuals (minutes)')
        axes[0, 1].set_title('Residual Plot')
        
        # Plot 3: Distribution comparison
        axes[1, 0].hist(y_true, bins=50, alpha=0.7, label='Actual', density=True)
        axes[1, 0].hist(y_pred, bins=50, alpha=0.7, label='Predicted', density=True)
        axes[1, 0].set_xlabel('Duration (minutes)')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Distribution Comparison')
        axes[1, 0].legend()
        
        # Plot 4: Error distribution
        errors = np.abs(y_true - y_pred)
        axes[1, 1].hist(errors, bins=50, alpha=0.7)
        axes[1, 1].axvline(np.mean(errors), color='r', linestyle='--', label=f'MAE: {np.mean(errors):.2f}')
        axes[1, 1].set_xlabel('Absolute Error (minutes)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Error Distribution')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(f"{save_dir}/model_performance.png", dpi=300, bbox_inches='tight')
            logger.info(f"📊 Performance plots saved to {save_dir}/model_performance.png")
        
        plt.show()


def evaluate_models(
    eta_model,
    df_test: pd.DataFrame,
    target_col: str = 'trip_duration_minutes',
    save_dir: str = None
) -> Dict[str, Any]:
    """
    Complete model evaluation pipeline.
    
    Args:
        eta_model: Trained ETA model
        df_test: Test dataset
        target_col: Target column name
        save_dir: Directory to save results
        
    Returns:
        Complete evaluation results
    """
    evaluator = ModelEvaluator()
    
    # Evaluate ETA model
    y_true = df_test[target_col].values
    y_pred = eta_model.predict(df_test)
    
    eta_metrics = evaluator.evaluate_eta_model(y_true, y_pred, "test")
    
    # Business strategy analysis
    promise_metrics = evaluator.analyze_promise_strategy(y_true, "test_data")
    
    # Borough analysis
    borough_metrics = evaluator.borough_analysis(df_test, target_col)
    
    # Generate plots
    if save_dir:
        evaluator.create_performance_plots(y_true, y_pred, save_dir)
    
    # Save comprehensive report
    if save_dir:
        evaluator.save_report(f"{save_dir}/evaluation_report.json")
    
    return evaluator.generate_evaluation_report()


if __name__ == "__main__":
    # Example usage for testing
    logger.info("🧪 Testing model evaluation...")
    
    # Generate sample data for testing
    np.random.seed(42)
    n_samples = 1000
    
    y_true = np.random.exponential(15, n_samples)  # Realistic trip durations
    y_pred = y_true + np.random.normal(0, 3, n_samples)  # Add some prediction error
    
    # Test evaluation
    evaluator = ModelEvaluator()
    eta_metrics = evaluator.evaluate_eta_model(y_true, y_pred, "test")
    promise_metrics = evaluator.analyze_promise_strategy(y_true, "sample")
    
    report = evaluator.generate_evaluation_report()
    
    logger.info(f"✅ Model evaluation test completed!")
    logger.info(f"📊 Sample MAE: {eta_metrics['test_mae']:.2f} minutes")
