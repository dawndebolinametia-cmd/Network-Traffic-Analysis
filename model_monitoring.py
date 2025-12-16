import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import os
from datetime import datetime
from logging_utils import logger
from config import MONITORING_FILE, RETRAIN_THRESHOLD

class ModelMonitor:
    """
    Monitors model performance and triggers retraining when necessary.
    """

    def __init__(self, monitoring_file=MONITORING_FILE):
        """
        Initialize the model monitor.

        Args:
            monitoring_file (str): Path to the monitoring data file
        """
        self.monitoring_file = monitoring_file
        self.performance_history = self._load_performance_history()

    def _load_performance_history(self):
        """
        Load performance history from file.
        """
        if os.path.exists(self.monitoring_file):
            try:
                with open(self.monitoring_file, 'r') as f:
                    data = json.load(f)
                return data
            except Exception as e:
                logger.error(f"Failed to load performance history: {e}")
                return []
        return []

    def _save_performance_history(self):
        """
        Save performance history to file.
        """
        try:
            with open(self.monitoring_file, 'w') as f:
                json.dump(self.performance_history, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save performance history: {e}")

    def record_performance(self, y_true, y_pred):
        """
        Record model performance metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
        """
        try:
            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

            # Create performance record
            record = {
                'timestamp': datetime.now().isoformat(),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'sample_size': len(y_true)
            }

            # Add to history
            self.performance_history.append(record)

            # Keep only last 100 records to prevent file from growing too large
            if len(self.performance_history) > 100:
                self.performance_history = self.performance_history[-100:]

            # Save to file
            self._save_performance_history()

            logger.info(f"Performance recorded - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

        except Exception as e:
            logger.error(f"Failed to record performance: {e}")

    def get_performance_trend(self, window=10):
        """
        Get performance trend over recent runs.

        Args:
            window (int): Number of recent records to analyze

        Returns:
            dict: Performance trend metrics
        """
        if len(self.performance_history) < window:
            logger.warning(f"Insufficient data for trend analysis. Need at least {window} records.")
            return None

        recent_records = self.performance_history[-window:]

        accuracies = [r['accuracy'] for r in recent_records]
        f1_scores = [r['f1_score'] for r in recent_records]

        trend = {
            'current_accuracy': accuracies[-1],
            'avg_accuracy': np.mean(accuracies),
            'accuracy_std': np.std(accuracies),
            'accuracy_trend': np.polyfit(range(len(accuracies)), accuracies, 1)[0],  # Linear trend
            'current_f1': f1_scores[-1],
            'avg_f1': np.mean(f1_scores),
            'f1_std': np.std(f1_scores),
            'f1_trend': np.polyfit(range(len(f1_scores)), f1_scores, 1)[0]
        }

        return trend

    def should_retrain(self, threshold=RETRAIN_THRESHOLD):
        """
        Check if model should be retrained based on performance degradation.

        Args:
            threshold (float): Performance drop threshold for retraining

        Returns:
            bool: True if retraining is recommended
        """
        trend = self.get_performance_trend()

        if trend is None:
            return False

        # Check if accuracy has dropped significantly
        if trend['current_accuracy'] < trend['avg_accuracy'] - threshold:
            logger.warning(f"Model performance degraded. Current accuracy: {trend['current_accuracy']:.4f}, "
                         f"Average: {trend['avg_accuracy']:.4f}")
            return True

        # Check if F1 score has dropped significantly
        if trend['current_f1'] < trend['avg_f1'] - threshold:
            logger.warning(f"Model F1 score degraded. Current F1: {trend['current_f1']:.4f}, "
                         f"Average: {trend['avg_f1']:.4f}")
            return True

        return False

    def get_monitoring_summary(self):
        """
        Get a summary of monitoring data.

        Returns:
            dict: Monitoring summary
        """
        if not self.performance_history:
            return {"message": "No performance data available"}

        latest = self.performance_history[-1]
        trend = self.get_performance_trend()

        summary = {
            'total_runs': len(self.performance_history),
            'latest_performance': {
                'timestamp': latest['timestamp'],
                'accuracy': latest['accuracy'],
                'precision': latest['precision'],
                'recall': latest['recall'],
                'f1_score': latest['f1_score'],
                'sample_size': latest['sample_size']
            },
            'retraining_needed': self.should_retrain()
        }

        if trend:
            summary['trend'] = trend

        return summary

# Global monitor instance
monitor = ModelMonitor()
