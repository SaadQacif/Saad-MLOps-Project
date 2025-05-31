"""
Model monitoring module for the MLOps project.
This module provides functionality for monitoring model performance 
and detecting drift in production.
"""

import os
import time
import json
import pickle
import datetime
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/model_monitoring.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("model_monitoring")


class ModelMonitor:
    """
    A class for monitoring model performance and detecting drift in production.
    """
    
    def __init__(self, 
                 model_path: str, 
                 encoder_path: Optional[str] = None,
                 reference_data_path: Optional[str] = None,
                 metrics_file: Optional[str] = None,
                 log_dir: Optional[str] = None):
        """
        Initialize the model monitor.
        
        Args:
            model_path: Path to the model file
            encoder_path: Path to the encoder file
            reference_data_path: Path to reference data (training data)
            metrics_file: Path to save metrics
            log_dir: Directory to store logs
        """
        self.model_path = model_path
        self.encoder_path = encoder_path
        self.reference_data_path = reference_data_path
        
        # Set up log directory
        if log_dir is None:
            log_dir = os.path.join(os.getcwd(), "logs", "model_monitoring")
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Set up metrics file
        if metrics_file is None:
            metrics_file = os.path.join(log_dir, "metrics.json")
        self.metrics_file = metrics_file
        
        # Load model and encoder
        self._load_model()
        
        # Initialize metrics
        self.metrics = self._load_metrics()
        
        # Load reference data if available
        self.reference_data = None
        self.reference_stats = None
        if reference_data_path:
            self._load_reference_data()
    
    def _load_model(self):
        """Load the model and encoder."""
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Check if it's a models_by_features dictionary or a direct model
            if isinstance(model_data, dict) and 'model' in model_data:
                self.model = model_data['model']
                self.model_info = model_data
            else:
                self.model = model_data
                self.model_info = {'model': model_data}
            
            logger.info(f"Model loaded successfully from {self.model_path}")
            
            # Load encoder if available
            if self.encoder_path and os.path.exists(self.encoder_path):
                with open(self.encoder_path, 'rb') as f:
                    self.encoder = pickle.load(f)
                logger.info(f"Encoder loaded successfully from {self.encoder_path}")
            else:
                self.encoder = None
        
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def _load_metrics(self) -> Dict:
        """Load metrics from file or create new metrics dict."""
        if os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Error decoding metrics file: {self.metrics_file}")
        
        return {
            "predictions": {
                "total": 0,
                "by_class": defaultdict(int),
                "by_hour": defaultdict(int),
            },
            "performance": {
                "latency": [],
                "avg_latency": 0,
                "p95_latency": 0,
            },
            "accuracy": {
                "total": 0,
                "correct": 0,
                "rate": 0,
                "by_class": {},
            },
            "drift": {
                "feature_drift": {},
                "prediction_drift": {},
                "last_checked": None,
            }
        }
    
    def _save_metrics(self):
        """Save metrics to file."""
        # Convert defaultdicts to regular dicts for serialization
        serializable_metrics = json.loads(json.dumps(self.metrics, default=lambda x: dict(x) if isinstance(x, defaultdict) else x))
        
        with open(self.metrics_file, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
    
    def _load_reference_data(self):
        """Load reference data for drift detection."""
        try:
            # Try to load as CSV first
            if self.reference_data_path.endswith('.csv'):
                self.reference_data = pd.read_csv(self.reference_data_path)
            # Try to load as numpy array
            elif self.reference_data_path.endswith('.npy'):
                self.reference_data = np.load(self.reference_data_path)
            # Try to load as pickle
            else:
                with open(self.reference_data_path, 'rb') as f:
                    self.reference_data = pickle.load(f)
            
            # Calculate reference statistics
            self._calculate_reference_stats()
            
            logger.info(f"Reference data loaded successfully from {self.reference_data_path}")
        
        except Exception as e:
            logger.error(f"Error loading reference data: {str(e)}")
            self.reference_data = None
    
    def _calculate_reference_stats(self):
        """Calculate statistics from reference data for drift detection."""
        if self.reference_data is None:
            return
        
        try:
            # Convert to numpy array if it's not already
            if isinstance(self.reference_data, pd.DataFrame):
                features = self.reference_data.select_dtypes(include=[np.number]).values
            else:
                features = np.array(self.reference_data)
            
            # Calculate statistics
            self.reference_stats = {
                "mean": np.mean(features, axis=0),
                "std": np.std(features, axis=0),
                "min": np.min(features, axis=0),
                "max": np.max(features, axis=0),
                "q25": np.percentile(features, 25, axis=0),
                "median": np.percentile(features, 50, axis=0),
                "q75": np.percentile(features, 75, axis=0)
            }
            
            logger.info("Reference statistics calculated")
        
        except Exception as e:
            logger.error(f"Error calculating reference statistics: {str(e)}")
            self.reference_stats = None
    
    def log_prediction(self, features, prediction, actual=None, latency=None):
        """
        Log a prediction.
        
        Args:
            features: Feature values used for prediction
            prediction: Model prediction
            actual: Actual label (if available)
            latency: Prediction latency in seconds
        """
        # Update prediction count
        self.metrics["predictions"]["total"] += 1
        self.metrics["predictions"]["by_class"][str(prediction)] += 1
        
        # Update hourly metrics
        hour = datetime.datetime.now().strftime("%Y-%m-%d-%H")
        self.metrics["predictions"]["by_hour"][hour] += 1
        
        # Update latency metrics
        if latency is not None:
            self.metrics["performance"]["latency"].append(latency)
            
            # Update average and p95 latency
            latencies = self.metrics["performance"]["latency"][-100:]  # Use last 100 predictions
            self.metrics["performance"]["avg_latency"] = sum(latencies) / len(latencies)
            self.metrics["performance"]["p95_latency"] = sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) >= 20 else 0
        
        # Update accuracy metrics if actual label is available
        if actual is not None:
            self.metrics["accuracy"]["total"] += 1
            
            if str(prediction) == str(actual):
                self.metrics["accuracy"]["correct"] += 1
            
            # Update accuracy rate
            self.metrics["accuracy"]["rate"] = (
                self.metrics["accuracy"]["correct"] / self.metrics["accuracy"]["total"]
            )
            
            # Update accuracy by class
            if str(actual) not in self.metrics["accuracy"]["by_class"]:
                self.metrics["accuracy"]["by_class"][str(actual)] = {
                    "total": 0,
                    "correct": 0,
                    "rate": 0
                }
            
            self.metrics["accuracy"]["by_class"][str(actual)]["total"] += 1
            
            if str(prediction) == str(actual):
                self.metrics["accuracy"]["by_class"][str(actual)]["correct"] += 1
            
            self.metrics["accuracy"]["by_class"][str(actual)]["rate"] = (
                self.metrics["accuracy"]["by_class"][str(actual)]["correct"] /
                self.metrics["accuracy"]["by_class"][str(actual)]["total"]
            )
        
        # Save metrics periodically (every 10 predictions)
        if self.metrics["predictions"]["total"] % 10 == 0:
            self._save_metrics()
    
    def check_drift(self, recent_features=None, recent_predictions=None, sample_size=100) -> Dict[str, Any]:
        """
        Check for data drift by comparing recent features to reference data.
        
        Args:
            recent_features: Recent feature values (if None, uses stored features)
            recent_predictions: Recent predictions (if None, uses stored predictions)
            sample_size: Size of sample to use for drift detection
            
        Returns:
            Dictionary with drift metrics
        """
        if self.reference_stats is None:
            logger.warning("Reference statistics not available for drift detection")
            return {}
        
        drift_metrics = {}
        
        try:
            # Feature drift
            if recent_features is not None:
                # Convert to numpy array
                recent_features = np.array(recent_features)
                
                # Calculate statistics
                recent_mean = np.mean(recent_features, axis=0)
                recent_std = np.std(recent_features, axis=0)
                
                # Calculate drift metrics
                mean_drift = np.abs(recent_mean - self.reference_stats["mean"]) / self.reference_stats["std"]
                std_drift = np.abs(recent_std - self.reference_stats["std"]) / self.reference_stats["std"]
                
                # Average drift across features
                avg_mean_drift = np.mean(mean_drift)
                avg_std_drift = np.mean(std_drift)
                
                drift_metrics["feature_drift"] = {
                    "mean_drift": avg_mean_drift,
                    "std_drift": avg_std_drift,
                    "drift_detected": avg_mean_drift > 0.5 or avg_std_drift > 0.5
                }
            
            # Prediction drift
            if recent_predictions is not None:
                # Calculate distribution
                recent_dist = {}
                for pred in recent_predictions:
                    pred_str = str(pred)
                    if pred_str not in recent_dist:
                        recent_dist[pred_str] = 0
                    recent_dist[pred_str] += 1
                
                # Normalize
                total = sum(recent_dist.values())
                recent_dist = {k: v / total for k, v in recent_dist.items()}
                
                # Get reference distribution
                ref_dist = {}
                total_preds = self.metrics["predictions"]["total"]
                
                if total_preds > 0:
                    for cls, count in self.metrics["predictions"]["by_class"].items():
                        ref_dist[cls] = count / total_preds
                    
                    # Calculate drift
                    classes = set(list(recent_dist.keys()) + list(ref_dist.keys()))
                    drift_sum = 0
                    
                    for cls in classes:
                        recent_val = recent_dist.get(cls, 0)
                        ref_val = ref_dist.get(cls, 0)
                        drift_sum += abs(recent_val - ref_val)
                    
                    drift_metrics["prediction_drift"] = {
                        "distribution_change": drift_sum / 2,  # Normalize to [0, 1]
                        "drift_detected": drift_sum / 2 > 0.2
                    }
            
            # Update drift metrics in storage
            self.metrics["drift"].update(drift_metrics)
            self.metrics["drift"]["last_checked"] = datetime.datetime.now().isoformat()
            self._save_metrics()
            
            return drift_metrics
        
        except Exception as e:
            logger.error(f"Error checking for drift: {str(e)}")
            return {}
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of metrics.
        
        Returns:
            Dictionary with metrics summary
        """
        # Create a deep copy to avoid modifying the original
        summary = json.loads(json.dumps(self.metrics))
        
        # Add additional calculated metrics
        if summary["predictions"]["total"] > 0:
            # Get top predictions
            top_predictions = sorted(
                summary["predictions"]["by_class"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            summary["top_predictions"] = dict(top_predictions)
            
            # Calculate prediction distribution
            total_preds = summary["predictions"]["total"]
            summary["predictions"]["distribution"] = {
                k: v / total_preds for k, v in summary["predictions"]["by_class"].items()
            }
        
        # Limit the size of latency array in the response
        if "latency" in summary["performance"]:
            summary["performance"]["latency"] = summary["performance"]["latency"][-10:]
        
        return summary


# Helper function to create a model monitor
def create_model_monitor(model_path, encoder_path=None, reference_data_path=None):
    """
    Create a model monitor instance.
    
    Args:
        model_path: Path to the model file
        encoder_path: Path to the encoder file
        reference_data_path: Path to reference data
        
    Returns:
        ModelMonitor instance
    """
    try:
        monitor = ModelMonitor(
            model_path=model_path,
            encoder_path=encoder_path,
            reference_data_path=reference_data_path
        )
        return monitor
    except Exception as e:
        logger.error(f"Error creating model monitor: {str(e)}")
        return None
