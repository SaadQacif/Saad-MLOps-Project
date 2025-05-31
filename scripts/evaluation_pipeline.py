#!/usr/bin/env python
"""
Model evaluation pipeline for validating model performance
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import mlflow
from mlflow.tracking import MlflowClient

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelEvaluationPipeline:
    """Comprehensive model evaluation pipeline"""
    
    def __init__(self, 
                 model_path: str,
                 test_data_path: str,
                 experiment_name: str = "potato-disease-classification",
                 mlflow_uri: str = None):
        self.model_path = Path(model_path)
        self.test_data_path = Path(test_data_path)
        self.experiment_name = experiment_name
        self.mlflow_uri = mlflow_uri or os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        
        # Initialize MLflow
        mlflow.set_tracking_uri(self.mlflow_uri)
        self.client = MlflowClient(tracking_uri=self.mlflow_uri)

    def load_model_artifacts(self):
        """Load the trained model and associated artifacts"""
        import pickle
        
        try:
            # Load model
            model_file = None
            for file in os.listdir(self.model_path / "bin"):
                if file.endswith('_model.pkl'):
                    model_file = self.model_path / "bin" / file
                    break
            
            if not model_file:
                raise FileNotFoundError("No model file found")
            
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            
            # Load label encoder
            encoder_path = self.model_path / "bin" / "label_encoder.pkl"
            with open(encoder_path, 'rb') as f:
                encoder = pickle.load(f)
            
            # Load scaler
            scaler_path = self.model_path / "bin" / "scaler.pkl"
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            
            # Load metadata
            metadata_path = self.model_path / "bin" / "model_metadata.json"
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            return model, encoder, scaler, metadata
            
        except Exception as e:
            logger.error(f"Failed to load model artifacts: {e}")
            raise

    def load_test_data(self):
        """Load test data for evaluation"""
        try:
            if self.test_data_path.is_file() and self.test_data_path.suffix == '.csv':
                # Load from CSV file
                df = pd.read_csv(self.test_data_path)
                return df
            else:
                # Load from directory structure
                logger.info("Loading test data from directory structure...")
                # This would use the feature extraction pipeline
                # For now, assume CSV format
                raise NotImplementedError("Directory-based test data loading not implemented")
                
        except Exception as e:
            logger.error(f"Failed to load test data: {e}")
            raise

    def evaluate_model_performance(self, model, encoder, scaler, test_data):
        """Comprehensive model evaluation"""
        from sklearn.metrics import (
            accuracy_score, f1_score, precision_score, recall_score,
            confusion_matrix, classification_report, roc_auc_score
        )
        
        try:
            # Prepare test data
            if 'class' in test_data.columns:
                X_test = test_data.drop('class', axis=1)
                y_test = test_data['class']
            else:
                raise ValueError("Test data must contain 'class' column")
            
            # Scale features
            X_test_scaled = scaler.transform(X_test)
            
            # Encode labels
            y_test_encoded = encoder.transform(y_test)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test_encoded, y_pred)
            f1 = f1_score(y_test_encoded, y_pred, average='weighted')
            precision = precision_score(y_test_encoded, y_pred, average='weighted')
            recall = recall_score(y_test_encoded, y_pred, average='weighted')
            
            # Class-specific metrics
            class_report = classification_report(y_test_encoded, y_pred, 
                                               target_names=encoder.classes_, 
                                               output_dict=True)
            
            # Confusion matrix
            cm = confusion_matrix(y_test_encoded, y_pred)
            
            # ROC AUC (for multiclass)
            try:
                roc_auc = roc_auc_score(y_test_encoded, y_pred_proba, 
                                       multi_class='ovr', average='weighted')
            except Exception:
                roc_auc = None
            
            metrics = {
                'accuracy': accuracy,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'roc_auc': roc_auc,
                'confusion_matrix': cm.tolist(),
                'classification_report': class_report,
                'num_test_samples': len(y_test),
                'num_classes': len(encoder.classes_),
                'class_names': encoder.classes_.tolist()
            }
            
            return metrics, y_pred, y_pred_proba
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            raise

    def generate_evaluation_artifacts(self, metrics, y_test, y_pred, y_pred_proba, encoder):
        """Generate evaluation artifacts and visualizations"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        artifacts_dir = Path("evaluation_artifacts")
        artifacts_dir.mkdir(exist_ok=True)
        
        try:
            # Confusion matrix visualization
            plt.figure(figsize=(10, 8))
            cm = np.array(metrics['confusion_matrix'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=encoder.classes_,
                       yticklabels=encoder.classes_)
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(artifacts_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Class distribution
            plt.figure(figsize=(10, 6))
            class_counts = pd.Series(y_test).value_counts()
            class_counts.plot(kind='bar')
            plt.title('Test Data Class Distribution')
            plt.xlabel('Class')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(artifacts_dir / 'class_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Performance by class
            class_metrics = metrics['classification_report']
            classes = [k for k in class_metrics.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            for i, metric in enumerate(['precision', 'recall', 'f1-score']):
                values = [class_metrics[cls][metric] for cls in classes]
                axes[i].bar(classes, values)
                axes[i].set_title(f'{metric.title()} by Class')
                axes[i].set_ylabel(metric.title())
                axes[i].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(artifacts_dir / 'class_metrics.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save metrics as JSON
            metrics_json = {k: v for k, v in metrics.items() 
                           if k not in ['confusion_matrix']}  # Remove non-serializable items
            
            with open(artifacts_dir / 'evaluation_metrics.json', 'w') as f:
                json.dump(metrics_json, f, indent=2)
            
            return artifacts_dir
            
        except Exception as e:
            logger.error(f"Failed to generate artifacts: {e}")
            return None

    def check_model_quality_gates(self, metrics):
        """Check if model meets quality gates for deployment"""
        quality_gates = {
            'min_accuracy': 0.85,
            'min_f1_score': 0.80,
            'min_precision': 0.80,
            'min_recall': 0.75
        }
        
        passed_gates = {}
        overall_pass = True
        
        for gate, threshold in quality_gates.items():
            metric_name = gate.replace('min_', '')
            metric_value = metrics.get(metric_name, 0)
            passed = metric_value >= threshold
            passed_gates[gate] = {
                'threshold': threshold,
                'actual': metric_value,
                'passed': passed
            }
            if not passed:
                overall_pass = False
        
        return overall_pass, passed_gates

    def run_evaluation(self):
        """Run complete model evaluation pipeline"""
        logger.info("Starting model evaluation pipeline...")
        
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(self.experiment_name)
        else:
            experiment_id = experiment.experiment_id
        
        with mlflow.start_run(experiment_id=experiment_id, run_name="model_evaluation") as run:
            try:
                # Load model and test data
                model, encoder, scaler, metadata = self.load_model_artifacts()
                test_data = self.load_test_data()
                
                # Log evaluation parameters
                mlflow.log_param("model_path", str(self.model_path))
                mlflow.log_param("test_data_path", str(self.test_data_path))
                mlflow.log_param("evaluation_timestamp", datetime.now().isoformat())
                
                # Evaluate model
                metrics, y_pred, y_pred_proba = self.evaluate_model_performance(
                    model, encoder, scaler, test_data
                )
                
                # Log metrics to MLflow
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f"eval_{metric_name}", value)
                
                # Generate artifacts
                artifacts_dir = self.generate_evaluation_artifacts(
                    metrics, test_data['class'], y_pred, y_pred_proba, encoder
                )
                
                if artifacts_dir:
                    # Log artifacts to MLflow
                    for artifact_file in artifacts_dir.glob('*'):
                        mlflow.log_artifact(str(artifact_file))
                
                # Check quality gates
                gates_passed, gate_results = self.check_model_quality_gates(metrics)
                
                mlflow.log_param("quality_gates_passed", gates_passed)
                for gate, result in gate_results.items():
                    mlflow.log_param(f"gate_{gate}", result['passed'])
                    mlflow.log_metric(f"gate_{gate}_threshold", result['threshold'])
                
                # Log evaluation summary
                evaluation_summary = {
                    'status': 'success',
                    'quality_gates_passed': gates_passed,
                    'key_metrics': {
                        'accuracy': metrics['accuracy'],
                        'f1_score': metrics['f1_score'],
                        'precision': metrics['precision'],
                        'recall': metrics['recall']
                    }
                }
                
                mlflow.log_dict(evaluation_summary, "evaluation_summary.json")
                
                logger.info(f"Model evaluation completed successfully")
                logger.info(f"Quality gates passed: {gates_passed}")
                logger.info(f"Key metrics - Accuracy: {metrics['accuracy']:.3f}, "
                           f"F1: {metrics['f1_score']:.3f}")
                
                return True
                
            except Exception as e:
                logger.error(f"Model evaluation failed: {e}")
                mlflow.log_param("status", "failed")
                mlflow.log_param("error", str(e))
                return False

def main():
    parser = argparse.ArgumentParser(description="Model Evaluation Pipeline")
    parser.add_argument("--model-path", required=True, help="Path to trained model")
    parser.add_argument("--test-data-path", required=True, help="Path to test data")
    parser.add_argument("--experiment-name", default="potato-disease-classification",
                       help="MLflow experiment name")
    parser.add_argument("--mlflow-uri", help="MLflow tracking URI")
    
    args = parser.parse_args()
    
    # Initialize and run evaluation
    evaluator = ModelEvaluationPipeline(
        model_path=args.model_path,
        test_data_path=args.test_data_path,
        experiment_name=args.experiment_name,
        mlflow_uri=args.mlflow_uri
    )
    
    success = evaluator.run_evaluation()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
