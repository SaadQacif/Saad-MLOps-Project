#!/usr/bin/env python
"""
Automated training pipeline that orchestrates the entire ML workflow
"""

import os
import sys
import json
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import mlflow
from mlflow.tracking import MlflowClient

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.mlflow_integration import MLflowManager
from src.model_tracking import MLFlowLikeTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MLTrainingPipeline:
    """Orchestrates the complete ML training pipeline"""
    
    def __init__(self, 
                 data_path: str,
                 output_path: str,
                 experiment_name: str = "potato-disease-classification",
                 mlflow_uri: str = None):
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.experiment_name = experiment_name
        self.mlflow_uri = mlflow_uri or os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        
        # Initialize MLflow
        mlflow.set_tracking_uri(self.mlflow_uri)
        self.client = MlflowClient(tracking_uri=self.mlflow_uri)
        
        # Ensure experiment exists
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                self.experiment_id = mlflow.create_experiment(experiment_name)
            else:
                self.experiment_id = experiment.experiment_id
        except Exception as e:
            logger.error(f"Failed to create/get experiment: {e}")
            raise

    def extract_features(self):
        """Extract features from raw images"""
        logger.info("Starting feature extraction...")
        
        with mlflow.start_run(experiment_id=self.experiment_id, run_name="feature_extraction") as run:
            try:
                # Run feature extraction script
                cmd = [
                    sys.executable,
                    str(project_root / "scripts" / "extract_potato_features.py"),
                    "--input-dir", str(self.data_path),
                    "--output-file", str(self.output_path / "features.csv")
                ]
                
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                
                # Log parameters
                mlflow.log_param("input_data_path", str(self.data_path))
                mlflow.log_param("feature_extraction_script", "extract_potato_features.py")
                
                # Log metrics
                features_df = self.load_features()
                mlflow.log_metric("num_samples", len(features_df))
                mlflow.log_metric("num_features", features_df.shape[1] - 1)  # -1 for class column
                
                logger.info("Feature extraction completed successfully")
                return True
                
            except subprocess.CalledProcessError as e:
                logger.error(f"Feature extraction failed: {e}")
                mlflow.log_param("status", "failed")
                mlflow.log_param("error", str(e))
                return False

    def load_features(self):
        """Load extracted features"""
        features_file = self.output_path / "features.csv"
        if not features_file.exists():
            raise FileNotFoundError(f"Features file not found: {features_file}")
        
        import pandas as pd
        return pd.read_csv(features_file)

    def train_model(self):
        """Train the machine learning model"""
        logger.info("Starting model training...")
        
        with mlflow.start_run(experiment_id=self.experiment_id, run_name="model_training") as run:
            try:
                # Run training script
                cmd = [
                    sys.executable,
                    str(project_root / "src" / "train_model_features.py"),
                    "--features", str(self.output_path / "features.csv"),
                    "--model-dir", str(self.output_path / "models"),
                    "--mlflow-tracking-uri", self.mlflow_uri
                ]
                
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                
                # Log training parameters
                mlflow.log_param("training_script", "train_model_features.py")
                mlflow.log_param("features_file", str(self.output_path / "features.csv"))
                
                logger.info("Model training completed successfully")
                return True
                
            except subprocess.CalledProcessError as e:
                logger.error(f"Model training failed: {e}")
                mlflow.log_param("status", "failed")
                mlflow.log_param("error", str(e))
                return False

    def evaluate_model(self):
        """Evaluate the trained model"""
        logger.info("Starting model evaluation...")
        
        with mlflow.start_run(experiment_id=self.experiment_id, run_name="model_evaluation") as run:
            try:
                # Run evaluation script
                cmd = [
                    sys.executable,
                    str(project_root / "src" / "evaluate_model.py"),
                    "--model-dir", str(self.output_path / "models"),
                    "--test-data", str(self.output_path / "features.csv"),
                    "--output-dir", str(self.output_path / "evaluation")
                ]
                
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                
                # Load and log evaluation metrics
                eval_metrics_file = self.output_path / "evaluation" / "metrics.json"
                if eval_metrics_file.exists():
                    with open(eval_metrics_file) as f:
                        metrics = json.load(f)
                    
                    for metric_name, value in metrics.items():
                        mlflow.log_metric(metric_name, value)
                
                logger.info("Model evaluation completed successfully")
                return True
                
            except subprocess.CalledProcessError as e:
                logger.error(f"Model evaluation failed: {e}")
                mlflow.log_param("status", "failed")
                mlflow.log_param("error", str(e))
                return False

    def register_model(self):
        """Register the model in MLflow Model Registry"""
        logger.info("Registering model...")
        
        try:
            # Find the latest model run
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="tags.mlflow.runName LIKE 'model_training%'",
                order_by=["start_time DESC"],
                max_results=1
            )
            
            if runs.empty:
                logger.error("No training runs found")
                return False
            
            latest_run = runs.iloc[0]
            run_id = latest_run.run_id
            
            # Register model
            model_name = "potato-disease-model"
            model_uri = f"runs:/{run_id}/model"
            
            try:
                # Create registered model if it doesn't exist
                self.client.create_registered_model(model_name)
            except Exception:
                # Model already exists
                pass
            
            # Create model version
            model_version = self.client.create_model_version(
                name=model_name,
                source=model_uri,
                run_id=run_id
            )
            
            # Get model metrics for validation
            accuracy = latest_run.get("metrics.accuracy", 0)
            
            # Auto-promote to staging if accuracy > 0.8
            if accuracy > 0.8:
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=model_version.version,
                    stage="Staging"
                )
                logger.info(f"Model version {model_version.version} promoted to Staging")
                
                # Auto-promote to production if accuracy > 0.9
                if accuracy > 0.9:
                    self.client.transition_model_version_stage(
                        name=model_name,
                        version=model_version.version,
                        stage="Production"
                    )
                    logger.info(f"Model version {model_version.version} promoted to Production")
            
            logger.info(f"Model registered successfully: {model_name} v{model_version.version}")
            return True
            
        except Exception as e:
            logger.error(f"Model registration failed: {e}")
            return False

    def run_complete_pipeline(self):
        """Run the complete ML pipeline"""
        logger.info("Starting complete ML pipeline...")
        
        start_time = datetime.now()
        
        with mlflow.start_run(experiment_id=self.experiment_id, run_name="complete_pipeline") as run:
            # Log pipeline start
            mlflow.log_param("pipeline_start_time", start_time.isoformat())
            mlflow.log_param("data_path", str(self.data_path))
            mlflow.log_param("output_path", str(self.output_path))
            
            success = True
            
            # Step 1: Feature extraction
            if not self.extract_features():
                success = False
                mlflow.log_param("failed_step", "feature_extraction")
            
            # Step 2: Model training
            if success and not self.train_model():
                success = False
                mlflow.log_param("failed_step", "model_training")
            
            # Step 3: Model evaluation
            if success and not self.evaluate_model():
                success = False
                mlflow.log_param("failed_step", "model_evaluation")
            
            # Step 4: Model registration
            if success and not self.register_model():
                success = False
                mlflow.log_param("failed_step", "model_registration")
            
            # Log pipeline end
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            mlflow.log_param("pipeline_end_time", end_time.isoformat())
            mlflow.log_metric("pipeline_duration_seconds", duration)
            mlflow.log_param("pipeline_status", "success" if success else "failed")
            
            if success:
                logger.info(f"Complete ML pipeline finished successfully in {duration:.2f} seconds")
            else:
                logger.error(f"ML pipeline failed after {duration:.2f} seconds")
            
            return success

def main():
    parser = argparse.ArgumentParser(description="Automated ML Training Pipeline")
    parser.add_argument("--data-path", required=True, help="Path to training data")
    parser.add_argument("--output-path", required=True, help="Path for output artifacts")
    parser.add_argument("--experiment-name", default="potato-disease-classification", 
                       help="MLflow experiment name")
    parser.add_argument("--mlflow-uri", help="MLflow tracking URI")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Initialize and run pipeline
    pipeline = MLTrainingPipeline(
        data_path=args.data_path,
        output_path=args.output_path,
        experiment_name=args.experiment_name,
        mlflow_uri=args.mlflow_uri
    )
    
    success = pipeline.run_complete_pipeline()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
