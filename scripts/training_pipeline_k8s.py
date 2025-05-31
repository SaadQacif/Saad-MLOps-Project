#!/usr/bin/env python3
"""
Comprehensive ML Pipeline for Potato Disease Classification
Orchestrated by Kubernetes Jobs
"""

import os
import sys
import json
import logging
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import mlflow
from mlflow.tracking import MlflowClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MLPipeline:
    """Comprehensive ML Pipeline for Kubernetes orchestration"""
    
    def __init__(self, config_path=None):
        self.config = self.load_config(config_path)
        self.setup_mlflow()
        
    def load_config(self, config_path=None):
        """Load pipeline configuration"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Default configuration
        return {
            "data": {
                "input_path": "/app/data/raw",
                "processed_path": "/app/data/processed",
                "features_path": "/app/data/features",
                "split_path": "/app/data/split"
            },
            "model": {
                "output_path": "/app/models/bin",
                "registry_name": "potato-disease-classifier",
                "experiment_name": "potato-disease-classification"
            },
            "mlflow": {
                "tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
                "registry_uri": os.getenv("MLFLOW_REGISTRY_URI", "sqlite:///mlflow.db")
            },
            "kubernetes": {
                "namespace": os.getenv("KUBERNETES_NAMESPACE", "potato-disease-ml"),
                "job_prefix": "potato-disease"
            }
        }
    
    def setup_mlflow(self):
        """Setup MLflow tracking"""
        mlflow.set_tracking_uri(self.config["mlflow"]["tracking_uri"])
        mlflow.set_registry_uri(self.config["mlflow"]["registry_uri"])
        
        # Ensure experiment exists
        experiment_name = self.config["model"]["experiment_name"]
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)
            logger.info(f"Created MLflow experiment: {experiment_name}")
    
    def run_step(self, step_name, command, check=True):
        """Run a pipeline step with logging"""
        logger.info(f"Starting step: {step_name}")
        start_time = datetime.now()
        
        try:
            if isinstance(command, list):
                result = subprocess.run(command, check=check, capture_output=True, text=True)
            else:
                result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
            
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Step '{step_name}' completed successfully in {duration:.2f}s")
            
            return result
            
        except subprocess.CalledProcessError as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"Step '{step_name}' failed after {duration:.2f}s: {e}")
            logger.error(f"stdout: {e.stdout}")
            logger.error(f"stderr: {e.stderr}")
            raise
    
    def data_preprocessing(self):
        """Step 1: Data preprocessing and validation"""
        logger.info("=" * 60)
        logger.info("STEP 1: DATA PREPROCESSING")
        logger.info("=" * 60)
        
        # Create directories
        for path in self.config["data"].values():
            os.makedirs(path, exist_ok=True)
        
        # Copy and preprocess data
        if os.path.exists("/app/Potato_Health_States"):
            self.run_step(
                "Copy original data",
                f"python scripts/preprocess_potato_data.py "
                f"--input /app/Potato_Health_States "
                f"--output {self.config['data']['processed_path']}"
            )
        else:
            logger.warning("Original data not found, using existing processed data")
        
        # Validate data
        self.run_step(
            "Validate data structure",
            f"python -c \"import os; "
            f"processed_path = '{self.config['data']['processed_path']}'; "
            f"classes = os.listdir(processed_path) if os.path.exists(processed_path) else []; "
            f"print(f'Found classes: {{classes}}'); "
            f"total_images = sum(len(os.listdir(os.path.join(processed_path, c))) for c in classes if os.path.isdir(os.path.join(processed_path, c))); "
            f"print(f'Total images: {{total_images}}'); "
            f"assert total_images > 0, 'No images found!'\""
        )
    
    def feature_extraction(self):
        """Step 2: Feature extraction"""
        logger.info("=" * 60)
        logger.info("STEP 2: FEATURE EXTRACTION")
        logger.info("=" * 60)
        
        features_file = os.path.join(self.config["data"]["features_path"], "features.csv")
        
        self.run_step(
            "Extract features",
            f"python scripts/extract_features.py "
            f"--input {self.config['data']['processed_path']} "
            f"--output {features_file}"
        )
        
        # Validate features
        self.run_step(
            "Validate features",
            f"python -c \"import pandas as pd; "
            f"df = pd.read_csv('{features_file}'); "
            f"print(f'Features shape: {{df.shape}}'); "
            f"print(f'Classes: {{df['class'].unique()}}'); "
            f"assert len(df) > 0, 'No features extracted!'\""
        )
    
    def model_training(self):
        """Step 3: Model training"""
        logger.info("=" * 60)
        logger.info("STEP 3: MODEL TRAINING")
        logger.info("=" * 60)
        
        features_file = os.path.join(self.config["data"]["features_path"], "features.csv")
        
        with mlflow.start_run(run_name=f"pipeline_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
            # Set pipeline tags
            mlflow.set_tags({
                "pipeline.version": os.getenv("PIPELINE_VERSION", "local"),
                "pipeline.environment": "kubernetes",
                "pipeline.step": "training",
                "pipeline.timestamp": datetime.now().isoformat()
            })
            
            self.run_step(
                "Train model",
                f"python src/train_model_features.py "
                f"--features {features_file} "
                f"--model-dir {self.config['model']['output_path']} "
                f"--output-dir /app/visualizations"
            )
            
            # Log pipeline metadata
            mlflow.log_param("pipeline.config", json.dumps(self.config))
            mlflow.log_artifact(features_file, "data")
    
    def model_evaluation(self):
        """Step 4: Model evaluation"""
        logger.info("=" * 60)
        logger.info("STEP 4: MODEL EVALUATION")
        logger.info("=" * 60)
        
        self.run_step(
            "Evaluate model",
            f"python scripts/evaluation_pipeline.py "
            f"--model-dir {self.config['model']['output_path']} "
            f"--output-dir /app/evaluation"
        )
    
    def model_registration(self):
        """Step 5: Model registration"""
        logger.info("=" * 60)
        logger.info("STEP 5: MODEL REGISTRATION")
        logger.info("=" * 60)
        
        try:
            client = MlflowClient()
            
            # Get latest run
            experiment = mlflow.get_experiment_by_name(self.config["model"]["experiment_name"])
            if experiment:
                runs = client.search_runs(
                    experiment.experiment_id, 
                    order_by=['start_time DESC'], 
                    max_results=1
                )
                
                if runs:
                    run_id = runs[0].info.run_id
                    model_uri = f"runs:/{run_id}/model"
                    
                    # Register model
                    model_version = mlflow.register_model(
                        model_uri=model_uri,
                        name=self.config["model"]["registry_name"]
                    )
                    
                    logger.info(f"Registered model version: {model_version.version}")
                    
                    # Transition to staging
                    client.transition_model_version_stage(
                        name=self.config["model"]["registry_name"],
                        version=model_version.version,
                        stage="Staging",
                        archive_existing_versions=False
                    )
                    
                    logger.info(f"Transitioned model version {model_version.version} to Staging")
                    
        except Exception as e:
            logger.error(f"Model registration failed: {e}")
            # Don't fail the pipeline for registration issues
    
    def cleanup(self):
        """Step 6: Cleanup and optimization"""
        logger.info("=" * 60)
        logger.info("STEP 6: CLEANUP")
        logger.info("=" * 60)
        
        # Clean up temporary files
        self.run_step(
            "Cleanup temporary files",
            "find /app -name '*.pyc' -delete; find /app -name '__pycache__' -type d -exec rm -rf {} +",
            check=False
        )
        
        # Compress artifacts
        self.run_step(
            "Compress artifacts",
            "cd /app && tar -czf artifacts.tar.gz visualizations/ evaluation/ || true",
            check=False
        )
    
    def run_pipeline(self, steps=None):
        """Run the complete ML pipeline"""
        logger.info("Starting ML Pipeline for Potato Disease Classification")
        logger.info(f"Configuration: {json.dumps(self.config, indent=2)}")
        
        pipeline_steps = [
            ("data_preprocessing", self.data_preprocessing),
            ("feature_extraction", self.feature_extraction),
            ("model_training", self.model_training),
            ("model_evaluation", self.model_evaluation),
            ("model_registration", self.model_registration),
            ("cleanup", self.cleanup)
        ]
        
        if steps:
            pipeline_steps = [(name, func) for name, func in pipeline_steps if name in steps]
        
        start_time = datetime.now()
        successful_steps = []
        failed_steps = []
        
        try:
            with mlflow.start_run(run_name=f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
                mlflow.set_tags({
                    "pipeline.type": "full_pipeline",
                    "pipeline.version": os.getenv("PIPELINE_VERSION", "local"),
                    "pipeline.environment": "kubernetes"
                })
                
                for step_name, step_func in pipeline_steps:
                    try:
                        step_func()
                        successful_steps.append(step_name)
                        mlflow.log_metric(f"step.{step_name}.success", 1)
                    except Exception as e:
                        failed_steps.append((step_name, str(e)))
                        mlflow.log_metric(f"step.{step_name}.success", 0)
                        logger.error(f"Pipeline step '{step_name}' failed: {e}")
                        if step_name in ["data_preprocessing", "feature_extraction", "model_training"]:
                            # Critical steps - fail the pipeline
                            raise
                        else:
                            # Non-critical steps - continue pipeline
                            logger.warning(f"Continuing pipeline despite {step_name} failure")
                
                # Log pipeline summary
                duration = (datetime.now() - start_time).total_seconds()
                mlflow.log_metrics({
                    "pipeline.duration_seconds": duration,
                    "pipeline.successful_steps": len(successful_steps),
                    "pipeline.failed_steps": len(failed_steps)
                })
                
                pipeline_summary = {
                    "successful_steps": successful_steps,
                    "failed_steps": failed_steps,
                    "duration_seconds": duration,
                    "timestamp": datetime.now().isoformat()
                }
                
                with open("/app/pipeline_summary.json", "w") as f:
                    json.dump(pipeline_summary, f, indent=2)
                
                mlflow.log_artifact("/app/pipeline_summary.json")
                
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return 1
        
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Successful steps: {successful_steps}")
        if failed_steps:
            logger.warning(f"Failed steps: {failed_steps}")
        logger.info(f"Total duration: {(datetime.now() - start_time).total_seconds():.2f}s")
        
        return 0 if not failed_steps or all(step not in ["data_preprocessing", "feature_extraction", "model_training"] for step, _ in failed_steps) else 1

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="ML Pipeline for Potato Disease Classification")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--steps", nargs="+", help="Specific steps to run", 
                       choices=["data_preprocessing", "feature_extraction", "model_training", 
                               "model_evaluation", "model_registration", "cleanup"])
    parser.add_argument("--pipeline-version", type=str, default="local", help="Pipeline version")
    
    args = parser.parse_args()
    
    # Set environment variable for pipeline version
    os.environ["PIPELINE_VERSION"] = args.pipeline_version
    
    # Run pipeline
    pipeline = MLPipeline(args.config)
    return pipeline.run_pipeline(args.steps)

if __name__ == "__main__":
    sys.exit(main())