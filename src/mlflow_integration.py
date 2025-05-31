"""
MLflow integration module for the MLOps project.
Provides functionality to track experiments, log metrics, and register models using MLflow.
"""

import os
import json
import mlflow
from mlflow.tracking import MlflowClient
import pickle
from typing import Dict, Any, Optional, List, Tuple
import logging
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLflowManager:
    """
    A class for managing MLflow tracking and model registry functionality.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the MLflow manager.
        
        Args:
            config_path: Path to the MLflow configuration file
        """
        # Load configuration
        if config_path is None:
            config_path = os.path.join(os.getcwd(), "configs", "mlflow_config.json")
        
        try:
            with open(config_path, "r") as f:
                self.config = json.load(f)["mlflow"]
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            logger.warning(f"Failed to load MLflow config from {config_path}. Using default config.")
            self.config = {
                "experiment_name": "potato-disease-classification",
                "tracking_uri": "sqlite:///mlflow.db",
                "artifact_location": "./mlruns",
                "registry_uri": "sqlite:///mlflow.db",
                "tags": {
                    "project": "potato-disease-classification",
                    "env": "local"
                },
                "server": {
                    "host": "0.0.0.0",
                    "port": 5001,
                    "backend_store_uri": "sqlite:///mlflow.db",
                    "default_artifact_root": "./mlruns"
                }
            }
        
        # Set up MLflow tracking URI
        mlflow.set_tracking_uri(self.config["tracking_uri"])
        
        # Create or get experiment
        experiment = mlflow.get_experiment_by_name(self.config["experiment_name"])
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(
                name=self.config["experiment_name"],
                artifact_location=self.config["artifact_location"]
            )
        else:
            self.experiment_id = experiment.experiment_id
        
        self.client = MlflowClient()
    
    def start_run(self, run_name: str = None) -> str:
        """
        Start a new MLflow run.
        
        Args:
            run_name: Optional name for the run
            
        Returns:
            The run ID of the created run
        """
        run = mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name
        )
        
        # Set default tags
        mlflow.set_tags(self.config["tags"])
        
        return run.info.run_id
    
    def end_run(self):
        """End the current MLflow run."""
        mlflow.end_run()
    
    def log_param(self, key: str, value: Any):
        """
        Log a parameter to the current run.
        
        Args:
            key: Parameter name
            value: Parameter value
        """
        mlflow.log_param(key, value)
    
    def log_params(self, params: Dict[str, Any]):
        """
        Log multiple parameters to the current run.
        
        Args:
            params: Dictionary of parameter names and values
        """
        mlflow.log_params(params)
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """
        Log a metric to the current run.
        
        Args:
            key: Metric name
            value: Metric value
            step: Optional step value
        """
        mlflow.log_metric(key, value, step=step)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log multiple metrics to the current run.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step value
        """
        mlflow.log_metrics(metrics, step=step)
    
    def log_artifact(self, local_path: str):
        """
        Log an artifact to the current run.
        
        Args:
            local_path: Local path to the artifact
        """
        mlflow.log_artifact(local_path)
    
    def log_artifacts(self, local_dir: str):
        """
        Log artifacts to the current run.
        
        Args:
            local_dir: Local directory containing artifacts
        """
        mlflow.log_artifacts(local_dir)
    
    def log_figure(self, figure, artifact_path: str):
        """
        Log a matplotlib figure to the current run.
        
        Args:
            figure: Matplotlib figure object
            artifact_path: Path within the artifact directory
        """
        mlflow.log_figure(figure, artifact_path)
    
    def log_model(self, model, artifact_path: str, **kwargs):
        """
        Log a model to the current run.
        
        Args:
            model: Model to log
            artifact_path: Path within the artifact directory
            kwargs: Additional keyword arguments to log as tags
        """
        # Create temporary file to save the model
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            pickle.dump(model, f)
            model_path = f.name
        
        # Log the model as an artifact
        mlflow.log_artifact(model_path, artifact_path)
        
        # Remove temporary file
        os.unlink(model_path)
        
        # Log additional tags
        for key, value in kwargs.items():
            mlflow.set_tag(f"model.{key}", value)
    
    def register_model(self, run_id: str, artifact_path: str, name: str):
        """
        Register a model from a run into the model registry.
        
        Args:
            run_id: ID of the run containing the model
            artifact_path: Path to the model within the run's artifacts
            name: Name to register the model under
            
        Returns:
            Model version
        """
        try:
            model_uri = f"runs:/{run_id}/{artifact_path}"
            result = mlflow.register_model(model_uri, name)
            return result.version
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            return None
    
    def promote_model(self, name: str, version: int, stage: str):
        """
        Promote a model to a new stage in the registry.
        
        Args:
            name: Name of the registered model
            version: Version of the model
            stage: Target stage ('Staging', 'Production', 'Archived')
        """
        try:
            self.client.transition_model_version_stage(
                name=name,
                version=version,
                stage=stage
            )
            logger.info(f"Model {name} version {version} promoted to {stage}")
        except Exception as e:
            logger.error(f"Failed to promote model: {e}")
    
    def load_model(self, name: str, stage: str = "Production"):
        """
        Load a model from the registry.
        
        Args:
            name: Name of the registered model
            stage: Stage to load from ('Staging', 'Production', etc.)
            
        Returns:
            The loaded model
        """
        try:
            model_uri = f"models:/{name}/{stage}"
            return mlflow.sklearn.load_model(model_uri)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None
