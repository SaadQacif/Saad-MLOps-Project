"""
Model tracking and versioning module.
Provides functionality to track model versions, metrics, and metadata.
"""

import os
import json
import uuid
import datetime
import pickle
from typing import Dict, Any, Optional, List, Tuple


class MLFlowLikeTracker:
    """
    A lightweight model tracking system similar to MLFlow but without external dependencies.
    Stores model metadata, parameters, metrics, and artifacts.
    """
    def __init__(self, tracking_dir: str = None):
        """
        Initialize the tracker.
        
        Args:
            tracking_dir: Directory to store tracking information
        """
        if tracking_dir is None:
            tracking_dir = os.path.join(os.getcwd(), "models", "tracking")
        
        self.tracking_dir = tracking_dir
        os.makedirs(tracking_dir, exist_ok=True)
        
        # Load existing runs
        self.runs_index_path = os.path.join(tracking_dir, "runs_index.json")
        self.runs_index = self._load_runs_index()
    
    def _load_runs_index(self) -> Dict:
        """Load the runs index or create a new one if it doesn't exist."""
        if os.path.exists(self.runs_index_path):
            with open(self.runs_index_path, "r") as f:
                return json.load(f)
        return {"runs": {}}
    
    def _save_runs_index(self):
        """Save the runs index to disk."""
        with open(self.runs_index_path, "w") as f:
            json.dump(self.runs_index, f, indent=2)
    
    def start_run(self, run_name: Optional[str] = None) -> str:
        """
        Start a new tracking run.
        
        Args:
            run_name: Optional name for the run
            
        Returns:
            run_id: Unique ID for the run
        """
        run_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().isoformat()
        
        if run_name is None:
            run_name = f"run_{timestamp}"
        
        run_dir = os.path.join(self.tracking_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(os.path.join(run_dir, "artifacts"), exist_ok=True)
        
        # Update the index
        self.runs_index["runs"][run_id] = {
            "run_name": run_name,
            "status": "RUNNING",
            "start_time": timestamp,
            "end_time": None,
            "params": {},
            "metrics": {},
            "tags": {},
            "artifacts": []
        }
        
        self._save_runs_index()
        return run_id
    
    def end_run(self, run_id: str, status: str = "FINISHED"):
        """
        End a tracking run.
        
        Args:
            run_id: ID of the run to end
            status: Status of the run (FINISHED, FAILED, etc.)
        """
        if run_id not in self.runs_index["runs"]:
            raise ValueError(f"Run {run_id} not found")
        
        self.runs_index["runs"][run_id]["status"] = status
        self.runs_index["runs"][run_id]["end_time"] = datetime.datetime.now().isoformat()
        self._save_runs_index()
    
    def log_param(self, run_id: str, key: str, value: Any):
        """
        Log a parameter for a run.
        
        Args:
            run_id: ID of the run
            key: Parameter name
            value: Parameter value
        """
        if run_id not in self.runs_index["runs"]:
            raise ValueError(f"Run {run_id} not found")
        
        self.runs_index["runs"][run_id]["params"][key] = value
        self._save_runs_index()
    
    def log_params(self, run_id: str, params: Dict[str, Any]):
        """
        Log multiple parameters for a run.
        
        Args:
            run_id: ID of the run
            params: Dictionary of parameter names and values
        """
        for key, value in params.items():
            self.log_param(run_id, key, value)
    
    def log_metric(self, run_id: str, key: str, value: float):
        """
        Log a metric for a run.
        
        Args:
            run_id: ID of the run
            key: Metric name
            value: Metric value
        """
        if run_id not in self.runs_index["runs"]:
            raise ValueError(f"Run {run_id} not found")
        
        if key not in self.runs_index["runs"][run_id]["metrics"]:
            self.runs_index["runs"][run_id]["metrics"][key] = []
        
        timestamp = datetime.datetime.now().isoformat()
        self.runs_index["runs"][run_id]["metrics"][key].append({
            "timestamp": timestamp,
            "value": value
        })
        
        self._save_runs_index()
    
    def log_artifact(self, run_id: str, artifact_path: str, local_path: str) -> str:
        """
        Log an artifact for a run.
        
        Args:
            run_id: ID of the run
            artifact_path: Path to store the artifact
            local_path: Local path of the artifact
            
        Returns:
            Path to the stored artifact
        """
        if run_id not in self.runs_index["runs"]:
            raise ValueError(f"Run {run_id} not found")
        
        run_dir = os.path.join(self.tracking_dir, run_id)
        artifact_dir = os.path.join(run_dir, "artifacts")
        
        # Create subdirectories if needed
        artifact_parts = artifact_path.split("/")
        if len(artifact_parts) > 1:
            subdir_path = os.path.join(artifact_dir, *artifact_parts[:-1])
            os.makedirs(subdir_path, exist_ok=True)
        
        target_path = os.path.join(artifact_dir, artifact_path)
        
        # Copy the artifact
        import shutil
        shutil.copy2(local_path, target_path)
        
        # Update the index
        self.runs_index["runs"][run_id]["artifacts"].append(artifact_path)
        self._save_runs_index()
        
        return target_path
    
    def log_model(self, run_id: str, model, artifact_path: str = "model.pkl", **kwargs):
        """
        Log a model for a run.
        
        Args:
            run_id: ID of the run
            model: Model object to log
            artifact_path: Path to store the model
            **kwargs: Additional metadata to store with the model
        """
        if run_id not in self.runs_index["runs"]:
            raise ValueError(f"Run {run_id} not found")
        
        run_dir = os.path.join(self.tracking_dir, run_id)
        artifact_dir = os.path.join(run_dir, "artifacts")
        
        # Create subdirectories if needed
        artifact_parts = artifact_path.split("/")
        if len(artifact_parts) > 1:
            subdir_path = os.path.join(artifact_dir, *artifact_parts[:-1])
            os.makedirs(subdir_path, exist_ok=True)
        
        # Save the model
        model_path = os.path.join(artifact_dir, artifact_path)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        
        # Save metadata
        metadata_path = os.path.join(artifact_dir, f"{artifact_path}.meta.json")
        metadata = {
            "timestamp": datetime.datetime.now().isoformat(),
            "type": type(model).__name__,
            **kwargs
        }
        
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Update the index
        self.runs_index["runs"][run_id]["artifacts"].extend([artifact_path, f"{artifact_path}.meta.json"])
        self._save_runs_index()
    
    def get_run(self, run_id: str) -> Dict[str, Any]:
        """
        Get information about a run.
        
        Args:
            run_id: ID of the run
            
        Returns:
            Run information
        """
        if run_id not in self.runs_index["runs"]:
            raise ValueError(f"Run {run_id} not found")
        
        return self.runs_index["runs"][run_id]
    
    def list_runs(self) -> List[Dict[str, Any]]:
        """
        List all runs.
        
        Returns:
            List of run information
        """
        return [
            {
                "run_id": run_id,
                **run_info
            }
            for run_id, run_info in self.runs_index["runs"].items()
        ]
    
    def get_best_run(self, metric: str, mode: str = "max") -> Tuple[str, Dict[str, Any]]:
        """
        Get the best run based on a metric.
        
        Args:
            metric: Metric to use for comparison
            mode: 'max' for higher is better, 'min' for lower is better
            
        Returns:
            run_id and run information of the best run
        """
        runs = self.list_runs()
        valid_runs = []
        
        for run in runs:
            if (
                metric in run["metrics"] and 
                len(run["metrics"][metric]) > 0 and
                run["status"] == "FINISHED"
            ):
                # Get the latest metric value
                value = run["metrics"][metric][-1]["value"]
                valid_runs.append((run["run_id"], run, value))
        
        if not valid_runs:
            raise ValueError(f"No runs found with metric {metric}")
        
        if mode == "max":
            best_run = max(valid_runs, key=lambda x: x[2])
        else:
            best_run = min(valid_runs, key=lambda x: x[2])
        
        return best_run[0], best_run[1]
    
    def load_model(self, run_id: str, artifact_path: str = "model.pkl"):
        """
        Load a model from a run.
        
        Args:
            run_id: ID of the run
            artifact_path: Path of the model in the run artifacts
            
        Returns:
            The loaded model
        """
        if run_id not in self.runs_index["runs"]:
            raise ValueError(f"Run {run_id} not found")
        
        model_path = os.path.join(self.tracking_dir, run_id, "artifacts", artifact_path)
        
        if not os.path.exists(model_path):
            raise ValueError(f"Model not found at {model_path}")
        
        with open(model_path, "rb") as f:
            return pickle.load(f)
    
    def promote_model_to_production(self, run_id: str, artifact_path: str = "model.pkl", target_path: str = None):
        """
        Promote a model to production.
        
        Args:
            run_id: ID of the run
            artifact_path: Path of the model in the run artifacts
            target_path: Target path to copy the model to
            
        Returns:
            Path to the production model
        """
        if target_path is None:
            target_path = os.path.join(os.getcwd(), "models", "bin", "production_model.pkl")
        
        model = self.load_model(run_id, artifact_path)
        
        # Create target directory if needed
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        
        # Save the model
        with open(target_path, "wb") as f:
            pickle.dump(model, f)
        
        # Create production metadata
        meta_path = f"{target_path}.meta.json"
        metadata = {
            "timestamp": datetime.datetime.now().isoformat(),
            "source_run_id": run_id,
            "artifact_path": artifact_path,
            "type": type(model).__name__
        }
        
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        return target_path
