"""
Configuration for MLflow tracing and evaluation logging
"""

import os
import sys
import json
import time
import yaml
import logging
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models.evaluation import evaluate

# Import scientific packages with error handling
try:
    import numpy as np
    import pandas as pd
except ImportError:
    logging.warning("NumPy or Pandas not available. Some features may not work.")
    np = None
    pd = None

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_mlflow_config():
    """
    Load MLflow configuration from mlflow_config.json
    """
    config_path = os.path.join(os.path.dirname(__file__), 'mlflow_config.json')
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config['mlflow']
    except Exception as e:
        logger.warning(f"Could not load MLflow config: {e}")
        return {
            "experiment_name": "potato-disease-classification",
            "tracking_uri": "sqlite:///mlflow.db",
            "artifact_location": "./mlruns"
        }

def enable_mlflow_tracing():
    """
    Enable MLflow traces for automatic logging
    """
    # Set the environment variable to enable MLflow tracing
    os.environ['MLFLOW_ENABLE_TRACES'] = 'true'
    
    # Load config
    config = load_mlflow_config()
    
    # Set tracking URI
    mlflow.set_tracking_uri(config['tracking_uri'])
    
    # Check if tracking URI is valid
    try:
        client = MlflowClient(tracking_uri=config['tracking_uri'])
        logger.info(f"Connected to MLflow tracking server at {config['tracking_uri']}")
    except Exception as e:
        logger.error(f"Failed to connect to MLflow tracking server: {e}")
        # Fallback to local tracking URI
        local_tracking_uri = "sqlite:///mlflow.db"
        logger.info(f"Falling back to local tracking URI: {local_tracking_uri}")
        mlflow.set_tracking_uri(local_tracking_uri)
    
    # Enable Auto Logging with appropriate settings
    mlflow.autolog(
        log_input_examples=True, 
        log_model_signatures=True,
        log_datasets=True,
        disable=False,
        exclusive=False,
        silent=False
    )
    
    # Get or create experiment with proper error handling
    experiment_name = config['experiment_name']
    try:
        # Check if experiment exists
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            # Create new experiment with artifact location
            artifact_location = config.get('artifact_location', './mlruns')
            experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=artifact_location
            )
            logger.info(f"Created new MLflow experiment: {experiment_name} with ID {experiment_id}")
        else:
            experiment_id = experiment.experiment_id
            logger.info(f"Using existing MLflow experiment: {experiment_name} with ID {experiment_id}")
              # Verify and ensure experiment structure
            ensure_experiment_structure(
                experiment_id, 
                experiment_name, 
                experiment.artifact_location
            )
    except Exception as e:
        logger.warning(f"Error setting up experiment: {e}")
        # Create default experiment as fallback
        try:
            default_experiment = mlflow.get_experiment_by_name("Default")
            if default_experiment:
                experiment_id = default_experiment.experiment_id
                logger.info(f"Using default experiment with ID: {experiment_id}")
            else:
                experiment_id = "0"  # Default MLflow experiment ID
        except:
            experiment_id = "0"
    
    logger.info("MLflow tracing and Auto Logging enabled")
    
    return experiment_id

def log_evaluation_metrics(model, X_test, y_test, evaluator_config=None, model_name=None):
    """
    Log evaluation metrics using MLflow's evaluation API
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        evaluator_config: Dictionary of evaluator configuration
        model_name: Name to give the model in the registry
        
    Returns:
        Dictionary of evaluation metrics
    """
    if evaluator_config is None:
        evaluator_config = {
            "accuracy_score": {"normalize": True}, 
            "precision_recall_fscore": {"average": "weighted"},
            "confusion_matrix": {},
            "roc_curve": {},
            "precision_recall_curve": {}
        }
    
    # Get the current run if one exists, else start a new one
    current_run = mlflow.active_run()
    if current_run is None:
        with mlflow.start_run() as run:
            # Evaluate and log metrics
            eval_results = evaluate(
                model=model,
                data=X_test,
                targets=y_test,
                model_type="classifier",
                evaluators=["default"],
                evaluator_config=evaluator_config
            )
            run_id = run.info.run_id
    else:
        # Evaluate and log metrics using the active run
        eval_results = evaluate(
            model=model,
            data=X_test,
            targets=y_test,
            model_type="classifier",
            evaluators=["default"],
            evaluator_config=evaluator_config
        )
        run_id = current_run.info.run_id
    
    # Register model if name is provided
    if model_name is not None:
        client = MlflowClient()
        
        try:
            # Check if model exists in registry
            try:
                client.get_registered_model(model_name)
                model_exists = True
            except:
                model_exists = False
            
            # Register model or create new version
            if not model_exists:
                mlflow.register_model(f"runs:/{run_id}/model", model_name)
                logger.info(f"Registered new model: {model_name}")
            else:
                new_version = mlflow.register_model(f"runs:/{run_id}/model", model_name)
                logger.info(f"Created new version of model {model_name}: {new_version.version}")
        
        except Exception as e:
            logger.error(f"Error registering model: {e}")
    
    return eval_results

def create_evaluation_tables(model, X_test, y_test, class_names, run_id=None):
    """
    Create evaluation tables for the MLflow UI with robust error handling
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        class_names: List of class names
        run_id: MLflow run ID
    """
    try:
        # Import required libraries within the function to ensure they're available
        try:
            import numpy as np
        except ImportError:
            logger.error("NumPy not available. Cannot create evaluation tables.")
            return None
            
        try:
            import pandas as pd
        except ImportError:
            logger.error("Pandas not available. Cannot create evaluation tables.")
            return None
            
        try:
            from sklearn.metrics import confusion_matrix, classification_report
        except ImportError:
            logger.error("Scikit-learn not available. Cannot create evaluation tables.")
            return None
        
        # Get probabilities and predictions
        y_proba = model.predict_proba(X_test)
        y_pred = model.predict(X_test)
        
        # Create prediction table
        pred_df = pd.DataFrame({
            'actual': [class_names[i] for i in y_test],
            'predicted': [class_names[i] for i in y_pred]
        })
        
        # Add probability columns
        for i, class_name in enumerate(class_names):
            pred_df[f'probability_{class_name}'] = y_proba[:, i]
        
        # Create confusion matrix table
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, columns=class_names, index=class_names)
        cm_df.index.name = 'Actual'
        cm_df.columns.name = 'Predicted'
        
        # Get classification report
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        
        # Log tables with multiple fallback strategies
        if run_id:
            try:
                client = MlflowClient()
                
                # Try method 1: Direct client logging
                try:
                    client.log_table(run_id=run_id, 
                                    data=pred_df, 
                                    artifact_file="predictions.json")
                    client.log_table(run_id=run_id, 
                                    data=cm_df.reset_index(), 
                                    artifact_file="confusion_matrix.json")
                    client.log_table(run_id=run_id, 
                                    data=report_df.reset_index().rename(columns={'index': 'class'}), 
                                    artifact_file="classification_report.json")
                    logger.info("Evaluation tables logged using client method")
                    
                except Exception as e:
                    logger.warning(f"Client logging failed: {e}. Trying mlflow.log_table...")
                    
                    # Try method 2: Using active run context
                    try:
                        with mlflow.start_run(run_id=run_id):
                            mlflow.log_table(pred_df, "predictions.json")
                            mlflow.log_table(cm_df.reset_index(), "confusion_matrix.json")
                            mlflow.log_table(report_df.reset_index().rename(columns={'index': 'class'}), 
                                           "classification_report.json")
                        logger.info("Evaluation tables logged using mlflow.log_table method")
                        
                    except Exception as e2:
                        logger.warning(f"MLflow log_table failed: {e2}. Trying CSV artifacts...")
                        
                        # Try method 3: Log as CSV artifacts
                        try:
                            # Create temp directory
                            import tempfile
                            with tempfile.TemporaryDirectory() as temp_dir:
                                pred_csv = os.path.join(temp_dir, "predictions.csv")
                                cm_csv = os.path.join(temp_dir, "confusion_matrix.csv")
                                report_csv = os.path.join(temp_dir, "classification_report.csv")
                                
                                pred_df.to_csv(pred_csv, index=False)
                                cm_df.to_csv(cm_csv)
                                report_df.to_csv(report_csv)
                                
                                with mlflow.start_run(run_id=run_id):
                                    mlflow.log_artifact(pred_csv)
                                    mlflow.log_artifact(cm_csv)
                                    mlflow.log_artifact(report_csv)
                                    
                            logger.info("Evaluation tables logged as CSV artifacts")
                            
                        except Exception as e3:
                            logger.error(f"All table logging methods failed: {e3}")
                            
            except Exception as e:
                logger.error(f"Error logging tables with run_id {run_id}: {e}")
        else:
            # Create new run and log tables
            try:
                with mlflow.start_run() as run:
                    mlflow.log_table(pred_df, "predictions.json")
                    mlflow.log_table(cm_df.reset_index(), "confusion_matrix.json")
                    mlflow.log_table(report_df.reset_index().rename(columns={'index': 'class'}), 
                                   "classification_report.json")
                    logger.info("Evaluation tables logged in new run")
            except Exception as e:
                logger.error(f"Error logging tables in new run: {e}")
                
        return {
            "predictions": pred_df,
            "confusion_matrix": cm_df,
            "classification_report": report_df
        }
        
    except Exception as e:
        logger.error(f"Error in create_evaluation_tables: {e}")
        return None

def ensure_experiment_structure(experiment_id, experiment_name, artifact_location="./mlruns"):
    """
    Ensure experiment has proper structure including meta.yaml files
    """
    try:
        experiment_dir = os.path.join('./mlruns', experiment_id)
        meta_path = os.path.join(experiment_dir, 'meta.yaml')
        
        if not os.path.exists(meta_path):
            logger.warning(f"Meta file missing for experiment {experiment_id}. Recreating...")
            
            # Create directory if it doesn't exist
            os.makedirs(experiment_dir, exist_ok=True)
            
            # Create meta.yaml file
            meta_data = {
                'artifact_location': artifact_location,
                'experiment_id': experiment_id,
                'name': experiment_name,
                'creation_time': int(time.time() * 1000),
                'last_update_time': int(time.time() * 1000)
            }
            
            with open(meta_path, 'w') as f:
                yaml.safe_dump(meta_data, f)
                
            logger.info(f"Recreated meta.yaml for experiment {experiment_id}")
            return True
        else:
            logger.info(f"Meta.yaml exists for experiment {experiment_id}")
            return True
            
    except Exception as e:
        logger.error(f"Error ensuring experiment structure: {e}")
        return False
