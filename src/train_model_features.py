"""
Train a model using extracted features and MLflow tracking.
"""

import os
import sys
import json
import pickle
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, auc
)
import mlflow
from mlflow.tracking import MlflowClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from JSON file."""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.json')
    with open(config_path, 'r') as f:
        return json.load(f)

def load_mlflow_config():
    """Load MLflow configuration from JSON file."""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'mlflow_config.json')
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # Return default config
        return {
            "mlflow": {
                "experiment_name": "potato-disease-classification",
                "tracking_uri": "sqlite:///mlflow.db",
                "artifact_location": "./mlruns",
                "registry_uri": "sqlite:///mlflow.db",
                "tags": {
                    "project": "potato-disease-classification",
                    "env": "local"
                }
            }
        }

def visualize_results(model, X_test, y_test, class_names, output_dir, feature_names=None):
    """
    Create and save visualizations for model results.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        class_names: List of class names
        output_dir: Directory to save visualizations
        feature_names: List of feature names
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Feature importance
    if hasattr(model, 'feature_importances_') and feature_names is not None:
        # Sort features by importance
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Plot top 15 features
        top_n = min(15, len(feature_names))
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importances')
        plt.bar(range(top_n), importances[indices[:top_n]], align='center')
        plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importances.png'))
        plt.close()
    
    # ROC curves for each class
    n_classes = len(class_names)
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(
            (y_test == i).astype(int), 
            y_proba[:, i]
        )
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    for i, color, cls in zip(range(n_classes), ['blue', 'red', 'green'], class_names):
        plt.plot(
            fpr[i], tpr[i], color=color, lw=2,
            label=f'ROC curve of class {cls} (area = {roc_auc[i]:.2f})'
        )
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'))
    plt.close()

def evaluate_model(model, X_test, y_test, class_names):
    """
    Evaluate model performance.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        class_names: List of class names
    
    Returns:
        Dictionary of evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    weighted_precision = precision_score(y_test, y_pred, average='weighted')
    weighted_recall = recall_score(y_test, y_pred, average='weighted')
    
    # Calculate per-class metrics
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    
    # Calculate macro ROC AUC
    try:
        roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
    except Exception:
        roc_auc = 0.0
    
    return {
        'accuracy': accuracy,
        'weighted_f1': weighted_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'roc_auc': roc_auc,
        'per_class_metrics': report
    }

def train_model_with_mlflow(
    features_file,
    model_dir="models/bin",
    output_dir="visualizations",
    tracking_uri=None
):
    """
    Train model using extracted features and track with MLflow.
    
    Args:
        features_file: Path to CSV file with extracted features
        model_dir: Directory to save the trained model
        output_dir: Directory to save visualizations
        tracking_uri: MLflow tracking URI
    
    Returns:
        Trained model and evaluation metrics
    """
    # Ensure directories exist
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load MLflow configuration
    mlflow_config = load_mlflow_config()
    
    # Set up MLflow
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        mlflow.set_tracking_uri(mlflow_config["mlflow"]["tracking_uri"])
    
    # Create experiment if it doesn't exist
    experiment_name = mlflow_config["mlflow"]["experiment_name"]
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                name=experiment_name,
                artifact_location=mlflow_config["mlflow"]["artifact_location"]
            )
        else:
            experiment_id = experiment.experiment_id
    except Exception as e:
        logger.error(f"Error setting up MLflow experiment: {e}")
        experiment_id = 0  # Use default experiment
    
    # Load features
    try:
        df = pd.read_csv(features_file)
        logger.info(f"Loaded {len(df)} samples from {features_file}")
    except Exception as e:
        logger.error(f"Error loading features from {features_file}: {e}")
        return None, None
    
    # Check if the class column exists
    if 'class' not in df.columns:
        logger.error("No 'class' column found in features file")
        return None, None
    
    # Remove file path column if it exists
    if 'image_path' in df.columns:
        df = df.drop(columns=['image_path'])
    
    # Handle missing values
    df = df.dropna()
    
    # Separate features and target
    X = df.drop(columns=['class'])
    y = df['class']
    
    # Get feature names
    feature_names = X.columns.tolist()
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_names = label_encoder.classes_
    
    logger.info(f"Classes: {class_names}")
    logger.info(f"Feature count: {X.shape[1]}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Start MLflow run
    with mlflow.start_run(experiment_id=experiment_id, run_name="potato_disease_classifier") as run:
        # Get run ID
        run_id = run.info.run_id
        logger.info(f"MLflow run ID: {run_id}")
        
        # Set tags
        mlflow.set_tags(mlflow_config["mlflow"]["tags"])
        
        # Log dataset info
        mlflow.log_params({
            "features_file": features_file,
            "n_samples": len(df),
            "n_features": X.shape[1],
            "n_classes": len(class_names),
            "classes": ", ".join(class_names)
        })
        
        # Define models to try
        models = {
            "RandomForest": {
                "model": RandomForestClassifier(random_state=42),
                "params": {
                    "n_estimators": [100, 200],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5]
                }
            },
            "GradientBoosting": {
                "model": GradientBoostingClassifier(random_state=42),
                "params": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [3, 5]
                }
            }
        }
        
        # Train and evaluate models
        best_model = None
        best_score = 0.0
        best_metrics = None
        
        for name, config in models.items():
            logger.info(f"Training {name}...")
            
            # Grid search
            grid_search = GridSearchCV(
                config["model"],
                config["params"],
                cv=5,
                scoring="accuracy",
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train_scaled, y_train)
            
            # Get best estimator
            model = grid_search.best_estimator_
            
            # Evaluate model
            metrics = evaluate_model(model, X_test_scaled, y_test, class_names)
            
            # Log model parameters
            mlflow.log_params({
                f"{name}_best_params": grid_search.best_params_,
                f"{name}_best_cv_score": grid_search.best_score_
            })
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(f"{name}_{metric_name}", metric_value)
            
            # Check if this model is better
            if metrics['accuracy'] > best_score:
                best_model = model
                best_score = metrics['accuracy']
                best_metrics = metrics
                best_model_name = name
        
        # Log best model
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            registered_model_name="potato_disease_classifier"
        )
        
        # Log feature importance for the best model
        if hasattr(best_model, 'feature_importances_'):
            # Create feature importance plot
            top_features = min(20, len(feature_names))
            plt.figure(figsize=(12, 8))
            importances = best_model.feature_importances_
            indices = np.argsort(importances)[::-1][:top_features]
            plt.title('Top Feature Importances')
            plt.bar(range(top_features), importances[indices], align='center')
            plt.xticks(range(top_features), [feature_names[i] for i in indices], rotation=90)
            plt.tight_layout()
            
            # Save and log
            importance_path = os.path.join(output_dir, 'feature_importance.png')
            plt.savefig(importance_path)
            mlflow.log_artifact(importance_path)
            
            # Log feature importance values
            for i, idx in enumerate(indices[:top_features]):
                mlflow.log_metric(f"feature_importance_{feature_names[idx]}", importances[idx])
        
        # Create visualizations
        visualize_results(
            best_model,
            X_test_scaled,
            y_test,
            class_names,
            output_dir,
            feature_names
        )
        
        # Log confusion matrix
        confusion_matrix_path = os.path.join(output_dir, 'confusion_matrix.png')
        if os.path.exists(confusion_matrix_path):
            mlflow.log_artifact(confusion_matrix_path)
        
        # Log ROC curves
        roc_curves_path = os.path.join(output_dir, 'roc_curves.png')
        if os.path.exists(roc_curves_path):
            mlflow.log_artifact(roc_curves_path)
        
        # Save best model
        model_path = os.path.join(model_dir, f"{best_model_name}_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)
        
        # Save label encoder
        encoder_path = os.path.join(model_dir, "label_encoder.pkl")
        with open(encoder_path, 'wb') as f:
            pickle.dump(label_encoder, f)
        
        # Save scaler
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        # Save model metadata
        metadata = {
            "model_type": best_model_name,
            "features": feature_names,
            "classes": class_names.tolist(),
            "metrics": {k: v for k, v in best_metrics.items() if isinstance(v, (int, float))},
            "feature_importance": best_model.feature_importances_.tolist() if hasattr(best_model, 'feature_importances_') else None
        }
        
        metadata_path = os.path.join(model_dir, "model_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Log important artifacts
        mlflow.log_artifact(model_path)
        mlflow.log_artifact(encoder_path)
        mlflow.log_artifact(scaler_path)
        mlflow.log_artifact(metadata_path)
        
        logger.info(f"Best model: {best_model_name}, Accuracy: {best_score:.4f}")
        logger.info(f"Model saved to {model_path}")
    
    return best_model, best_metrics

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train a model using extracted features")
    parser.add_argument("--features", type=str, required=True, help="Path to CSV file with extracted features")
    parser.add_argument("--model-dir", type=str, default="models/bin", help="Directory to save the trained model")
    parser.add_argument("--output-dir", type=str, default="visualizations", help="Directory to save visualizations")
    parser.add_argument("--mlflow-tracking-uri", type=str, help="MLflow tracking URI")
    
    args = parser.parse_args()
    
    # Train model
    model, metrics = train_model_with_mlflow(
        args.features,
        args.model_dir,
        args.output_dir,
        args.mlflow_tracking_uri
    )
    
    # Check if training was successful
    if model is None:
        logger.error("Training failed")
        return 1
    
    logger.info("Training completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())
