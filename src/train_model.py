"""
Model training module for the MLOps project.
Includes hyperparameter tuning and model tracking.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix, 
    precision_recall_curve, roc_curve, auc, f1_score, precision_score, recall_score
)
import time
import joblib

from segmentation import contour_detection, segment_image
from image_processing import contour, apply_clahe
from src.utils import calcul_dev
from mlflow_integration import MLflowManager

# For backward compatibility
from model_tracking import MLFlowLikeTracker


def process_image(image_path, output_dir="tmp"):
    """
    Process a single image through the full pipeline.
    
    Args:
        image_path: Path to the input image
        output_dir: Directory to store temporary files
        
    Returns:
        Processed image array and features
    """
    import os
    from PIL import Image
    import numpy as np
    import cv2
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the image
    with Image.open(image_path) as im:
        im_arr = np.asarray(im)
    
    # Filter the image to detect contours
    contours = contour(im_arr)
    filtered_path = os.path.join(output_dir, "filtered.png")
    cv2.imwrite(filtered_path, contours)
    
    # Segment the image
    cropped_path = os.path.join(output_dir, "cropped.png")
    segment_image(filtered_path, cropped_path, image_path)
    
    # Open the cropped image
    with Image.open(cropped_path) as im:
        im_arr = np.asarray(im)
    
    # Apply CLAHE for contrast enhancement
    enhanced_arr = apply_clahe(im_arr)
    
    # Calculate features
    features = calcul_dev(enhanced_arr)
    
    return enhanced_arr, features


def extract_features_from_directory(directory):
    """
    Extract features from all images in a directory.
    
    Args:
        directory: Path to directory containing images
        
    Returns:
        Dictionary of features and labels
    """
    features = []
    labels = []
    file_names = []
    
    # Get subdirectories (classes)
    classes = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    
    for class_name in classes:
        class_dir = os.path.join(directory, class_name)
        
        # Get image files in the class directory
        image_files = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f)) and 
                      f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"Processing class '{class_name}' with {len(image_files)} images")
        
        for image_file in image_files:
            image_path = os.path.join(class_dir, image_file)
            
            try:
                # Process the image and extract features
                _, image_features = process_image(image_path)
                
                # Append to lists
                features.append(image_features)
                labels.append(class_name)
                file_names.append(image_path)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
    
    return {
        "features": np.array(features),
        "labels": np.array(labels),
        "file_names": np.array(file_names)
    }


def evaluate_model(model, X_test, y_test, class_names, output_dir="visualizations"):
    """
    Evaluate a model and create visualizations.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        class_names: List of class names
        output_dir: Directory to save visualizations
        
    Returns:
        Dictionary of evaluation metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Generate classification report
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    
    # Overall metrics
    metrics = {
        "accuracy": accuracy,
        "weighted_f1": report["weighted avg"]["f1-score"],
        "weighted_precision": report["weighted avg"]["precision"],
        "weighted_recall": report["weighted avg"]["recall"],
        "per_class": report
    }
    
    # Save metrics as JSON
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix\nAccuracy: {accuracy:.4f}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()
    
    # Plot ROC curves for multi-class
    n_classes = len(class_names)
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i, class_name in enumerate(class_names):
        class_idx = list(model.classes_).index(i)  # Get the index in predict_proba output
        fpr[i], tpr[i], _ = roc_curve((y_test == i).astype(int), y_prob[:, class_idx])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot all ROC curves
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        plt.plot(fpr[i], tpr[i], label=f'{class_name} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, "roc_curves.png"))
    plt.close()
    
    return metrics


def perform_hyperparameter_tuning(X_train, y_train):
    """
    Perform hyperparameter tuning for the model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        Best model and parameters
    """
    print("Performing hyperparameter tuning...")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Create grid search
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit grid search
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_


def train_model(data_dir, output_dir="models/bin", visualizations_dir="visualizations", use_mlflow=True):
    """
    Train a model on the data.
    
    Args:
        data_dir: Path to directory containing training data
        output_dir: Directory to save trained model
        visualizations_dir: Directory to save visualizations
        use_mlflow: Whether to use MLflow for tracking (if False, uses legacy tracker)
        
    Returns:
        Trained model
    """
    # Initialize tracking
    if use_mlflow:
        tracker = MLflowManager()
        run_id = tracker.start_run("image_classification_training")
        is_mlflow = True
    else:
        tracker = MLFlowLikeTracker()
        run_id = tracker.start_run("image_classification_training")
        is_mlflow = False
    
    try:
        print(f"Starting model training run: {run_id}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(visualizations_dir, exist_ok=True)
        
        # Extract features from images
        print(f"Extracting features from {data_dir}")
        data = extract_features_from_directory(data_dir)
        
        X = data["features"]
        y = data["labels"]
        
        # Encode labels
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        
        # Save the encoder
        encoder_path = os.path.join(output_dir, "encoder.pkl")
        with open(encoder_path, "wb") as f:
            pickle.dump(encoder, f)
        
        # Log encoder as artifact
        if is_mlflow:
            tracker.log_artifact(encoder_path)
        else:
            tracker.log_artifact(run_id, "encoder.pkl", encoder_path)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.25, random_state=42
        )
        
        # Log parameters
        params = {
            "data_dir": data_dir,
            "n_samples": len(X),
            "n_features": X.shape[1],
            "n_classes": len(encoder.classes_),
            "test_size": 0.25,
            "random_state": 42,
            "classes": list(encoder.classes_)
        }
        
        if is_mlflow:
            tracker.log_params(params)
        else:
            tracker.log_params(run_id, params)
        
        # Perform hyperparameter tuning
        best_model, best_params = perform_hyperparameter_tuning(X_train, y_train)
        
        # Log best parameters
        if is_mlflow:
            tracker.log_params(best_params)
        else:
            tracker.log_params(run_id, best_params)
        
        # Train model on all training data with best parameters
        print("Training final model with best parameters")
        model = RandomForestClassifier(random_state=42, **best_params)
        model.fit(X_train, y_train)
        
        # Evaluate the model
        print("Evaluating model...")
        metrics = evaluate_model(
            model, X_test, y_test, 
            encoder.classes_, 
            output_dir=visualizations_dir
        )
        
        # Log evaluation metrics
        metrics_to_log = {}
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                metrics_to_log[metric_name] = metric_value
                if not is_mlflow:
                    tracker.log_metric(run_id, metric_name, metric_value)
        
        if is_mlflow:
            tracker.log_metrics(metrics_to_log)
        
        # Save the trained model
        model_path = os.path.join(output_dir, "model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        
        # Also save with joblib for better handling of large models
        joblib.dump(model, os.path.join(output_dir, "model.joblib"))
        
        # Log model
        if is_mlflow:
            mlflow_model_path = "model"
            tracker.log_model(model, mlflow_model_path, 
                             accuracy=metrics["accuracy"],
                             weighted_f1=metrics["weighted_f1"])
            
            # Register model in MLflow registry
            model_name = "potato_disease_classifier"
            model_version = tracker.register_model(run_id, mlflow_model_path, model_name)
            
            # Promote to production if it's better than existing models
            if model_version:
                tracker.promote_model(model_name, model_version, "Production")
        else:
            # Legacy tracking
            tracker.log_model(
                run_id, 
                model, 
                "model.pkl",
                accuracy=metrics["accuracy"],
                weighted_f1=metrics["weighted_f1"]
            )
        
        # Create feature importance visualization
        feature_importances = model.feature_importances_
        indices = np.argsort(feature_importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title("Feature Importances")
        plt.bar(range(X.shape[1]), feature_importances[indices], align="center")
        plt.xticks(range(X.shape[1]), indices)
        plt.xlim([-1, min(10, X.shape[1])])
        plt.tight_layout()
        
        # Save and log visualization
        feature_importance_path = os.path.join(visualizations_dir, "feature_importances.png")
        plt.savefig(feature_importance_path)
        if is_mlflow:
            tracker.log_artifact(feature_importance_path)
        plt.close()
        
        # Create a dictionary with models by feature
        models_by_features = {
            "model": model,
            "encoder": encoder,
            "features": X.shape[1],
            "classes": list(encoder.classes_),
            "metrics": metrics
        }
        
        # Save the models_by_features dictionary
        models_by_features_path = os.path.join(output_dir, "models_by_features.pkl")
        with open(models_by_features_path, "wb") as f:
            pickle.dump(models_by_features, f)
        
        # Log completion
        if is_mlflow:
            tracker.end_run()
        else:
            tracker.end_run(run_id, "FINISHED")
            # Promote model to production in legacy system
            tracker.promote_model_to_production(run_id, "model.pkl")
        
        print(f"Model training completed successfully. Results saved to {output_dir} and {visualizations_dir}")
        print(f"Best accuracy: {metrics['accuracy']:.4f}")
        
        return model, metrics
    
    except Exception as e:
        # Log failure
        if is_mlflow:
            tracker.log_param("error", str(e))
            tracker.end_run()
        else:
            tracker.log_param(run_id, "error", str(e))
            tracker.end_run(run_id, "FAILED")
        print(f"Error during model training: {e}")
        raise


if __name__ == "__main__":
    config_path = os.path.join(os.getcwd(), "configs", "config.json")
    
    # Load configuration
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Config file not found at {config_path}")
        config = {
            "paths": {
                "data": {"input": "data/inputs"},
                "models": {"directory": "models/bin"},
                "visualizations": "visualizations"
            }
        }
    
    data_dir = os.path.join(os.getcwd(), config["paths"]["data"]["input"])
    model_dir = os.path.join(os.getcwd(), config["paths"]["models"]["directory"])
    vis_dir = os.path.join(os.getcwd(), config["paths"]["visualizations"])
    
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        print("Please add data to the input directory before training.")
        sys.exit(1)
    
    train_model(data_dir, model_dir, vis_dir)
