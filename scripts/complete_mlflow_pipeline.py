#!/usr/bin/env python
"""
Complete MLOps Pipeline with MLflow Integration and Tracing
This script runs the entire pipeline from data preprocessing to model training
with comprehensive MLflow experiment tracking, dataset logging, and visualization.
"""

import os
import sys
import json
import pickle
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import tempfile
import shutil

# ML libraries
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, auc
)

# Computer vision
import cv2
from PIL import Image

# MLflow imports
import mlflow
import mlflow.sklearn
import mlflow.data
from mlflow.data.pandas_dataset import PandasDataset
from mlflow.tracking import MlflowClient

# Suppress warnings
warnings.filterwarnings('ignore')

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import custom modules
try:
    from scripts.extract_potato_features import extract_features, get_feature_names
except ImportError:
    print("Warning: Could not import feature extraction functions")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MLflowPotateoPipeline:
    """Complete MLflow-integrated pipeline for potato disease classification"""
    
    def __init__(self, 
                 data_dir: str = "Potato_Health_States",
                 models_dir: str = "models",
                 experiment_name: str = "Potato_Disease_Classification_Complete"):
        """
        Initialize the pipeline
        
        Args:
            data_dir: Directory containing the dataset
            models_dir: Directory to save models
            experiment_name: MLflow experiment name
        """
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.experiment_name = experiment_name
        
        # Create directories
        self.models_dir.mkdir(exist_ok=True)
        
        # MLflow setup
        self.setup_mlflow()
        
        # Model configurations
        self.model_configs = {
            'RandomForest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'SVM': {
                'model': SVC(random_state=42, probability=True),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                }
            },
            'MLP': {
                'model': MLPClassifier(random_state=42, max_iter=1000),
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive']
                }
            }
        }
        
        # Results storage
        self.results = {}
        self.dataset_info = {}
        
    def setup_mlflow(self):
        """Setup MLflow experiment and tracking"""
        try:
            # Set MLflow tracking URI to relative path
            mlflow.set_tracking_uri("file:./mlruns")
            
            # Enable autologging for sklearn
            mlflow.sklearn.autolog(
                log_input_examples=True,
                log_model_signatures=True,
                log_models=True,
                disable=False,
                exclusive=False,
                disable_for_unsupported_versions=False,
                silent=False
            )
            
            # Create or get experiment
            try:
                experiment = mlflow.get_experiment_by_name(self.experiment_name)
                if experiment is None:
                    self.experiment_id = mlflow.create_experiment(self.experiment_name)
                    logger.info(f"Created new experiment: {self.experiment_name}")
                else:
                    self.experiment_id = experiment.experiment_id
                    logger.info(f"Using existing experiment: {self.experiment_name}")
            except Exception as e:
                logger.warning(f"Could not setup experiment: {e}")
                self.experiment_id = None
                
        except Exception as e:
            logger.error(f"MLflow setup failed: {e}")
            self.experiment_id = None
    
    @mlflow.trace(name="data_preprocessing")
    def preprocess_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Preprocess the dataset and extract features with MLflow tracing
        
        Returns:
            Features array, labels array, and image paths
        """
        logger.info("Starting data preprocessing...")
        
        # Initialize storage
        features_list = []
        labels_list = []
        paths_list = []
        
        # Get class directories
        class_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        logger.info(f"Found {len(class_dirs)} classes: {[d.name for d in class_dirs]}")
        
        # Store dataset info for MLflow
        self.dataset_info = {
            'total_classes': len(class_dirs),
            'class_names': [d.name for d in class_dirs],
            'samples_per_class': {}
        }
        
        # Process each class
        for class_dir in class_dirs:
            class_name = class_dir.name
            logger.info(f"Processing class: {class_name}")
            
            # Get image files
            image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.png"))
            self.dataset_info['samples_per_class'][class_name] = len(image_files)
            
            logger.info(f"Found {len(image_files)} images in {class_name}")
            
            # Process images
            for img_path in image_files:
                try:
                    # Load image
                    image = cv2.imread(str(img_path))
                    if image is None:
                        continue
                    
                    # Extract features
                    features = extract_features(image)
                    if features is not None and len(features) > 0:
                        features_list.append(features)
                        labels_list.append(class_name)
                        paths_list.append(str(img_path))
                        
                except Exception as e:
                    logger.warning(f"Error processing {img_path}: {e}")
                    continue
        
        # Convert to arrays
        X = np.array(features_list)
        y = np.array(labels_list)
        
        # Update dataset info
        self.dataset_info['total_samples'] = len(X)
        self.dataset_info['feature_dimensions'] = X.shape[1] if len(X) > 0 else 0
        
        logger.info(f"Preprocessing complete. Shape: {X.shape}")
        logger.info(f"Dataset info: {self.dataset_info}")
        
        return X, y, paths_list
    
    @mlflow.trace(name="create_mlflow_dataset")
    def create_mlflow_dataset(self, X: np.ndarray, y: np.ndarray, paths: List[str]) -> PandasDataset:
        """
        Create and log MLflow dataset
        
        Args:
            X: Features array
            y: Labels array
            paths: Image paths
            
        Returns:
            MLflow dataset
        """
        logger.info("Creating MLflow dataset...")
        
        # Create DataFrame
        feature_names = get_feature_names()
        if len(feature_names) != X.shape[1]:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        df = pd.DataFrame(X, columns=feature_names)
        df['label'] = y
        df['image_path'] = paths
        
        # Create MLflow dataset
        dataset = mlflow.data.from_pandas(
            df,
            source=str(self.data_dir),
            name="potato_disease_dataset",
            targets="label"
        )
        
        return dataset
    
    @mlflow.trace(name="visualize_dataset")
    def visualize_dataset(self, X: np.ndarray, y: np.ndarray) -> Dict[str, str]:
        """
        Create dataset visualizations and log to MLflow
        
        Args:
            X: Features array
            y: Labels array
            
        Returns:
            Dictionary of visualization file paths
        """
        logger.info("Creating dataset visualizations...")
        
        viz_dir = Path("visualizations")
        viz_dir.mkdir(exist_ok=True)
        
        visualization_paths = {}
        
        # Class distribution
        plt.figure(figsize=(10, 6))
        unique, counts = np.unique(y, return_counts=True)
        plt.bar(unique, counts)
        plt.title('Dataset Class Distribution')
        plt.xlabel('Disease Class')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        dist_path = viz_dir / "class_distribution.png"
        plt.savefig(dist_path, dpi=300, bbox_inches='tight')
        plt.close()
        visualization_paths['class_distribution'] = str(dist_path)
        
        # Feature correlation heatmap
        if X.shape[1] <= 50:  # Only for reasonable number of features
            plt.figure(figsize=(12, 10))
            feature_names = get_feature_names()[:X.shape[1]]
            df_features = pd.DataFrame(X, columns=feature_names)
            
            correlation_matrix = df_features.corr()
            sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            
            corr_path = viz_dir / "feature_correlation.png"
            plt.savefig(corr_path, dpi=300, bbox_inches='tight')
            plt.close()
            visualization_paths['feature_correlation'] = str(corr_path)
        
        # Feature distributions by class
        if X.shape[1] <= 20:  # Only for small number of features
            fig, axes = plt.subplots(4, 5, figsize=(20, 16))
            axes = axes.ravel()
            
            feature_names = get_feature_names()[:X.shape[1]]
            
            for i in range(min(20, X.shape[1])):
                for j, class_name in enumerate(np.unique(y)):
                    class_data = X[y == class_name, i]
                    axes[i].hist(class_data, alpha=0.7, label=class_name, bins=20)
                
                axes[i].set_title(f'Feature {i}: {feature_names[i] if i < len(feature_names) else f"feature_{i}"}')
                axes[i].legend()
            
            plt.tight_layout()
            feat_dist_path = viz_dir / "feature_distributions.png"
            plt.savefig(feat_dist_path, dpi=300, bbox_inches='tight')
            plt.close()
            visualization_paths['feature_distributions'] = str(feat_dist_path)
        
        return visualization_paths
    
    @mlflow.trace(name="train_model")
    def train_single_model(self, 
                          model_name: str, 
                          X_train: np.ndarray, 
                          X_test: np.ndarray,
                          y_train: np.ndarray, 
                          y_test: np.ndarray) -> Dict[str, Any]:
        """
        Train a single model with hyperparameter tuning
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
            
        Returns:
            Dictionary containing model results
        """
        logger.info(f"Training {model_name}...")
        
        config = self.model_configs[model_name]
        
        # Hyperparameter tuning
        grid_search = GridSearchCV(
            config['model'],
            config['params'],
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit the model
        grid_search.fit(X_train, y_train)
        
        # Best model
        best_model = grid_search.best_estimator_
        
        # Predictions
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test) if hasattr(best_model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        results = {
            'model': best_model,
            'best_params': grid_search.best_params_,
            'cv_score': grid_search.best_score_,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': class_report,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba
        }
        
        logger.info(f"{model_name} training complete. Accuracy: {accuracy:.4f}")
        return results
    
    @mlflow.trace(name="create_model_visualizations")
    def create_model_visualizations(self, 
                                  model_name: str, 
                                  results: Dict[str, Any],
                                  y_test: np.ndarray) -> Dict[str, str]:
        """
        Create visualizations for a trained model
        
        Args:
            model_name: Name of the model
            results: Model training results
            y_test: True test labels
            
        Returns:
            Dictionary of visualization file paths
        """
        viz_dir = Path("visualizations") / model_name.lower()
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        visualization_paths = {}
        
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(results['confusion_matrix'], 
                   annot=True, 
                   fmt='d', 
                   cmap='Blues',
                   xticklabels=np.unique(y_test),
                   yticklabels=np.unique(y_test))
        plt.title(f'{model_name} - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        cm_path = viz_dir / "confusion_matrix.png"
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        visualization_paths['confusion_matrix'] = str(cm_path)
        
        # Feature importance (if available)
        if hasattr(results['model'], 'feature_importances_'):
            plt.figure(figsize=(12, 8))
            feature_names = get_feature_names()
            if len(feature_names) != len(results['model'].feature_importances_):
                feature_names = [f"feature_{i}" for i in range(len(results['model'].feature_importances_))]
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': results['model'].feature_importances_
            }).sort_values('importance', ascending=False).head(20)
            
            sns.barplot(data=importance_df, x='importance', y='feature')
            plt.title(f'{model_name} - Top 20 Feature Importances')
            plt.xlabel('Importance')
            
            fi_path = viz_dir / "feature_importance.png"
            plt.savefig(fi_path, dpi=300, bbox_inches='tight')
            plt.close()
            visualization_paths['feature_importance'] = str(fi_path)
        
        return visualization_paths
    
    def run_complete_pipeline(self):
        """Run the complete MLflow-integrated pipeline"""
        
        # Start main MLflow run
        with mlflow.start_run(experiment_id=self.experiment_id, run_name="Complete_Pipeline_Run") as main_run:
            
            logger.info("=" * 80)
            logger.info("STARTING COMPLETE MLFLOW PIPELINE")
            logger.info("=" * 80)
            
            # Log pipeline parameters
            mlflow.log_params({
                "data_directory": str(self.data_dir),
                "models_directory": str(self.models_dir),
                "experiment_name": self.experiment_name,
                "pipeline_start_time": datetime.now().isoformat()
            })
            
            # Step 1: Data Preprocessing
            logger.info("Step 1: Data Preprocessing")
            X, y, paths = self.preprocess_data()
            
            if len(X) == 0:
                logger.error("No valid features extracted. Pipeline terminated.")
                return
            
            # Log dataset info
            mlflow.log_params(self.dataset_info)
            
            # Step 2: Create and log MLflow dataset
            logger.info("Step 2: Creating MLflow Dataset")
            dataset = self.create_mlflow_dataset(X, y, paths)
            mlflow.log_input(dataset, context="training")
            
            # Step 3: Dataset visualizations
            logger.info("Step 3: Creating Dataset Visualizations")
            dataset_viz_paths = self.visualize_dataset(X, y)
            
            # Log dataset visualizations
            for viz_name, viz_path in dataset_viz_paths.items():
                mlflow.log_artifact(viz_path, f"dataset_visualizations/{viz_name}")
            
            # Step 4: Prepare data for training
            logger.info("Step 4: Preparing Data for Training")
            
            # Encode labels
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Log data split info
            mlflow.log_params({
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "train_test_split": 0.2,
                "random_state": 42
            })
            
            # Save preprocessing objects
            preprocessing_dir = self.models_dir / "preprocessing"
            preprocessing_dir.mkdir(exist_ok=True)
            
            with open(preprocessing_dir / "label_encoder.pkl", 'wb') as f:
                pickle.dump(label_encoder, f)
            with open(preprocessing_dir / "scaler.pkl", 'wb') as f:
                pickle.dump(scaler, f)
            
            mlflow.log_artifacts(str(preprocessing_dir), "preprocessing")
            
            # Step 5: Model Training
            logger.info("Step 5: Model Training")
            
            model_results = {}
            
            for model_name in self.model_configs.keys():
                logger.info(f"\n--- Training {model_name} ---")
                
                # Create nested run for each model
                with mlflow.start_run(nested=True, run_name=f"{model_name}_Training") as model_run:
                    
                    # Train model
                    results = self.train_single_model(
                        model_name, X_train_scaled, X_test_scaled, y_train, y_test
                    )
                    
                    # Log model parameters and metrics
                    mlflow.log_params({
                        "model_type": model_name,
                        **results['best_params']
                    })
                    
                    mlflow.log_metrics({
                        "accuracy": results['accuracy'],
                        "precision": results['precision'],
                        "recall": results['recall'],
                        "f1_score": results['f1_score'],
                        "cv_score": results['cv_score']
                    })
                    
                    # Log per-class metrics
                    for class_name, metrics in results['classification_report'].items():
                        if isinstance(metrics, dict) and 'f1-score' in metrics:
                            mlflow.log_metrics({
                                f"{class_name}_precision": metrics['precision'],
                                f"{class_name}_recall": metrics['recall'],
                                f"{class_name}_f1_score": metrics['f1-score']
                            })
                    
                    # Create and log model visualizations
                    model_viz_paths = self.create_model_visualizations(
                        model_name, results, 
                        label_encoder.inverse_transform(y_test)
                    )
                    
                    for viz_name, viz_path in model_viz_paths.items():
                        mlflow.log_artifact(viz_path, f"model_visualizations/{viz_name}")
                    
                    # Save model
                    model_path = self.models_dir / f"{model_name.lower()}_model.pkl"
                    with open(model_path, 'wb') as f:
                        pickle.dump(results['model'], f)
                    
                    # Log model to MLflow
                    mlflow.sklearn.log_model(
                        results['model'],
                        f"{model_name}_model",
                        registered_model_name=f"Potato_Disease_{model_name}"
                    )
                    
                    model_results[model_name] = results
                    
                    logger.info(f"{model_name} completed - Accuracy: {results['accuracy']:.4f}")
            
            # Step 6: Model Comparison
            logger.info("Step 6: Creating Model Comparison")
            
            # Create comparison DataFrame
            comparison_data = []
            for model_name, results in model_results.items():
                comparison_data.append({
                    'Model': model_name,
                    'Accuracy': results['accuracy'],
                    'Precision': results['precision'],
                    'Recall': results['recall'],
                    'F1-Score': results['f1_score'],
                    'CV Score': results['cv_score']
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Save comparison
            comparison_path = self.models_dir / "model_comparison.csv"
            comparison_df.to_csv(comparison_path, index=False)
            mlflow.log_artifact(str(comparison_path), "model_comparison")
            
            # Create comparison visualization
            plt.figure(figsize=(12, 8))
            
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            x = np.arange(len(model_results))
            width = 0.2
            
            for i, metric in enumerate(metrics):
                values = [comparison_df[comparison_df['Model'] == model][metric].iloc[0] 
                         for model in model_results.keys()]
                plt.bar(x + i * width, values, width, label=metric)
            
            plt.xlabel('Models')
            plt.ylabel('Score')
            plt.title('Model Performance Comparison')
            plt.xticks(x + width * 1.5, model_results.keys())
            plt.legend()
            plt.ylim(0, 1)
            
            for i, model in enumerate(model_results.keys()):
                for j, metric in enumerate(metrics):
                    value = comparison_df[comparison_df['Model'] == model][metric].iloc[0]
                    plt.text(i + j * width, value + 0.01, f'{value:.3f}', 
                            ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            comparison_viz_path = Path("visualizations") / "model_comparison.png"
            comparison_viz_path.parent.mkdir(exist_ok=True)
            plt.savefig(comparison_viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            mlflow.log_artifact(str(comparison_viz_path), "model_comparison")
            
            # Log best model info
            best_model_name = comparison_df.loc[comparison_df['Accuracy'].idxmax(), 'Model']
            best_accuracy = comparison_df['Accuracy'].max()
            
            mlflow.log_params({
                "best_model": best_model_name,
                "best_accuracy": best_accuracy
            })
            
            # Step 7: Final Summary
            logger.info("Step 7: Pipeline Summary")
            
            summary = {
                "pipeline_completion_time": datetime.now().isoformat(),
                "total_samples_processed": len(X),
                "total_models_trained": len(model_results),
                "best_performing_model": best_model_name,
                "best_model_accuracy": float(best_accuracy),
                "dataset_info": self.dataset_info
            }
            
            summary_path = self.models_dir / "pipeline_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            mlflow.log_artifact(str(summary_path), "pipeline_summary")
            
            # Log final metrics
            mlflow.log_metrics({
                "pipeline_best_accuracy": best_accuracy,
                "total_models_trained": len(model_results)
            })
            
            logger.info("=" * 80)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info(f"Best Model: {best_model_name} (Accuracy: {best_accuracy:.4f})")
            logger.info(f"MLflow Experiment: {self.experiment_name}")
            logger.info(f"Run ID: {main_run.info.run_id}")
            logger.info("=" * 80)
            
            return summary, model_results


def main():
    """Main function to run the complete pipeline"""
    
    # Configuration
    data_dir = "Potato_Health_States"
    models_dir = "models"
    experiment_name = "Potato_Disease_Classification_Complete"
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        logger.error(f"Data directory '{data_dir}' not found!")
        return
    
    # Initialize and run pipeline
    pipeline = MLflowPotateoPipeline(
        data_dir=data_dir,
        models_dir=models_dir,
        experiment_name=experiment_name
    )
    
    try:
        summary, results = pipeline.run_complete_pipeline()
        
        print("\n" + "="*50)
        print("PIPELINE SUMMARY")
        print("="*50)
        print(f"Total samples processed: {summary['total_samples_processed']}")
        print(f"Models trained: {summary['total_models_trained']}")
        print(f"Best model: {summary['best_performing_model']}")
        print(f"Best accuracy: {summary['best_model_accuracy']:.4f}")
        print("\nCheck MLflow UI for detailed experiment tracking and visualizations!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
