#!/usr/bin/env python
"""
Simplified Multi-Model Training Script without MLflow complexity
Trains Random Forest, SVM, and MLP models using the original feature extraction method
"""

import os
import sys
import json
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Computer vision
import cv2
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import original feature extraction
import scripts.extract_potato_features as feature_extractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleModelTrainer:
    """
    Simple trainer for multiple ML models using original feature extraction
    """
    
    def __init__(self, 
                 data_path: str,
                 output_path: str = "models/bin"):
        
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        
        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Define models to train (simplified hyperparameters for faster training)
        self.models = {
            'Random_Forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20],
                    'min_samples_split': [2, 5]
                }
            },
            'SVM': {
                'model': SVC(probability=True, random_state=42),
                'params': {
                    'C': [1, 10],
                    'gamma': ['scale', 'auto'],
                    'kernel': ['rbf', 'poly']
                }
            },
            'MLP': {
                'model': MLPClassifier(random_state=42, max_iter=500),
                'params': {
                    'hidden_layer_sizes': [(100,), (100, 50)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001]
                }
            }
        }
        
        self.results = {}
        
    def load_and_extract_features(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load images and extract original features (30 features)
        """
        logger.info("Loading images and extracting original features...")
        
        features_list = []
        labels_list = []
        class_names = []
        
        # Expected class directories
        class_dirs = ['Potato___healthy', 'Potato___Early_blight', 'Potato___Late_blight']
        
        total_images = 0
        for class_dir in class_dirs:
            class_path = self.data_path / class_dir
            if class_path.exists():
                image_files = list(class_path.glob('*.jpg')) + list(class_path.glob('*.jpeg')) + list(class_path.glob('*.png'))
                total_images += len(image_files)
        
        logger.info(f"Found {total_images} total images across {len(class_dirs)} classes")
        
        processed_count = 0
        for class_idx, class_dir in enumerate(class_dirs):
            class_path = self.data_path / class_dir
            if not class_path.exists():
                logger.warning(f"Class directory not found: {class_path}")
                continue
                
            class_names.append(class_dir)
            image_files = list(class_path.glob('*.jpg')) + list(class_path.glob('*.jpeg')) + list(class_path.glob('*.png'))
            
            logger.info(f"Processing class '{class_dir}': {len(image_files)} images")
            
            for img_file in image_files:
                try:
                    # Load image
                    image = cv2.imread(str(img_file))
                    if image is None:
                        logger.warning(f"Could not load image: {img_file}")
                        continue
                    
                    # Extract features using original method
                    features = feature_extractor.extract_features(image)
                    
                    if features is not None and len(features) > 0:
                        features_list.append(features)
                        labels_list.append(class_idx)
                        processed_count += 1
                        
                        if processed_count % 100 == 0:
                            logger.info(f"Processed {processed_count}/{total_images} images")
                    
                except Exception as e:
                    logger.error(f"Error processing {img_file}: {e}")
                    continue
        
        logger.info(f"Successfully extracted features from {len(features_list)} images")
        
        if len(features_list) == 0:
            raise ValueError("No features extracted from any images")
        
        # Convert to numpy arrays
        X = np.array(features_list)
        y = np.array(labels_list)
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Labels shape: {y.shape}")
        logger.info(f"Classes: {class_names}")
        
        return X, y, class_names
    
    def train_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray,
                   X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Train a single model with grid search
        """
        logger.info(f"Training {model_name}...")
        
        model_config = self.models[model_name]
        model = model_config['model']
        params = model_config['params']
        
        # Perform grid search
        grid_search = GridSearchCV(
            model, 
            params, 
            cv=3,  # 3-fold cross-validation for speed
            scoring='accuracy',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        
        # Make predictions
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Generate classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        logger.info(f"{model_name} - Best params: {grid_search.best_params_}")
        logger.info(f"{model_name} - Test accuracy: {accuracy:.4f}")
        
        # Save model
        model_path = self.output_path / f"{model_name}_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)
        
        results = {
            'model': best_model,
            'best_params': grid_search.best_params_,
            'test_accuracy': accuracy,
            'classification_report': class_report,
            'model_path': str(model_path)
        }
        
        return results
    
    def train_all_models(self) -> Dict[str, Any]:
        """
        Train all models and return results
        """
        logger.info("Starting multi-model training pipeline...")
        
        # Load and extract features
        X, y, class_names = self.load_and_extract_features()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Save scaler
        scaler_path = self.output_path / "scaler_multi.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        # Create label encoder
        label_encoder = LabelEncoder()
        label_encoder.fit(range(len(class_names)))
        
        # Save label encoder and class names
        encoder_path = self.output_path / "label_encoder_multi.pkl"
        with open(encoder_path, 'wb') as f:
            pickle.dump(label_encoder, f)
        
        classes_path = self.output_path / "class_names_multi.json"
        with open(classes_path, 'w') as f:
            json.dump(class_names, f)
        
        logger.info(f"Training set shape: {X_train_scaled.shape}")
        logger.info(f"Test set shape: {X_test_scaled.shape}")
        
        # Train all models
        all_results = {}
        for model_name in self.models.keys():
            try:
                model_results = self.train_model(
                    model_name, X_train_scaled, y_train, X_test_scaled, y_test
                )
                all_results[model_name] = model_results
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
        
        # Save training summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_samples': len(X),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_count': X.shape[1],
            'classes': class_names,
            'model_results': {
                name: {
                    'test_accuracy': results['test_accuracy'],
                    'best_params': results['best_params']
                }
                for name, results in all_results.items()
            }
        }
        
        summary_path = self.output_path / "training_summary_multi.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("Training completed!")
        logger.info("Model performance summary:")
        for name, results in all_results.items():
            logger.info(f"  {name}: {results['test_accuracy']:.4f}")
        
        return all_results

def main():
    """
    Main training function
    """
    # Data path
    data_path = "Potato_Health_States"
    
    # Initialize trainer
    trainer = SimpleModelTrainer(data_path)
    
    # Train all models
    results = trainer.train_all_models()
    
    print("\\n=== TRAINING COMPLETED ===")
    print("Model Performance:")
    for name, result in results.items():
        print(f"  {name}: {result['test_accuracy']:.4f}")

if __name__ == "__main__":
    main()
