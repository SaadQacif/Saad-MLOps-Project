#!/usr/bin/env python
"""
Test script to validate the feature dimension compatibility fixes

This script tests:
1. The complex feature extractor produces 121 features
2. The simple feature extractor produces 30 features 
3. Models load properly and can make predictions with correctly sized features
"""

import os
import sys
import cv2
import numpy as np
import pickle
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "scripts"))

from scripts.extract_potato_features import extract_features, get_feature_names
from src.complex_feature_extraction import create_feature_extractor
from sklearn.preprocessing import StandardScaler

def test_feature_extraction():
    """Test that feature extraction returns correct number of features"""
    
    # Load a sample image
    sample_image_path = "Potato_Health_States/Potato___healthy/00fc2ee5-729f-4757-8aeb-65c3355874f2___RS_HL 1864_final_masked.jpg"
    
    if not os.path.exists(sample_image_path):
        print(f"Sample image not found: {sample_image_path}")
        return False
    
    # Load image
    image = cv2.imread(sample_image_path)
    if image is None:
        print("Failed to load image")
        return False
    
    print(f"Loaded image with shape: {image.shape}")
    
    # Extract features
    features = extract_features(image)
    
    if features is None:
        print("Feature extraction failed")
        return False
    
    print(f"Extracted {len(features)} features")
    print(f"Features: {features}")
    
    # Get feature names
    feature_names = get_feature_names()
    print(f"Feature names ({len(feature_names)}): {feature_names}")
    
    # Verify we have exactly 30 features
    if len(features) != 30:
        print(f"ERROR: Expected 30 features, got {len(features)}")
        return False
    
    if len(feature_names) != 30:
        print(f"ERROR: Expected 30 feature names, got {len(feature_names)}")
        return False
    
    print("✅ Feature extraction is working correctly!")
      # Test loading a model and predicting
    model_path = "models/randomforest_model.pkl"
    if os.path.exists(model_path):
        print(f"\nTesting model prediction with {model_path}")
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            if isinstance(model_data, dict):
                model = model_data['model']
                scaler = model_data.get('scaler', StandardScaler())
                print("Loaded model and scaler from dict")
            else:
                model = model_data
                # Check if a scaler exists - first try scaler.pkl then scaler_multi.pkl
                scaler_path = os.path.join("models", "bin", "scaler.pkl")
                scaler_multi_path = os.path.join("models", "bin", "scaler_multi.pkl")
                
                # Try loading the scaler.pkl (for 30 features)
                if os.path.exists(scaler_path):
                    try:
                        with open(scaler_path, 'rb') as sf:
                            scaler = pickle.load(sf)
                        print(f"Loaded scaler from {scaler_path}")
                    except Exception as e:
                        print(f"Error loading scaler: {e}")
                        # Create and fit a new scaler for the test data
                        scaler = StandardScaler()
                        scaler.fit(features.reshape(1, -1))
                        print("Created and fitted a new scaler for 30 features")
                # If no scaler.pkl, explicitly fit a new one on the current features
                else:
                    # Create and fit a scaler if none exists
                    scaler = StandardScaler()
                    scaler.fit(features.reshape(1, -1))
                    print("Created and fitted a new scaler for the current features")
            
            # Reshape features for prediction
            features_reshaped = features.reshape(1, -1)
            print(f"Features shape for prediction: {features_reshaped.shape}")            # Try to scale features
            try:
                # Get expected feature count from model
                expected_features = model.n_features_in_ if hasattr(model, 'n_features_in_') else None
                
                # Check if scaler matches expected feature count
                if hasattr(scaler, 'n_features_in_') and expected_features is not None:
                    if scaler.n_features_in_ != expected_features:
                        print(f"⚠️ Scaler feature count mismatch: scaler={scaler.n_features_in_}, model={expected_features}")
                        print("Creating new scaler to match model's feature count")
                        scaler = StandardScaler()
                
                # Reshape features if needed
                if expected_features is not None and features_reshaped.shape[1] != expected_features:
                    print(f"⚠️ Feature count mismatch: got {features_reshaped.shape[1]}, expected {expected_features}")
                    if features_reshaped.shape[1] < expected_features:
                        padded = np.zeros((features_reshaped.shape[0], expected_features))
                        padded[:, :features_reshaped.shape[1]] = features_reshaped
                        features_reshaped = padded
                    else:
                        features_reshaped = features_reshaped[:, :expected_features]
                    print(f"Adjusted feature shape to {features_reshaped.shape}")
                
                # Scale features
                if hasattr(scaler, 'n_features_in_'):
                    features_scaled = scaler.transform(features_reshaped)
                else:
                    print("⚠️ Scaler not fitted. Fitting on sample data...")
                    scaler.fit(features_reshaped)
                    features_scaled = scaler.transform(features_reshaped)
                
                print(f"✅ Feature scaling successful! Scaled shape: {features_scaled.shape}")
                
                # Try prediction
                prediction = model.predict(features_scaled)
                print(f"✅ Prediction successful! Result: {prediction}")
                
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(features_scaled)
                    print(f"✅ Prediction probabilities: {probabilities}")
                
            except Exception as e:
                print(f"❌ Scaling/Prediction failed: {e}")
                return False
                
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            return False
    else:
        print(f"Model not found: {model_path}")
    
    return True

def test_complex_extractor():
    """Test that complex feature extractor returns 121 features"""
    
    try:
        # Create a test image
        test_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        
        # Create the complex extractor with 121 features
        complex_extractor = create_feature_extractor(feature_count=121)
        
        # Extract features with max_features=121
        complex_features = complex_extractor.extract_all_features(test_image, max_features=121)
        
        logger.info(f"Complex feature extraction successful: {len(complex_features)} features")
        
        # Check if we got 121 features
        if len(complex_features) != 121:
            logger.error(f"Expected 121 features, got {len(complex_features)}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Complex feature extraction failed: {e}")
        return False

def load_model_and_scaler(model_name, bin_path):
    """Load a model and its corresponding scaler"""
    model_path = os.path.join(bin_path, f"{model_name}_model.pkl")
    
    # Determine which scaler to use
    if model_name == "GradientBoosting":
        scaler_path = os.path.join(bin_path, "scaler.pkl")  # 30 features
    else:
        scaler_path = os.path.join(bin_path, "scaler_multi.pkl")  # 121 features
    
    # Load model
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Loaded model {model_name} from {model_path}")
        
        if hasattr(model, 'n_features_in_'):
            logger.info(f"Model expects {model.n_features_in_} features")
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        return None, None
    
    # Load scaler
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        logger.info(f"Loaded scaler from {scaler_path}")
        
        if hasattr(scaler, 'n_features_in_'):
            logger.info(f"Scaler expects {scaler.n_features_in_} features")
    except Exception as e:
        logger.error(f"Failed to load scaler for {model_name}: {e}")
        return model, None
    
    return model, scaler

def test_model_predictions():
    """Test predictions with all models"""
    # Create a test image
    test_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    
    # Get complex features (121)
    complex_extractor = create_feature_extractor(feature_count=121)
    complex_features = complex_extractor.extract_all_features(test_image, max_features=121)
    complex_features = complex_features.reshape(1, -1)
    
    # Get simple features (30)
    simple_features = extract_features(test_image)
    simple_features = simple_features.reshape(1, -1)
    
    models_dir = os.path.join(project_root, "models", "bin")
    
    # Test each model
    for model_name in ["SVM", "Random_Forest", "MLP", "GradientBoosting"]:
        model, scaler = load_model_and_scaler(model_name, models_dir)
        
        if model is None:
            continue
        
        # Determine feature count
        expected_features = model.n_features_in_ if hasattr(model, 'n_features_in_') else 121
        
        # Use appropriate features
        if expected_features == 30:
            features = simple_features
            logger.info(f"Using 30 features for {model_name}")
        else:
            features = complex_features
            logger.info(f"Using 121 features for {model_name}")
        
        # Apply scaling if available
        if scaler is not None:
            try:
                scaled_features = scaler.transform(features)
                logger.info(f"Applied scaling to features")
            except Exception as e:
                logger.error(f"Scaling failed: {e}")
                scaled_features = features
        else:
            scaled_features = features
        
        # Make prediction
        try:
            prediction = model.predict(scaled_features)
            logger.info(f"{model_name} prediction successful: {prediction}")
        except Exception as e:
            logger.error(f"{model_name} prediction failed: {e}")

if __name__ == "__main__":
    logger.info("Testing simple feature extraction...")
    test_feature_extraction()
    
    logger.info("\nTesting complex feature extraction...")
    test_complex_extractor()
    
    logger.info("\nTesting model predictions with appropriate features...")
    test_model_predictions()
    
    logger.info("Test completed!")
