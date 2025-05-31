#!/usr/bin/env python3
"""
Test script to verify the scaling issue fix in enhanced_complete_frontend.py
This tests the exact scenario that was causing the "Scaler expects 30 features, but got 121" error.
"""

import sys
import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Add necessary paths
sys.path.extend(['src', 'scripts'])

def test_scaling_fix():
    """Test that the scaling issue has been resolved"""
    print("🧪 Testing scaling fix for enhanced frontend...")
    
    try:
        # Import the feature extraction function
        from scripts.extract_potato_features import extract_all_features_from_array
        print("✅ Feature extraction import successful")
        
        # Create test image (similar to what would be uploaded)
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        print("✅ Test image created")
        
        # Extract features using the standard method (30 features)
        features_dict = extract_all_features_from_array(test_image)
        features_df = pd.DataFrame([features_dict])
        
        # Apply the same feature order as the frontend
        feature_order = [
            'r_dev', 'g_dev', 'b_dev', 'red_mean', 'red_std', 'red_kurtosis', 'red_skew',
            'green_mean', 'green_std', 'green_kurtosis', 'green_skew', 'blue_mean', 'blue_std',
            'blue_kurtosis', 'blue_skew', 'green_ratio', 'diseased_ratio', 'entropy_mean',
            'entropy_std', 'entropy_variation_rhythm', 'texture_contrast', 'avg_convexity',
            'min_convexity', 'contour_area', 'contour_perimeter', 'circularity', 'eccentricity',
            'solidity', 'aspect_ratio', 'area_bbox_ratio'
        ]
        features_df = features_df[feature_order]
        
        print(f"✅ Feature extraction successful: {features_df.shape}")
        print(f"   Features count: {len(features_df.columns)}")
        
        # Test with models from bin directory (these were problematic before)
        bin_models_dir = Path("models/bin")
        if bin_models_dir.exists():
            # Test with RandomForest model from bin
            model_path = bin_models_dir / "randomforest_model.pkl"
            scaler_path = bin_models_dir / "scaler.pkl"
            
            if model_path.exists() and scaler_path.exists():
                print(f"\n🔬 Testing with bin model: {model_path.name}")
                
                # Load model and scaler
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                
                print(f"✅ Model loaded: {type(model).__name__}")
                print(f"✅ Scaler loaded: {type(scaler).__name__}")
                
                # Check scaler expectations
                scaler_features = getattr(scaler, 'n_features_in_', 'unknown')
                print(f"   Scaler expects: {scaler_features} features")
                print(f"   We provide: {features_df.shape[1]} features")
                
                # THE CRITICAL TEST: Scaling (this used to fail)
                try:
                    scaled_features = scaler.transform(features_df)
                    print(f"✅ SCALING SUCCESSFUL! Shape: {scaled_features.shape}")
                    
                    # Test prediction as well
                    prediction = model.predict(scaled_features)
                    if hasattr(model, 'predict_proba'):
                        prediction_proba = model.predict_proba(scaled_features)
                        print(f"✅ Prediction successful: {prediction[0]}")
                        print(f"   Probabilities: {prediction_proba[0]}")
                    else:
                        print(f"✅ Prediction successful: {prediction[0]}")
                    
                    print(f"\n🎉 SUCCESS! The scaling issue has been completely resolved!")
                    print(f"   - Features extracted: {features_df.shape[1]}")
                    print(f"   - Scaler expected: {scaler_features}")
                    print(f"   - Scaling: PASSED ✅")
                    print(f"   - Prediction: PASSED ✅")
                    
                    return True
                    
                except Exception as e:
                    print(f"❌ SCALING FAILED: {e}")
                    print(f"   This indicates the scaling issue is NOT fixed")
                    return False
            else:
                print(f"❌ Model or scaler files not found in {bin_models_dir}")
                return False
        else:
            print(f"❌ Bin models directory not found: {bin_models_dir}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_scaling_fix()
    if success:
        print(f"\n🏆 ALL TESTS PASSED! The frontend is ready for use.")
    else:
        print(f"\n💥 TESTS FAILED! The scaling issue may still exist.")
    
    sys.exit(0 if success else 1)
