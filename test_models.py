#!/usr/bin/env python
"""Test script to check available models"""
import sys
import os
sys.path.append('/app')
sys.path.append('/app/scripts')

from src.simplified_frontend import FeatureModelManager

def test_models():
    manager = FeatureModelManager('/app/models')
    models = manager.discover_feature_models()
    
    print("Available Feature-Based Models:")
    for name, info in models.items():
        print(f"- {info['display_name']} ({name})")
    
    print(f"\nTotal models found: {len(models)}")
    
    # Test if model files exist
    print("\nModel file status:")
    for name, info in models.items():
        model_exists = os.path.exists(info['model_path'])
        encoder_exists = os.path.exists(info['encoder_path'])
        scaler_exists = os.path.exists(info['scaler_path'])
        print(f"  {name}:")
        print(f"    Model: {'✓' if model_exists else '✗'}")
        print(f"    Encoder: {'✓' if encoder_exists else '✗'}")
        print(f"    Scaler: {'✓' if scaler_exists else '✗'}")

if __name__ == "__main__":
    test_models()
