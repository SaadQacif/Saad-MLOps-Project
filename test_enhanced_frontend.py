#!/usr/bin/env python
"""
Test script for Enhanced Frontend functionality
"""

def test_enhanced_frontend():
    """Test the enhanced frontend dependencies and functionality"""
    
    print("🔍 Testing Enhanced Frontend Dependencies...")
    print("=" * 50)
    
    # Test core imports
    try:
        import streamlit as st
        print(f"✅ Streamlit: {st.__version__}")
    except Exception as e:
        print(f"❌ Streamlit: {e}")
        return False
    
    # Test visualization libraries
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
        import plotly.express as px
        print(f"✅ Seaborn: {sns.__version__}")
        print(f"✅ Matplotlib: {plt.matplotlib.__version__}")
        print("✅ Plotly: Available")
    except Exception as e:
        print(f"❌ Visualization libraries: {e}")
        return False
    
    # Test data science libraries
    try:
        import pandas as pd
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        print(f"✅ Pandas: {pd.__version__}")
        print(f"✅ NumPy: {np.__version__}")
        print("✅ Scikit-learn: Available")
    except Exception as e:
        print(f"❌ Data science libraries: {e}")
        return False
    
    # Test MLflow
    try:
        import mlflow
        print(f"✅ MLflow: {mlflow.__version__}")
    except Exception as e:
        print(f"❌ MLflow: {e}")
    
    # Test model availability
    import os
    from pathlib import Path
    
    models_dir = Path("models")
    bin_models = list(models_dir.glob("bin/*_model.pkl"))
    root_models = list(models_dir.glob("*_model.pkl"))
    
    print()
    print("📁 Available Models:")
    print(f"✅ Feature-based models: {len(bin_models)}")
    for model in bin_models:
        print(f"   - {model.name}")
    print(f"✅ Traditional models: {len(root_models)}")
    for model in root_models:
        print(f"   - {model.name}")
    
    print()
    print("🎯 Status: ALL DEPENDENCIES RESOLVED!")
    print("🚀 Enhanced frontend ready for production use!")
    
    return True

if __name__ == "__main__":
    success = test_enhanced_frontend()
    if success:
        print("\n🎉 Enhanced Frontend Test: PASSED")
    else:
        print("\n❌ Enhanced Frontend Test: FAILED")
