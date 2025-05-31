#!/usr/bin/env python3
"""
Initialize MLflow tracking server with proper experiment setup
"""
import os
import mlflow
from mlflow.tracking import MlflowClient

def initialize_mlflow():
    """Initialize MLflow with clean experiments"""
    
    # Set tracking URI to the local workspace mlruns directory (relative path)
    mlflow.set_tracking_uri("file:./mlruns")
    
    # Create client
    client = MlflowClient()
    
    # Create default experiment if it doesn't exist
    try:
        default_exp = client.get_experiment_by_name("Default")
        if default_exp is None:
            print("Creating Default experiment...")
            client.create_experiment("Default", artifact_location="./mlruns/0")
        else:
            print("Default experiment already exists")
    except Exception as e:
        print(f"Default experiment setup: {e}")
    
    # Create potato disease classification experiment
    try:
        potato_exp = client.get_experiment_by_name("potato-disease-classification")
        if potato_exp is None:
            print("Creating potato-disease-classification experiment...")
            client.create_experiment("potato-disease-classification", artifact_location="./mlruns/1")
        else:
            print("Potato disease classification experiment already exists")
    except Exception as e:
        print(f"Potato experiment setup: {e}")
    
    print("MLflow initialization completed!")

if __name__ == "__main__":
    initialize_mlflow()
