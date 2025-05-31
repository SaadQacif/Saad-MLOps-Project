"""
Environment setup script for the MLOps project.
This script creates the necessary directory structure, initializes the environment,
and sets up configurations for the project.
"""

import os
import sys
import json
import argparse
import subprocess
import shutil
from pathlib import Path


def create_directory_structure(base_dir):
    """
    Create the directory structure for the project.
    
    Args:
        base_dir: Base directory for the project
    """
    directories = [
        "configs",
        "data/inputs",
        "data/outputs",
        "data/uploads",
        "deployment",
        "docs",
        "models/bin",
        "models/tracking",
        "notebooks",
        "scripts",
        "src",
        "tests",
        "tmp",
        "visualizations",
        "logs"
    ]
    
    for directory in directories:
        dir_path = os.path.join(base_dir, directory)
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")
        
        # Create .gitkeep files in empty directories
        if not os.listdir(dir_path):
            with open(os.path.join(dir_path, ".gitkeep"), "w") as f:
                pass


def copy_sample_data(base_dir, sample_data_dir):
    """
    Copy sample data to the project.
    
    Args:
        base_dir: Base directory for the project
        sample_data_dir: Directory containing sample data
    """
    if not os.path.exists(sample_data_dir):
        print(f"Sample data directory not found: {sample_data_dir}")
        return
    
    destination = os.path.join(base_dir, "data", "inputs")
    
    # Copy sample data
    for file_name in os.listdir(sample_data_dir):
        file_path = os.path.join(sample_data_dir, file_name)
        if os.path.isfile(file_path) and file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            shutil.copy(file_path, destination)
            print(f"Copied {file_name} to {destination}")


def setup_python_environment(requirements_file, venv_dir=None):
    """
    Set up the Python virtual environment.
    
    Args:
        requirements_file: Path to requirements.txt file
        venv_dir: Directory for the virtual environment
    """
    if not os.path.exists(requirements_file):
        print(f"Requirements file not found: {requirements_file}")
        return
    
    if venv_dir is None:
        venv_dir = "venv"
    
    # Create virtual environment
    os.system(f"python -m venv {venv_dir}")
    
    # Determine the pip path based on the platform
    if sys.platform.startswith('win'):
        pip_path = os.path.join(venv_dir, "Scripts", "pip")
    else:
        pip_path = os.path.join(venv_dir, "bin", "pip")
    
    # Install requirements
    os.system(f"{pip_path} install --upgrade pip")
    os.system(f"{pip_path} install -r {requirements_file}")
    
    print(f"Virtual environment created at {venv_dir}")
    print(f"Installed requirements from {requirements_file}")


def setup_config(base_dir):
    """
    Set up the configuration files.
    
    Args:
        base_dir: Base directory for the project
    """
    config_dir = os.path.join(base_dir, "configs")
    config_file = os.path.join(config_dir, "config.json")
    
    # Create default config if it doesn't exist
    if not os.path.exists(config_file):
        print("Creating default configuration...")
        
        config = {
            "paths": {
                "data": {
                    "input": "data/inputs",
                    "output": "data/outputs",
                    "upload": "data/uploads",
                    "temp": "tmp"
                },
                "models": {
                    "directory": "models/bin",
                    "model_file": "models_by_features.pkl",
                    "encoder_file": "encoder.pkl",
                    "metadata_file": "model_metadata.json"
                },
                "visualizations": "visualizations"
            },
            "model": {
                "type": "RandomForest",
                "parameters": {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "random_state": 42
                },
                "training": {
                    "test_size": 0.25,
                    "random_state": 42
                }
            },
            "image_processing": {
                "contour": {
                    "threshold": 70,
                    "kernel_size": 3
                },
                "clahe": {
                    "clip_limit": 2.0,
                    "tile_grid_size": [8, 8]
                }
            },
            "api": {
                "host": "0.0.0.0",
                "port": 5000,
                "debug": False
            }
        }
        
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"Created default configuration at {config_file}")


def setup_env_file(base_dir):
    """
    Create the .env file for environment variables.
    
    Args:
        base_dir: Base directory for the project
    """
    env_file = os.path.join(base_dir, ".env")
    
    if not os.path.exists(env_file):
        print("Creating .env file...")
        
        with open(env_file, "w") as f:
            f.write("# Environment variables for the MLOps project\n")
            f.write("PYTHONPATH=${PYTHONPATH}:${PWD}\n")
            f.write("API_ADMIN_KEY=admin-secret-key\n")
            f.write("FLASK_APP=src/api.py\n")
            f.write("FLASK_ENV=development\n")
        
        print(f"Created .env file at {env_file}")


def main():
    """Main function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Environment setup script")
    parser.add_argument("--base-dir", type=str, default=".", help="Base directory for the project")
    parser.add_argument("--sample-data", type=str, help="Directory containing sample data")
    parser.add_argument("--requirements", type=str, default="requirements.txt", help="Path to requirements.txt")
    parser.add_argument("--venv", type=str, default="venv", help="Directory for virtual environment")
    parser.add_argument("--skip-venv", action="store_true", help="Skip virtual environment creation")
    parser.add_argument("--init-models", action="store_true", help="Initialize model tracking")
    
    args = parser.parse_args()
    
    # Resolve absolute paths
    base_dir = os.path.abspath(args.base_dir)
    requirements_file = os.path.abspath(args.requirements)
    
    # Create directory structure
    create_directory_structure(base_dir)
    
    # Set up configuration files
    setup_config(base_dir)
    
    # Set up .env file
    setup_env_file(base_dir)
    
    # Copy sample data if provided
    if args.sample_data:
        sample_data_dir = os.path.abspath(args.sample_data)
        copy_sample_data(base_dir, sample_data_dir)
    
    # Set up Python environment
    if not args.skip_venv:
        venv_dir = os.path.abspath(args.venv)
        setup_python_environment(requirements_file, venv_dir)
    
    print("\nEnvironment setup complete!")
    print("\nNext steps:")
    print("1. Add data to the input directory")
    print("2. Train the model with:")
    print("   python src/train_model.py")
    print("3. Run the API with:")
    print("   python src/api.py")
    print("4. Start the application with:")
    print("   python src/main.py")
    print("\nOr use Docker Compose:")
    print("   docker-compose up api")
    print("   docker-compose --profile training up")
    print("   docker-compose --profile batch up")


if __name__ == "__main__":
    main()
