"""
Model deployment script for the MLOps project.
This script handles model deployment to different platforms and environments.
Includes model versioning, validation, and deployment tracking.
"""

import os
import sys
import json
import argparse
import shutil
import pickle
import time
import datetime
import logging
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), '..', 'logs', 'deployment.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("model_deployment")

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Try to import the model tracker
try:
    from model_tracking import MLFlowLikeTracker
    tracker_available = True
except ImportError:
    logger.warning("MLFlowLikeTracker not available. Some features will be disabled.")
    tracker_available = False


def load_config():
    """
    Load configuration from JSON file.
    
    Returns:
        Configuration dictionary or default config if file not found
    """
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.json')
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"Config file not found: {config_path}")
        return {
            "paths": {
                "models": {
                    "directory": "models/bin",
                    "model_file": "models_by_features.pkl",
                    "encoder_file": "encoder.pkl",
                    "metadata_file": "model_metadata.json"
                }
            }
        }


def validate_model(model_path, encoder_path=None):
    """
    Validate that the model can be loaded and basic operations performed.
    
    Args:
        model_path: Path to the model file
        encoder_path: Path to the encoder file
        
    Returns:
        True if validation successful, False otherwise
    """
    try:
        logger.info(f"Validating model at {model_path}")
        
        # Try to load the model
        start_time = time.time()
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        load_time = time.time() - start_time
        
        # Check if it's a models_by_features dictionary or a direct model
        if isinstance(model_data, dict) and 'model' in model_data:
            model = model_data['model']
        else:
            model = model_data
        
        # Check if model has predict method
        if not hasattr(model, 'predict'):
            logger.error("Model does not have predict method")
            return False
        
        # Load encoder if specified
        if encoder_path and os.path.exists(encoder_path):
            with open(encoder_path, 'rb') as f:
                encoder = pickle.load(f)
            logger.info(f"Encoder loaded successfully from {encoder_path}")
        
        logger.info(f"Model validation successful (loaded in {load_time:.2f}s)")
        return True
    
    except Exception as e:
        logger.error(f"Model validation failed: {str(e)}")
        return False


def create_deployment_metadata(source_dir, target_dir, environment, version=None):
    """
    Create metadata for the deployment.
    
    Args:
        source_dir: Source directory of model
        target_dir: Target deployment directory
        environment: Deployment environment (dev, test, prod)
        version: Optional version string
        
    Returns:
        Dictionary of metadata
    """
    # Generate version if not provided
    if not version:
        version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    metadata = {
        "version": version,
        "environment": environment,
        "source_directory": source_dir,
        "target_directory": target_dir,
        "deployment_timestamp": datetime.datetime.now().isoformat(),
        "deployed_by": os.environ.get("USER", os.environ.get("USERNAME", "unknown"))
    }
    
    # Try to get git information
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], 
                                          stderr=subprocess.DEVNULL).decode().strip()
        git_branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], 
                                           stderr=subprocess.DEVNULL).decode().strip()
        
        metadata["git_hash"] = git_hash
        metadata["git_branch"] = git_branch
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.warning("Could not get git information")
    
    return metadata


def package_model(model_dir, output_dir, environment="dev", version=None, config_file=None):
    """
    Package the model files for deployment.
    
    Args:
        model_dir: Directory containing model files
        output_dir: Directory to save the packaged model
        environment: Deployment environment (dev, test, prod)
        version: Optional version string
        config_file: Optional configuration file to include
        
    Returns:
        Dictionary with packaging information
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load config
    config = load_config()
    
    # Define model files
    model_file = os.path.join(model_dir, config['paths']['models']['model_file'])
    encoder_file = os.path.join(model_dir, config['paths']['models']['encoder_file'])
    metadata_file = os.path.join(model_dir, config['paths']['models']['metadata_file'])
    
    # Check if files exist
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}")
    if not os.path.exists(encoder_file):
        raise FileNotFoundError(f"Encoder file not found: {encoder_file}")
    
    # Validate model before packaging
    if not validate_model(model_file, encoder_file):
        raise ValueError("Model validation failed")
    
    # Copy model files to output directory
    shutil.copy(model_file, os.path.join(output_dir, os.path.basename(model_file)))
    shutil.copy(encoder_file, os.path.join(output_dir, os.path.basename(encoder_file)))
    
    if os.path.exists(metadata_file):
        shutil.copy(metadata_file, os.path.join(output_dir, os.path.basename(metadata_file)))
    
    # Copy configuration file if provided
    if config_file and os.path.exists(config_file):
        shutil.copy(config_file, os.path.join(output_dir, os.path.basename(config_file)))
    
    # Create a deployment information file
    deployment_info = {
        "model_version": "1.0.0",
        "deployment_date": get_current_date(),
        "model_type": config['model']['type'],
        "files": {
            "model": os.path.basename(model_file),
            "encoder": os.path.basename(encoder_file),
            "metadata": os.path.basename(metadata_file) if os.path.exists(metadata_file) else None,
            "config": os.path.basename(config_file) if config_file and os.path.exists(config_file) else None
        }
    }
    
    # Write deployment information to file
    with open(os.path.join(output_dir, "deployment_info.json"), "w") as f:
        json.dump(deployment_info, f, indent=2)
    
    print(f"Model packaged successfully to {output_dir}")
    return os.path.join(output_dir, "deployment_info.json")


def get_current_date():
    """Get current date in ISO format."""
    from datetime import datetime
    return datetime.now().isoformat()


def create_docker_deployment(package_dir, output_dir, platform=None):
    """
    Create Docker deployment files.
    
    Args:
        package_dir: Directory containing packaged model
        output_dir: Directory to save deployment files
        platform: Target platform (e.g., 'aws', 'azure', 'gcp')
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy packaged model files to output directory
    for file_name in os.listdir(package_dir):
        file_path = os.path.join(package_dir, file_name)
        if os.path.isfile(file_path):
            shutil.copy(file_path, os.path.join(output_dir, file_name))
    
    # Create Dockerfile
    dockerfile_content = """FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model
COPY . .

# Create required directories
RUN mkdir -p data/inputs data/outputs data/uploads tmp

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/models_by_features.pkl
ENV ENCODER_PATH=/app/encoder.pkl

# Expose port for API
EXPOSE 5000

# Command to run the API
CMD ["python", "api.py"]
"""
    
    # Add platform-specific configurations
    if platform == 'aws':
        dockerfile_content += """
# AWS-specific configurations
ENV AWS_REGION=us-west-2
"""
    elif platform == 'azure':
        dockerfile_content += """
# Azure-specific configurations
ENV AZURE_REGION=westus2
"""
    elif platform == 'gcp':
        dockerfile_content += """
# GCP-specific configurations
ENV GCP_REGION=us-central1
"""
    
    # Write Dockerfile
    with open(os.path.join(output_dir, "Dockerfile"), "w") as f:
        f.write(dockerfile_content)
    
    # Copy API file
    api_file = os.path.join(os.path.dirname(__file__), '..', 'src', 'api.py')
    if os.path.exists(api_file):
        shutil.copy(api_file, os.path.join(output_dir, "api.py"))
    
    # Copy requirements file
    requirements_file = os.path.join(os.path.dirname(__file__), '..', 'requirements.txt')
    if os.path.exists(requirements_file):
        shutil.copy(requirements_file, os.path.join(output_dir, "requirements.txt"))
    
    # Copy necessary source files
    src_files = ['segmentation.py', 'image_processing.py', 'utils.py']
    src_dir = os.path.join(os.path.dirname(__file__), '..', 'src')
    for file_name in src_files:
        file_path = os.path.join(src_dir, file_name)
        if os.path.exists(file_path):
            shutil.copy(file_path, os.path.join(output_dir, file_name))
    
    # Create docker-compose.yml file
    compose_content = """version: '3'

services:
  api:
    build: .
    image: mlops-python-api
    ports:
      - "5000:5000"
    environment:
      - PYTHONUNBUFFERED=1
"""
    
    # Write docker-compose.yml
    with open(os.path.join(output_dir, "docker-compose.yml"), "w") as f:
        f.write(compose_content)
    
    # Create a README file
    readme_content = f"""# MLOps Python Model Deployment

This directory contains the files needed to deploy the MLOps Python model as a Docker container.

## Files

- `Dockerfile`: Docker configuration file
- `docker-compose.yml`: Docker Compose configuration file
- `api.py`: API server for model serving
- `models_by_features.pkl`: The trained model
- `encoder.pkl`: The label encoder
- `deployment_info.json`: Information about the deployment
- Additional source files needed for running the model

## Deployment

To deploy the model, run:

```bash
docker-compose up -d
```

The API will be available at http://localhost:5000

## API Endpoints

- `GET /health`: Check API health and model status
- `POST /predict`: Submit a single image for prediction
- `POST /batch_predict`: Submit multiple images for prediction
"""
    
    # Add platform-specific instructions
    if platform == 'aws':
        readme_content += """
## AWS Deployment

To deploy to AWS:

1. Build the Docker image:
   ```
   docker build -t mlops-python-api .
   ```

2. Tag the image for Amazon ECR:
   ```
   docker tag mlops-python-api:latest <your-aws-account-id>.dkr.ecr.<region>.amazonaws.com/mlops-python-api:latest
   ```

3. Push the image to Amazon ECR:
   ```
   aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <your-aws-account-id>.dkr.ecr.<region>.amazonaws.com
   docker push <your-aws-account-id>.dkr.ecr.<region>.amazonaws.com/mlops-python-api:latest
   ```

4. Deploy using AWS ECS or EKS
"""
    elif platform == 'azure':
        readme_content += """
## Azure Deployment

To deploy to Azure:

1. Build the Docker image:
   ```
   docker build -t mlops-python-api .
   ```

2. Tag the image for Azure Container Registry:
   ```
   docker tag mlops-python-api:latest <your-registry-name>.azurecr.io/mlops-python-api:latest
   ```

3. Push the image to Azure Container Registry:
   ```
   az acr login --name <your-registry-name>
   docker push <your-registry-name>.azurecr.io/mlops-python-api:latest
   ```

4. Deploy using Azure Container Instances or AKS
"""
    elif platform == 'gcp':
        readme_content += """
## GCP Deployment

To deploy to Google Cloud:

1. Build the Docker image:
   ```
   docker build -t mlops-python-api .
   ```

2. Tag the image for Google Container Registry:
   ```
   docker tag mlops-python-api:latest gcr.io/<your-project-id>/mlops-python-api:latest
   ```

3. Push the image to Google Container Registry:
   ```
   gcloud auth configure-docker
   docker push gcr.io/<your-project-id>/mlops-python-api:latest
   ```

4. Deploy using Google Cloud Run or GKE
"""
    
    # Write README file
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(readme_content)
    
    print(f"Docker deployment files created successfully in {output_dir}")


def main():
    """Main function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Model deployment script")
    parser.add_argument("--package", action="store_true", help="Package the model")
    parser.add_argument("--docker", action="store_true", help="Create Docker deployment")
    parser.add_argument("--model-dir", type=str, help="Directory containing model files")
    parser.add_argument("--output-dir", type=str, help="Output directory for deployment files")
    parser.add_argument("--config", type=str, help="Configuration file to include")
    parser.add_argument("--platform", type=str, choices=["aws", "azure", "gcp"], help="Target platform for deployment")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config()
    
    # Set default directories if not provided
    if not args.model_dir:
        args.model_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'bin')
    if not args.output_dir:
        args.output_dir = os.path.join(os.path.dirname(__file__), '..', 'deployment', 'package')
    
    try:
        # Run selected operations
        if args.package:
            package_info = package_model(args.model_dir, args.output_dir, args.config)
            print(f"Model packaged successfully: {package_info}")
        
        if args.docker:
            if not args.package:
                # If packaging wasn't done, check if output directory has the model files
                model_file = os.path.join(args.output_dir, config['paths']['models']['model_file'])
                if not os.path.exists(model_file):
                    print("Warning: Model files not found in output directory. Please run packaging first.")
            
            docker_output_dir = os.path.join(os.path.dirname(__file__), '..', 'deployment', 'docker')
            create_docker_deployment(args.output_dir, docker_output_dir, args.platform)
        
        # If no operation selected, print help
        if not (args.package or args.docker):
            parser.print_help()
            
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
