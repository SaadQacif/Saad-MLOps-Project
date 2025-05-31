#!/usr/bin/env python3
"""
Docker container build script for MLOps pipeline.
Builds all necessary containers for local deployment.
"""

import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_command(command, cwd=None):
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"✅ Command succeeded: {command}")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Command failed: {command}")
        logger.error(f"Error: {e.stderr}")
        return False, e.stderr


def build_frontend_container():
    """Build the frontend Streamlit container."""
    logger.info("Building frontend container...")
    
    dockerfile_content = """FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    libglib2.0-0 \\
    libsm6 \\
    libxext6 \\
    libxrender-dev \\
    libgomp1 \\
    libgl1-mesa-glx \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional frontend dependencies
RUN pip install --no-cache-dir streamlit pillow opentelemetry-api opentelemetry-sdk

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/
COPY models/ ./models/
COPY data/ ./data/

# Create necessary directories
RUN mkdir -p /app/tmp /app/visualizations /app/mlruns

# Set environment variables
ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=http://mlflow-server:5000
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false

# Expose port for Streamlit
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Command to run the Streamlit app
CMD ["streamlit", "run", "src/frontend.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
"""
    
    # Write Dockerfile.frontend
    with open("Dockerfile.frontend", "w") as f:
        f.write(dockerfile_content)
    
    # Build container
    success, output = run_command("docker build -f Dockerfile.frontend -t potato-disease-frontend:latest .")
    return success


def build_mlflow_container():
    """Build the MLflow server container."""
    logger.info("Building MLflow container...")
    
    dockerfile_content = """FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
RUN pip install mlflow==2.8.1 psycopg2-binary==2.9.7 boto3==1.29.7

# Create required directories
RUN mkdir -p /data/mlflow/artifacts

# Set environment variables
ENV MLFLOW_BACKEND_STORE_URI=sqlite:///data/mlflow.db
ENV MLFLOW_DEFAULT_ARTIFACT_ROOT=/data/mlflow/artifacts
ENV MLFLOW_SERVE_ARTIFACTS=true

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \\
    CMD curl -f http://localhost:5000/health || exit 1

# Command to run the MLflow server
CMD mlflow server \\
    --host 0.0.0.0 \\
    --port 5000 \\
    --backend-store-uri $MLFLOW_BACKEND_STORE_URI \\
    --default-artifact-root $MLFLOW_DEFAULT_ARTIFACT_ROOT \\
    --serve-artifacts
"""
    
    # Write Dockerfile.mlflow
    with open("Dockerfile.mlflow", "w") as f:
        f.write(dockerfile_content)
    
    # Build container
    success, output = run_command("docker build -f Dockerfile.mlflow -t potato-disease-mlflow:latest .")
    return success


def build_training_container():
    """Build the training container."""
    logger.info("Building training container...")
    
    dockerfile_content = """FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    libglib2.0-0 \\
    libsm6 \\
    libxext6 \\
    libxrender-dev \\
    libgomp1 \\
    libgl1-mesa-glx \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY data/ ./data/
COPY models/ ./models/

# Create necessary directories
RUN mkdir -p /app/tmp /app/visualizations /app/mlruns

# Set environment variables
ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=http://mlflow-server:5000

# Default command - can be overridden
CMD ["python", "scripts/training_pipeline.py"]
"""
    
    # Write Dockerfile.training
    with open("Dockerfile.training", "w") as f:
        f.write(dockerfile_content)
    
    # Build container
    success, output = run_command("docker build -f Dockerfile.training -t potato-disease-training:latest .")
    return success


def build_all_containers():
    """Build all containers."""
    logger.info("🚀 Starting container build process...")
    
    containers = [
        ("frontend", build_frontend_container),
        ("mlflow", build_mlflow_container),
        ("training", build_training_container)
    ]
    
    results = {}
    
    for name, build_func in containers:
        logger.info(f"Building {name} container...")
        success = build_func()
        results[name] = success
        
        if success:
            logger.info(f"✅ {name} container built successfully")
        else:
            logger.error(f"❌ {name} container build failed")
    
    # Summary
    logger.info("📋 Build Summary:")
    for name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        logger.info(f"  {name}: {status}")
    
    all_success = all(results.values())
    if all_success:
        logger.info("🎉 All containers built successfully!")
        
        # List built images
        logger.info("📦 Built Docker images:")
        success, output = run_command("docker images potato-disease-*")
        if success:
            print(output)
    else:
        logger.error("❌ Some containers failed to build")
    
    return all_success


def tag_and_save_containers(version="latest"):
    """Tag and save containers for distribution."""
    logger.info(f"Tagging and saving containers with version {version}...")
    
    containers = ["frontend", "mlflow", "training"]
    
    for container in containers:
        # Tag with version
        success, _ = run_command(f"docker tag potato-disease-{container}:latest potato-disease-{container}:{version}")
        if not success:
            logger.error(f"Failed to tag {container} container")
            continue
        
        # Save to tar.gz
        success, _ = run_command(f"docker save potato-disease-{container}:{version} | gzip > {container}-{version}.tar.gz")
        if success:
            logger.info(f"✅ Saved {container} container to {container}-{version}.tar.gz")
        else:
            logger.error(f"❌ Failed to save {container} container")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Build Docker containers for MLOps pipeline")
    parser.add_argument("--save", action="store_true", help="Save containers as tar.gz files")
    parser.add_argument("--version", default="latest", help="Version tag for containers")
    parser.add_argument("--container", choices=["frontend", "mlflow", "training", "all"], 
                       default="all", help="Specific container to build")
    
    args = parser.parse_args()
    
    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    try:
        if args.container == "all":
            success = build_all_containers()
        elif args.container == "frontend":
            success = build_frontend_container()
        elif args.container == "mlflow":
            success = build_mlflow_container()
        elif args.container == "training":
            success = build_training_container()
        
        if success and args.save:
            tag_and_save_containers(args.version)
        
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
