#!/usr/bin/env python
"""
Script to deploy and test the Kubernetes configuration locally.
"""

import os
import sys
import time
import subprocess
import requests
from pathlib import Path

def run_command(command):
    """Run a shell command and print output."""
    print(f"Running: {' '.join(command)}")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    return process.returncode

def build_docker_images():
    """Build Docker images for the application and MLflow."""
    print("\n=== Building Docker Images ===\n")
    
    # Build main application image
    app_result = run_command(["docker", "build", "-t", "mlops-python:latest", "."])
    
    if app_result != 0:
        print("Error building application Docker image")
        return False
    
    # Build MLflow image
    mlflow_result = run_command(["docker", "build", "-t", "mlops-python-mlflow:latest", "-f", "Dockerfile.mlflow", "."])
    
    if mlflow_result != 0:
        print("Error building MLflow Docker image")
        return False
    
    print("Docker images built successfully")
    return True

def check_kubernetes_cluster():
    """Check if Kubernetes cluster is running and load images into it."""
    print("\n=== Checking Kubernetes Cluster ===\n")
    
    # Check if minikube is available
    try:
        subprocess.run(["minikube", "version"], check=True, capture_output=True)
        use_minikube = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        use_minikube = False
        
    # Check if kind is available if minikube is not
    if not use_minikube:
        try:
            subprocess.run(["kind", "version"], check=True, capture_output=True)
            use_kind = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            use_kind = False
            print("Neither minikube nor kind is available. Please install one of them.")
            return False
    
    # Start the cluster and load images
    if use_minikube:
        # Check if minikube is running
        status = subprocess.run(["minikube", "status"], capture_output=True, text=True)
        
        if "Running" not in status.stdout:
            run_command(["minikube", "start"])
        
        # Load the Docker images into minikube
        run_command(["minikube", "image", "load", "mlops-python:latest"])
        run_command(["minikube", "image", "load", "mlops-python-mlflow:latest"])
    
    elif use_kind:
        # Check if kind cluster exists
        clusters = subprocess.run(["kind", "get", "clusters"], capture_output=True, text=True)
        
        if "mlops-cluster" not in clusters.stdout:
            run_command(["kind", "create", "cluster", "--name", "mlops-cluster"])
        
        # Load the Docker images into kind
        run_command(["kind", "load", "docker-image", "mlops-python:latest", "--name", "mlops-cluster"])
        run_command(["kind", "load", "docker-image", "mlops-python-mlflow:latest", "--name", "mlops-cluster"])
    
    print("Kubernetes cluster is ready")
    return True

def deploy_to_kubernetes():
    """Deploy the application to Kubernetes."""
    print("\n=== Deploying to Kubernetes ===\n")
    
    # Create namespace if it doesn't exist
    run_command(["kubectl", "apply", "-f", "kubernetes/deployment.yaml"])
    
    # Check if deployments are running
    print("\nWaiting for deployments to be ready...")
    max_attempts = 20
    
    for attempt in range(max_attempts):
        time.sleep(3)
        status = subprocess.run(
            ["kubectl", "get", "pods", "-n", "potato-disease-classification"],
            capture_output=True,
            text=True
        )
        
        if "Running" in status.stdout and "mlflow-server" in status.stdout and "model-api" in status.stdout:
            print("Deployments are ready!")
            return True
        
        print(f"Waiting for deployments to be ready (attempt {attempt + 1}/{max_attempts})...")
    
    print("Deployments are not ready after waiting. Please check kubectl logs for more information.")
    return False

def port_forward_services():
    """Port-forward the MLflow service."""
    print("\n=== Setting up Port Forwarding ===\n")
    
    # Port-forward MLflow server (background process)
    mlflow_process = subprocess.Popen(
        ["kubectl", "port-forward", "-n", "potato-disease-classification", "service/mlflow-server", "5001:5000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    
    # Give some time for port forwarding to establish
    time.sleep(5)
    
    return mlflow_process

def test_services():
    """Test the deployed services."""
    print("\n=== Testing Services ===\n")
    
    # Test MLflow server
    try:
        mlflow_response = requests.get("http://localhost:5001")
        if mlflow_response.status_code == 200:
            print("✅ MLflow server is accessible")
        else:
            print(f"❌ MLflow server returned status code: {mlflow_response.status_code}")
    except requests.RequestException as e:
        print(f"❌ Error accessing MLflow server: {e}")
    
    # Test frontend (if port-forwarded)
    try:
        frontend_response = requests.get("http://localhost:8501/_stcore/health")
        if frontend_response.status_code == 200:
            print("✅ Frontend is accessible")
        else:
            print(f"❌ Frontend returned status code: {frontend_response.status_code}")
    except requests.RequestException as e:
        print(f"❌ Error accessing Frontend: {e}")

def cleanup(mlflow_process=None):
    """Clean up resources."""
    print("\n=== Cleaning Up ===\n")
    
    # Terminate port-forwarding processes
    if mlflow_process:
        mlflow_process.terminate()
    
    print("Port forwarding stopped")

def main():
    """Main function."""
    print("=== Potato Disease Classification K8s Deployment Tester ===")
    
    try:
        if not build_docker_images():
            return 1
        
        if not check_kubernetes_cluster():
            return 1
        if not deploy_to_kubernetes():
            return 1
        
        mlflow_process = port_forward_services()
        
        test_services()
        
        input("\nPress Enter to stop the services and clean up...\n")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return 1
    
    finally:
        cleanup(mlflow_process)
    
    print("Test completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
