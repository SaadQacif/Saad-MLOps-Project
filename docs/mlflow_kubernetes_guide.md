# MLflow and Kubernetes Integration Guide

This document provides instructions on how to use MLflow for model tracking and Kubernetes for deployment in this MLOps project.

## Table of Contents
- [Prerequisites](#prerequisites)
- [MLflow Setup](#mlflow-setup)
- [Docker Setup](#docker-setup)
- [Kubernetes Setup](#kubernetes-setup)
- [Model Training with MLflow](#model-training-with-mlflow)
- [Model Deployment](#model-deployment)
- [CI/CD with GitHub Actions](#cicd-with-github-actions)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- Python 3.8+ (3.10 recommended)
- Docker
- Kubernetes (minikube, Docker Desktop Kubernetes, or other local distribution)
- kubectl command-line tool

## MLflow Setup

MLflow is used for tracking experiments, models, and metrics. The project includes a fully local MLflow setup that stores data in a SQLite database and local filesystem.

### Starting MLflow Server

Run the setup script which configures and starts the MLflow server:

**On Windows:**
```
.\setup.ps1
```

**On Linux/macOS:**
```
./setup.sh
```

Alternatively, you can start MLflow manually:
```
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 \
  --port 5001
```

### MLflow UI

Once the server is running, access the MLflow UI at: http://localhost:5001

## Docker Setup

The project includes three Dockerfiles:
1. `Dockerfile` - Main application image for API, training, and evaluation
2. `Dockerfile.mlflow` - Dedicated MLflow server image
3. `Dockerfile.frontend` - Streamlit frontend application with MLflow tracing

### Building Docker Images

Build the images using Docker:

```
docker build -t mlops-python-api .
docker build -t mlops-python-mlflow -f Dockerfile.mlflow .
docker build -t mlops-python-frontend -f Dockerfile.frontend .
```

Alternatively, use Docker Compose to build and run all services:

```
# Build all services
docker-compose build

# Run all services
docker-compose --profile all up
```

### Docker Compose Profiles

The project uses Docker Compose profiles to manage different service combinations:

- `mlflow`: Just the MLflow server
- `api`: API service with MLflow server
- `frontend`: Streamlit frontend with MLflow server
- `training`: Model training with MLflow server
- `all`: All services together

## Kubernetes Setup

Kubernetes is used for deploying and scaling the application in a container environment.

### Deploying to Kubernetes

1. Make sure your Kubernetes cluster is running (minikube, Docker Desktop, etc.)
2. Apply the deployment manifests:

```
kubectl apply -f kubernetes/deployment.yaml
```

This creates:
- Namespace: `potato-disease-classification`
- Deployments: `mlflow-server`, `model-api`, and `frontend`
- Services: exposing all components

The deployment configures MLflow tracing across all components with the following environment variables:
```
MLFLOW_TRACKING_URI=http://mlflow-server:5001
MLFLOW_ENABLE_TRACES=true
```

### Checking Deployment Status

```
kubectl get pods -n potato-disease-classification
kubectl get services -n potato-disease-classification
```

## Model Training with MLflow

The project now uses MLflow to track model training experiments. All metrics, parameters, and artifacts are logged to MLflow.

### Training a Model

Using the standard workflow:
```
python scripts/mlops_workflow_fixed.py
```

Or using Docker:
```
docker-compose --profile training up
```

This automatically:
1. Creates an MLflow experiment if it doesn't exist
2. Starts a new run
3. Logs parameters, metrics, and artifacts
4. Registers the model in the MLflow Model Registry
5. Promotes the model to production if it's better than existing ones
6. Traces all operations with distributed tracing enabled

### Enhanced MLflow Tracking Features

The MLflow integration includes:

1. **Input Feature Tracking** - All input features are tracked in MLflow 
2. **Visualization Artifacts** - Confusion matrices and evaluation plots are stored
3. **Model Signatures** - Model input/output schema is tracked
4. **Prediction Monitoring** - Frontend and API predictions are logged
5. **Distributed Tracing** - Operations across services are traced
6. **Parallel Experiment Runs** - Support for comparing different model configurations

## Model Deployment

Once a model is trained and registered in MLflow, it can be deployed:

### Local Deployment

```
docker-compose up api
```

### Kubernetes Deployment

```
python scripts/k8s_deploy.py deploy
```

## CI/CD with GitHub Actions

The project includes a GitHub Actions workflow that:
1. Runs tests on every push and pull request
2. Builds Docker images when changes are pushed to main
3. (Optionally) Deploys to a Kubernetes cluster

The workflow is defined in `.github/workflows/ci-cd.yml`.

## Troubleshooting

### MLflow Issues

If you encounter issues with MLflow:
- Check if the MLflow server is running (`ps aux | grep mlflow` or Task Manager)
- Verify the MLflow SQLite database exists and has permissions
- Check that port 5001 is not being used by another application

#### MLflow Connection Issues

To verify MLflow connection:

```bash
# Check if MLflow server is accessible
curl http://localhost:5001

# Check environment variable in container
docker-compose exec frontend env | grep MLFLOW

# Verify MLflow experiment creation 
python -c "import mlflow; print(mlflow.get_experiment_by_name('potato-disease-classification'))"
```

#### Missing Logs or Artifacts

If MLflow runs are not recording properly:
- Check the `mlruns` directory permissions
- Verify that `MLFLOW_TRACKING_URI` is set correctly
- Ensure model training completes successfully

#### Tracing Issues

If distributed tracing isn't working:
- Verify `MLFLOW_ENABLE_TRACES=true` is set in all services
- Check `configs/mlflow_tracing.py` is properly imported
- Look for any import errors in your application logs

### Kubernetes Issues

Common Kubernetes issues:
- Images not found: Make sure you've built and loaded the images correctly
- PersistentVolume issues: Verify filesystem permissions
- Service not reachable: Check service NodePort and firewall settings

For more help, run:
```
kubectl describe pods -n potato-disease-classification
```
