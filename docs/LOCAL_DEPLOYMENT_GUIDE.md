# MLOps Local Deployment Guide

This guide walks you through setting up the complete MLOps pipeline locally using Kubernetes orchestration.

## 🎯 Overview

The local MLOps deployment includes:
- **MLflow Server**: Experiment tracking and model registry
- **Streamlit Frontend**: Interactive web application for image classification
- **Training Pipeline**: Kubernetes-based ML model training
- **Monitoring**: Data drift detection and model performance monitoring

## 📋 Prerequisites

### Required Tools
- **Docker Desktop** (with Kubernetes enabled) or **Minikube**
- **kubectl** (Kubernetes CLI)
- **Python 3.8+**

### Installation
1. **Docker Desktop**: Download from [docker.com](https://www.docker.com/products/docker-desktop)
   - Enable Kubernetes in Docker Desktop settings
2. **kubectl**: Install from [kubernetes.io](https://kubernetes.io/docs/tasks/tools/)
3. **Python**: Download from [python.org](https://www.python.org/downloads/)

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)
```bash
# Complete setup with one command
python setup_local_mlops.py
```

### Option 2: Manual Step-by-Step Setup
```bash
# 1. Build containers
python scripts/build_containers.py

# 2. Deploy to Kubernetes
python scripts/deploy_local.py

# 3. Access the applications
# Frontend: http://localhost:8501
# MLflow: http://localhost:5000
```

## 📦 Container Components

### 1. Frontend Container (Streamlit)
- **Image**: `potato-disease-frontend:latest`
- **Port**: 8501
- **Features**: Image upload, classification, model comparison

### 2. MLflow Server Container
- **Image**: `potato-disease-mlflow:latest`
- **Port**: 5000
- **Features**: Experiment tracking, model registry, artifact storage

### 3. Training Container
- **Image**: `potato-disease-training:latest`
- **Features**: ML pipeline execution, model training, evaluation

## 🔧 Configuration

### Environment Variables
```bash
# MLflow configuration
MLFLOW_TRACKING_URI=http://mlflow-server:5000
MLFLOW_BACKEND_STORE_URI=sqlite:///data/mlflow.db
MLFLOW_DEFAULT_ARTIFACT_ROOT=/data/mlflow/artifacts

# Application settings
PYTHONPATH=/app
STREAMLIT_SERVER_HEADLESS=true
```

### Kubernetes Resources
- **Namespace**: `potato-disease-ml`
- **Persistent Volumes**: MLflow data storage
- **Services**: ClusterIP for internal communication
- **Deployments**: Frontend and MLflow server

## 🌐 Access URLs

After successful deployment:
- **Frontend**: http://localhost:8501
- **MLflow Server**: http://localhost:5000

## 💡 Usage Examples

### 1. Image Classification
1. Open the frontend at http://localhost:8501
2. Upload a potato leaf image
3. View classification results and confidence scores
4. Compare different model predictions

### 2. Model Training
```bash
# Run training pipeline
kubectl apply -f kubernetes/ml-pipeline-jobs.yaml

# Monitor training progress
kubectl logs -f job/training-pipeline -n potato-disease-ml
```

### 3. Experiment Tracking
1. Open MLflow at http://localhost:5000
2. View experiments and runs
3. Compare model metrics
4. Download model artifacts

## 🔧 Management Commands

### Check Status
```bash
python setup_local_mlops.py --status
```

### Build Only Containers
```bash
python setup_local_mlops.py --build-only
```

### Cleanup Deployment
```bash
python setup_local_mlops.py --cleanup
```

### Manual Kubernetes Commands
```bash
# View pods
kubectl get pods -n potato-disease-ml

# View services
kubectl get services -n potato-disease-ml

# View logs
kubectl logs deployment/frontend -n potato-disease-ml

# Port forward manually
kubectl port-forward service/frontend 8501:8501 -n potato-disease-ml
```

## 📊 Monitoring and Logging

### View Application Logs
```bash
# Frontend logs
kubectl logs deployment/frontend -n potato-disease-ml -f

# MLflow logs
kubectl logs deployment/mlflow-server -n potato-disease-ml -f

# Training job logs
kubectl logs job/training-pipeline -n potato-disease-ml -f
```

### Resource Usage
```bash
# Pod resource usage
kubectl top pods -n potato-disease-ml

# Node resource usage
kubectl top nodes
```

## 🛠️ Troubleshooting

### Common Issues

#### 1. Pods Not Starting
```bash
# Check pod status
kubectl describe pod <pod-name> -n potato-disease-ml

# Check events
kubectl get events -n potato-disease-ml --sort-by='.lastTimestamp'
```

#### 2. Images Not Found
```bash
# For minikube users
minikube image load potato-disease-frontend:latest
minikube image load potato-disease-mlflow:latest
minikube image load potato-disease-training:latest
```

#### 3. Port Forwarding Issues
```bash
# Kill existing port forwarding
pkill -f "kubectl port-forward"

# Restart port forwarding
kubectl port-forward service/frontend 8501:8501 -n potato-disease-ml
```

#### 4. Storage Issues
```bash
# Check persistent volumes
kubectl get pv,pvc -n potato-disease-ml

# Recreate persistent volumes
kubectl delete pvc mlflow-pvc -n potato-disease-ml
kubectl apply -f kubernetes/persistent-volumes.yaml
```

### Performance Tuning

#### Resource Limits
Edit the deployment files to adjust resource limits:
```yaml
resources:
  requests:
    memory: "512Mi"
    cpu: "250m"
  limits:
    memory: "1Gi"
    cpu: "500m"
```

#### Scaling
```bash
# Scale frontend replicas
kubectl scale deployment frontend --replicas=3 -n potato-disease-ml

# Scale MLflow replicas
kubectl scale deployment mlflow-server --replicas=2 -n potato-disease-ml
```

## 🔄 CI/CD Integration

The project includes GitHub Actions workflow for automated:
- Code quality checks
- Container building
- Model training
- Deployment artifact generation

Workflow file: `.github/workflows/mlops-local-pipeline.yml`

## 📁 Project Structure

```
mlops-python/
├── src/                    # Application source code
├── scripts/               # Deployment and utility scripts
├── kubernetes/            # Kubernetes manifests
├── configs/              # Configuration files
├── data/                 # Data directories
├── models/               # Model storage
├── Dockerfile.*          # Container definitions
├── docker-compose.yml    # Docker Compose configuration
└── setup_local_mlops.py  # Main setup script
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test locally using the setup scripts
5. Submit a pull request

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review Kubernetes logs
- Contact: qacifsaad@gmail.com

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Happy MLOps! 🚀**
