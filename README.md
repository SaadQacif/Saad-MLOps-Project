# Potato Disease Classification MLOps Project

This repository contains a comprehensive Python-based MLOps project for potato disease image classification. It implements a complete MLOps workflow with proper project structure, MLflow model tracking, Kubernetes deployment, and Docker containerization. The system classifies potato leaf images into three categories: Healthy, Early Blight, and Late Blight.

## Project Structure

```
mlops-python/
├── configs/              # Configuration files
├── data/                 # Data directory
│   ├── inputs/           # Input images
│   ├── outputs/          # Output predictions
│   └── uploads/          # API uploads
├── deployment/           # Deployment configurations
├── docs/                 # Documentation
├── logs/                 # Log files
├── models/               # Trained models
│   ├── bin/              # Serialized model files
│   └── tracking/         # Model tracking data
├── notebooks/            # Jupyter notebooks
├── scripts/              # Utility scripts
├── src/                  # Source code
│   ├── api.py            # Flask API with Swagger documentation
│   ├── evaluate_model.py # Model evaluation script
│   ├── image_processing.py # Image processing functions
│   ├── main.py           # Main script for batch processing
│   ├── model_monitoring.py # Model monitoring system
│   ├── model_tracking.py # MLFlow-like model tracking
│   ├── monitoring.py     # API monitoring system
│   ├── segmentation.py   # Image segmentation module
│   ├── train_model.py    # Model training with hyperparameter tuning
│   └── utils.py          # Utility functions
├── tests/                # Test files
├── visualizations/       # Visualization outputs
├── .github/workflows/    # GitHub Actions CI/CD
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose configuration
└── requirements.txt      # Python dependencies
```

## Features

- **Image Processing**
  - Advanced image segmentation and contour detection
  - Feature extraction from images using statistical techniques
  - CLAHE enhancement for better feature detection

- **Machine Learning**
  - Automated hyperparameter tuning with GridSearchCV
  - Comprehensive model evaluation with multiple metrics
  - Feature importance analysis and visualization
  - Model tracking and versioning system similar to MLFlow

- **Monitoring & Observability**
  - Real-time model performance monitoring
  - Feature and prediction drift detection
  - API usage and performance tracking
  - Comprehensive logging system

- **API & Deployment**
  - RESTful API with Swagger documentation
  - Multi-environment deployment (dev, test, prod)
  - Dockerized services with Docker Compose
  - CI/CD pipeline with GitHub Actions

- **MLOps Best Practices**
  - Configuration management
  - Model versioning and governance
  - Reproducible training and evaluation
  - Environment consistency with containers

## Requirements

- Python 3.8+ (3.10+ recommended)
- Docker and Docker Compose
- Required Python packages (see `requirements.txt`)
- 4GB+ RAM for model training

## Advanced Features

### Model Tracking System

The project includes a lightweight model tracking system similar to MLFlow:

```python
from model_tracking import MLFlowLikeTracker

# Initialize tracker
tracker = MLFlowLikeTracker()
run_id = tracker.start_run("training_run")

# Log parameters
tracker.log_params(run_id, {"n_estimators": 100, "max_depth": 10})

# Log metrics
tracker.log_metric(run_id, "accuracy", 0.95)

# Log model
tracker.log_model(run_id, model, "model.pkl")

# End run
tracker.end_run(run_id)

# Get best model
best_run_id, best_run = tracker.get_best_run("accuracy", "max")
```

### Model Monitoring

The system includes real-time model monitoring:

```python
from model_monitoring import ModelMonitor

# Initialize monitor
monitor = ModelMonitor(model_path="models/bin/model.pkl", 
                       encoder_path="models/bin/encoder.pkl")

# Log predictions
monitor.log_prediction(features=features, prediction=prediction, latency=0.05)

# Check drift
drift_metrics = monitor.check_drift(recent_features=recent_batch)
if drift_metrics["feature_drift"]["drift_detected"]:
    send_alert("Feature drift detected!")
```

### Interactive API Documentation

The API includes Swagger UI documentation at the `/docs` endpoint, making it easy to explore and test endpoints.

## Importing Data from Original MLOps Project

To import data from the original MLOps project, use the following PowerShell commands:

```powershell
# Create necessary directories if they don't exist
New-Item -Path "c:\Users\PC\Desktop\mlops-python\data\inputs" -ItemType Directory -Force
New-Item -Path "c:\Users\PC\Desktop\mlops-python\models\bin" -ItemType Directory -Force

# Copy input images from original project
Copy-Item -Path "C:\Users\PC\Desktop\MLOps\inputs\*" -Destination "c:\Users\PC\Desktop\mlops-python\data\inputs\" -Recurse -Force

# Copy trained models from original project
Copy-Item -Path "C:\Users\PC\Desktop\MLOps\bin\*" -Destination "c:\Users\PC\Desktop\mlops-python\models\bin\" -Force
```

For Unix/Linux/macOS:

```bash
# Create necessary directories if they don't exist
mkdir -p data/inputs
mkdir -p models/bin

# Copy input images from original project
cp -r /path/to/original/MLOps/inputs/* data/inputs/

# Copy trained models from original project
cp -r /path/to/original/MLOps/bin/* models/bin/
```

## Installation

### Option 1: Local Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Option 2: Docker

1. Clone the repository
2. Build and run using Docker Compose:
   ```
   docker-compose up
   ```

## Usage

### API Service

Start the API service:

```bash
docker-compose up api
```

This will start the Flask API on port 5000, which can be accessed at http://localhost:5000.

### Batch Processing

To process a batch of images:

```bash
docker-compose --profile batch up
```

### Model Training

To train a new model:

```bash
docker-compose --profile training up
```

### Model Evaluation

To evaluate the model:

```bash
docker-compose --profile evaluation up
```

## Usage Workflow

### 1. Data Preprocessing

The first step is to preprocess the raw potato disease images:

```bash
python scripts/preprocess_potato_data.py --input Potato_Health_States --output data/processed
```

This will create properly formatted image datasets for the three classes: Early Blight, Late Blight, and Healthy.

### 2. Feature Extraction

Extract meaningful features from the processed images:

```bash
python scripts/extract_potato_features.py --input data/processed --output data/features/potato_features.csv
```

This extracts various features including:
- RGB channel statistics
- Entropy features
- Shape features (convexity ratio, eccentricity)
- Reduced HOG features

### 3. Model Training with MLflow

Start the MLflow server:

```bash
mlflow server --host 0.0.0.0 --port 5001 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

Train the model with MLflow tracking:

```bash
python src/train_model_features.py --features data/features/potato_features.csv --model-dir models/bin --mlflow-tracking-uri http://localhost:5001
```

View the MLflow dashboard at http://localhost:5001 to track experiments, compare models, and manage the model registry.

### 4. Docker Deployment

Build and run the Docker containers:

```bash
docker-compose up --build
```

### 5. Kubernetes Deployment

Deploy the application to Kubernetes:

```bash
python scripts/test_kubernetes_deployment.py
```

Alternatively, use kubectl directly:

```bash
kubectl apply -f kubernetes/deployment.yaml
```

### 6. Running Tests

Execute the test suite:

```bash
python -m unittest discover tests
```

## API Endpoints

- `GET /health`: Check API health and model status
- `POST /predict`: Submit a single image for prediction
- `POST /batch_predict`: Submit multiple images for prediction

## Example API Usage

```python
import requests

# Single image prediction
url = 'http://localhost:5000/predict'
files = {'image': open('path/to/image.jpg', 'rb')}
response = requests.post(url, files=files)
result = response.json()
print(result)

# Batch prediction
url = 'http://localhost:5000/batch_predict'
files = [
    ('images', open('path/to/image1.jpg', 'rb')),
    ('images', open('path/to/image2.jpg', 'rb'))
]
response = requests.post(url, files=files)
results = response.json()
print(results)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## System Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌────────────────┐
│                 │     │              │     │                │
│  Preprocessing  │────▶│   Feature    │────▶│  Model         │
│  & Data         │     │  Extraction  │     │  Training      │
│  Preparation    │     │              │     │  with MLflow   │
│                 │     │              │     │                │
└─────────────────┘     └──────────────┘     └────────┬───────┘
                                                     │
                                                     ▼
┌─────────────────┐     ┌──────────────┐     ┌────────────────┐
│                 │     │              │     │                │
│   CI/CD         │◀────│  Kubernetes  │◀────│  Docker        │
│   Pipeline      │     │  Deployment  │     │  Containerization│
│   (GitHub       │     │              │     │                │
│   Actions)      │     │              │     │                │
└─────────────────┘     └──────────────┘     └────────────────┘
```
