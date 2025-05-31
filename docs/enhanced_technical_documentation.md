# MLOps Python Project - Enhanced Technical Documentation

## Overview

This document provides technical documentation for the MLOps Python project, a comprehensive machine learning operations system for image processing and classification. The project replaces a mixed Python/Rust implementation with a pure Python solution, following MLOps best practices for maintainability, scalability, and reproducibility.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Components](#components)
   - [Image Processing](#image-processing)
   - [Segmentation](#segmentation)
   - [Feature Extraction](#feature-extraction)
   - [Model Training](#model-training)
   - [Model Evaluation](#model-evaluation)
   - [Model Tracking](#model-tracking)
   - [Model Monitoring](#model-monitoring)
   - [API](#api)
3. [Workflow](#workflow)
4. [Deployment](#deployment)
   - [Docker Deployment](#docker-deployment)
   - [CI/CD Pipeline](#ci-cd-pipeline)
5. [Best Practices](#best-practices)
6. [Troubleshooting](#troubleshooting)

## Project Structure

```
mlops-python/
├── configs/                # Configuration files
│   └── config.json         # Main configuration
├── data/                   # Data directory
│   ├── inputs/             # Input images
│   ├── outputs/            # Output predictions
│   └── uploads/            # API uploads
├── deployment/             # Deployment configurations
│   ├── development/        # Development environment
│   ├── testing/            # Testing environment
│   └── production/         # Production environment
├── docs/                   # Documentation
├── logs/                   # Log files
├── models/                 # Model files
│   ├── bin/                # Serialized models
│   └── tracking/           # Model tracking data
├── notebooks/              # Jupyter notebooks
│   ├── data_exploration.ipynb
│   └── model_training.ipynb
├── scripts/                # Utility scripts
│   ├── copy_original_data.ps1
│   ├── copy_original_data.sh
│   ├── deploy_model.py
│   ├── process_data.py
│   └── setup_environment.py
├── src/                    # Source code
│   ├── api.py              # Flask API
│   ├── evaluate_model.py   # Model evaluation
│   ├── image_processing.py # Image processing
│   ├── main.py             # Main application
│   ├── model_monitoring.py # Model monitoring
│   ├── model_tracking.py   # Model version tracking
│   ├── segmentation.py     # Image segmentation
│   ├── train_model.py      # Model training
│   └── utils.py            # Utilities
├── tests/                  # Test files
│   └── test_modules.py
├── visualizations/         # Visualization outputs
├── .env                    # Environment variables
├── .github/workflows/      # GitHub Actions workflows
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker Compose
└── requirements.txt        # Python dependencies
```

## Components

### Image Processing

The `image_processing.py` module provides functionality for processing images, including:

- **Contour Detection**: Detects edges and contours in images
- **CLAHE Enhancement**: Applies Contrast Limited Adaptive Histogram Equalization
- **Color Channel Processing**: Processes RGB color channels separately

### Segmentation

The `segmentation.py` module provides functionality for segmenting images:

- **BoundingRect Class**: Represents a rectangular region in an image
- **Contour Detection**: Detects contours in an image using OpenCV
- **Image Segmentation**: Segments images based on contour detection

### Feature Extraction

The `utils.py` module provides functionality for extracting features from images, which are used for classification:

- **calcul_dev Function**: Calculates statistical features from image arrays
- **Histogram Features**: Extracts histogram-based features
- **Statistical Features**: Calculates statistical measures (mean, std, etc.)

### Model Training

The `train_model.py` module provides functionality for training machine learning models:

- **Feature Extraction**: Extracts features from images
- **Hyperparameter Tuning**: Uses GridSearchCV for hyperparameter optimization
- **Model Training**: Trains RandomForest and other models
- **Model Persistence**: Saves trained models and encoders

Enhancements:
- **Cross-validation**: Implements k-fold cross-validation
- **Feature Importance**: Visualizes feature importance
- **Model Versioning**: Integrates with the model tracking system
- **Performance Metrics**: Calculates and visualizes detailed metrics

### Model Evaluation

The `evaluate_model.py` module provides functionality for evaluating trained models:

- **Performance Metrics**: Calculates accuracy, precision, recall, F1-score
- **Confusion Matrix**: Visualizes confusion matrix
- **ROC Curves**: Plots Receiver Operating Characteristic curves
- **Visualization**: Creates visualizations of model performance

### Model Tracking

The `model_tracking.py` module provides a lightweight model tracking system similar to MLflow:

- **Run Tracking**: Tracks training runs with unique IDs
- **Parameter Logging**: Logs hyperparameters and configurations
- **Metric Logging**: Logs performance metrics during training
- **Artifact Management**: Stores model files and metadata
- **Model Comparison**: Compares models based on metrics
- **Production Promotion**: Promotes best models to production

### Model Monitoring

The `model_monitoring.py` module provides continuous monitoring of deployed models:

- **Performance Tracking**: Monitors prediction accuracy in production
- **Drift Detection**: Detects feature and prediction distribution drift
- **Latency Monitoring**: Tracks prediction latency
- **Alerting**: Logs warnings when performance degrades
- **Statistics**: Maintains statistics on predictions and performance

### API

The `api.py` module provides a RESTful API for serving the trained model:

- **Flask-RESTx**: Uses Flask-RESTx for API documentation
- **Swagger UI**: Provides interactive API documentation at `/docs`
- **Prediction Endpoints**: Endpoints for single image and batch prediction
- **File Upload**: Handles image file uploads
- **Error Handling**: Comprehensive error handling
- **Monitoring**: Integrates with the API monitoring system

Enhancements:
- **API Documentation**: Interactive API documentation with Swagger UI
- **Request Validation**: Validates incoming requests
- **Access Control**: API key authentication for admin endpoints
- **Performance Metrics**: Endpoint for retrieving API performance metrics
- **Visualization Access**: Endpoint for accessing model visualizations

## Workflow

The typical workflow for this project is:

1. **Setup Environment**: Run `scripts/setup_environment.py` to set up the environment
2. **Process Data**: Prepare and process images using the image processing modules
3. **Train Model**: Train models using `train_model.py` or Jupyter notebooks
4. **Evaluate Model**: Evaluate models using `evaluate_model.py`
5. **Deploy Model**: Deploy models using `deploy_model.py`
6. **Serve Predictions**: Serve predictions using the API

## Deployment

### Docker Deployment

The project includes Docker and Docker Compose configurations for easy deployment:

- **API Service**: Serves the API for predictions
- **Batch Processing**: Processes batches of images
- **Model Training**: Trains models in containerized environment
- **Model Evaluation**: Evaluates models in containerized environment

To deploy with Docker Compose:

```bash
# Start API service
docker-compose up api

# Run model training
docker-compose --profile training up

# Run batch processing
docker-compose --profile batch up
```

### CI/CD Pipeline

The project includes a GitHub Actions workflow for continuous integration and deployment:

- **Test**: Runs tests on multiple Python versions
- **Build**: Builds Docker images
- **Deploy to Dev**: Deploys to development environment
- **Deploy to Prod**: Deploys to production environment

## Best Practices

The project follows these MLOps best practices:

1. **Modular Design**: Separates concerns into different modules
2. **Configuration Management**: Uses centralized configuration
3. **Versioning**: Version controls code, models, and data
4. **Reproducibility**: Makes training and evaluation reproducible
5. **Monitoring**: Monitors model and API performance
6. **Containerization**: Uses Docker for consistent environments
7. **CI/CD**: Implements continuous integration and deployment
8. **Documentation**: Provides comprehensive documentation
9. **Testing**: Includes automated tests for components

## Troubleshooting

Common issues and solutions:

1. **Model Loading Errors**:
   - Check if model files exist in the correct location
   - Verify that the model format is compatible

2. **API Connection Issues**:
   - Check if the API service is running
   - Verify that ports are correctly exposed and not blocked

3. **Image Processing Errors**:
   - Ensure images are in a supported format (PNG, JPEG)
   - Check that images are not corrupted

4. **Docker Issues**:
   - Ensure Docker and Docker Compose are installed
   - Check that ports are not already in use

5. **Memory Issues**:
   - Adjust Docker container memory limits
   - Use batch processing for large datasets
