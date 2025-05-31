# Using MLflow Traces in the Potato Disease Classification Project

This guide explains how to use and interpret MLflow traces in the Potato Disease Classification project to track the model training process and monitor predictions.

## What is MLflow Tracing?

MLflow traces provide detailed insights into the execution of your ML workflow, capturing:
- Input parameters
- Execution times
- Logging events
- Model predictions
- Feature values
- Error states

## Setting Up MLflow Tracing

In this project, MLflow tracing is configured in `configs/mlflow_tracing.py`. The tracing is enabled by default in the Docker and Kubernetes configurations.

To activate tracing when running locally:

```bash
# Windows PowerShell
$env:MLFLOW_ENABLE_TRACES="true"
$env:MLFLOW_TRACKING_URI="http://localhost:5001"

# Linux/macOS
export MLFLOW_ENABLE_TRACES=true
export MLFLOW_TRACKING_URI=http://localhost:5001
```

## Traces in the Frontend Application

The Streamlit frontend captures traces for each prediction:

1. **User Uploads Image**: The frontend logs the image metadata
2. **Feature Extraction**: Input features are tracked in MLflow
3. **Model Prediction**: The prediction and confidence scores are logged
4. **Results Display**: The display time is tracked

To view these traces:
1. Open the MLflow UI (http://localhost:5001)
2. Navigate to the "potato-disease-classification" experiment
3. Look for runs with the source name "frontend_prediction"
4. Click on a run to see detailed metrics and parameters

## Traces in Model Training

During model training, traces are captured for:

1. **Data Loading**: File access and preprocessing steps
2. **Feature Extraction**: Feature computation and normalization
3. **Model Training**: Algorithm execution and hyperparameter tuning
4. **Model Evaluation**: Metrics calculation

These traces help identify:
- Performance bottlenecks
- Data processing issues
- Error patterns

## Distributed Tracing Across Services

When running with Docker Compose or Kubernetes, traces follow requests across services:

1. Frontend → API → Model → Database
2. Training Job → Model Registry → Artifact Storage

This enables end-to-end monitoring of the ML workflow.

## Troubleshooting with Traces

When something goes wrong, traces can help identify the issue:

1. **Model Accuracy Issues**: Check feature extraction traces
2. **Performance Problems**: Look for slow execution spans
3. **API Failures**: Trace requests from frontend to backend

Example trace query in MLflow:

```python
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()
runs = client.search_runs(
    experiment_ids=["experiment_id"],
    filter_string="tags.mlflow.source.name = 'frontend_prediction'"
)

# Find runs with errors
error_runs = [run for run in runs if "error" in run.data.metrics]
```

## Common Trace Patterns

Look for these patterns in your MLflow traces:

1. **Successful Prediction**:
   - Image upload → Feature extraction → Model prediction → Result

2. **Training Pipeline**:
   - Data loading → Feature extraction → Train/test split → Model training → Evaluation

3. **Batch Prediction**:
   - Load images → Extract features → Batch predict → Save results

## Extending MLflow Tracing

To add custom traces to your code:

```python
import mlflow

# Start a run
with mlflow.start_run(run_name="custom_analysis") as run:
    # Log parameters
    mlflow.log_param("image_size", 224)
    
    # Log metrics
    mlflow.log_metric("processing_time", 0.45)
    
    # Log artifacts
    mlflow.log_artifact("path/to/visualization.png")
```

## Further Resources

- [Official MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [OpenTelemetry Integration](https://www.mlflow.org/docs/latest/python_api/mlflow.opentelemetry.html)
- Project's `configs/mlflow_tracing.py` for implementation details
