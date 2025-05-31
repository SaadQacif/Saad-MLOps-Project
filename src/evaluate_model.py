"""
Model evaluation module for the MLOps project.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import cross_val_score

# Add src to path to avoid import errors
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import calcul_dev from utils
from src.utils import calcul_dev


def load_model(model_path, encoder_path):
    """
    Load the trained model and label encoder.
    
    Args:
        model_path: Path to the trained model
        encoder_path: Path to the label encoder
        
    Returns:
        Tuple of (model, encoder)
    """
    with open(model_path, 'rb') as f:
        models_by_features = pickle.load(f)
    
    with open(encoder_path, 'rb') as f:
        encoder = pickle.load(f)
    
    return models_by_features['rgb'], encoder


def evaluate_model(model, encoder, features, labels):
    """
    Evaluate the model on test data.
    
    Args:
        model: Trained model
        encoder: Label encoder
        features: Test features
        labels: Test labels
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Encode labels
    encoded_labels = encoder.transform(labels)
    
    # Make predictions
    predictions = model.predict(features)
    
    # Calculate accuracy
    accuracy = accuracy_score(encoded_labels, predictions)
    
    # Generate classification report
    report = classification_report(encoded_labels, predictions, 
                                  target_names=encoder.classes_, output_dict=True)
    
    # Generate confusion matrix
    cm = confusion_matrix(encoded_labels, predictions)
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm,
        'predictions': predictions,
        'true_labels': encoded_labels
    }


def visualize_results(evaluation_results, encoder, output_dir="visualizations"):
    """
    Generate and save visualizations of model evaluation results.
    
    Args:
        evaluation_results: Dictionary of evaluation metrics
        encoder: Label encoder
        output_dir: Directory to save visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Confusion matrix
    plt.figure(figsize=(10, 8))
    cm = evaluation_results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    
    # Classification report visualization
    report = evaluation_results['classification_report']
    classes = list(report.keys())[:-3]  # Exclude 'accuracy', 'macro avg', 'weighted avg'
    
    # Extract metrics for each class
    precision = [report[c]['precision'] for c in classes]
    recall = [report[c]['recall'] for c in classes]
    f1_score = [report[c]['f1-score'] for c in classes]
    
    # Create DataFrame
    df = pd.DataFrame({
        'Class': classes,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1_score
    })
    
    # Melt the DataFrame for easier plotting
    df_melted = df.melt(id_vars=['Class'], var_name='Metric', value_name='Score')
    
    # Create plot
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_melted, x='Class', y='Score', hue='Metric')
    plt.title('Classification Metrics by Class')
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_metrics.png'))
    
    # Overall accuracy
    plt.figure(figsize=(6, 4))
    plt.bar(['Accuracy'], [evaluation_results['accuracy']], color='green')
    plt.title('Model Accuracy')
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_accuracy.png'))


def extract_features_from_directory(data_dir):
    """
    Extract features from preprocessed images in a directory.
    This is a simplified function that loads features from existing CSV files.
    
    Args:
        data_dir: Directory containing preprocessed images or path to CSV features file
        
    Returns:
        Dictionary with features and labels
    """
    # Check if data_dir is a CSV file
    if data_dir.endswith('.csv') and os.path.isfile(data_dir):
        features_file = data_dir
    else:
        # Find the CSV file with features
        features_file = None
        for root, _, files in os.walk(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'features')):
            for file in files:
                if file.endswith('.csv'):
                    features_file = os.path.join(root, file)
                    break
            if features_file:
                break
        
        if not features_file:
            raise FileNotFoundError("No feature CSV file found in the data/features directory")
    
    # Load features
    print(f"Loading features from {features_file}")
    df = pd.read_csv(features_file)
    
    # Separate features and labels
    X = df.drop(['class'], axis=1)
    if 'image_path' in X.columns:
        X = X.drop(['image_path'], axis=1)
    y = df['class']
    
    return {
        'features': X,
        'labels': y.values
    }


def log_evaluation_tables_to_mlflow(evaluation_results, encoder, run_id=None):
    """Log evaluation tables to MLflow"""
    try:
        # Import mlflow at function call time to avoid import errors if not installed
        import mlflow
        from mlflow.tracking import MlflowClient
        
        # Prepare data frames
        class_names = encoder.classes_
        y_true = evaluation_results['true_labels']
        y_pred = evaluation_results['predictions']
        
        # Create confusion matrix dataframe
        cm = evaluation_results['confusion_matrix']
        cm_df = pd.DataFrame(cm, columns=class_names, index=class_names)
        cm_df.index.name = 'Actual'
        cm_df.columns.name = 'Predicted'
        cm_df_reset = cm_df.reset_index()
        
        # Create classification report dataframe
        report = evaluation_results['classification_report']
        report_df = pd.DataFrame.from_dict(report).transpose()
        report_df = report_df.reset_index().rename(columns={'index': 'class'})
        
        # Create prediction dataframe
        pred_df = pd.DataFrame({
            'actual_class_idx': y_true,
            'predicted_class_idx': y_pred,
            'actual_class': [class_names[i] for i in y_true],
            'predicted_class': [class_names[i] for i in y_pred]
        })
        
        # Log tables
        if run_id:
            client = MlflowClient()
            client.log_table(run_id, cm_df_reset, "confusion_matrix.json")
            client.log_table(run_id, report_df, "classification_report.json")
            client.log_table(run_id, pred_df, "predictions.json")
            print("Evaluation tables logged to MLflow")
        else:
            with mlflow.start_run() as run:
                mlflow.log_table(cm_df_reset, "confusion_matrix.json")
                mlflow.log_table(report_df, "classification_report.json")
                mlflow.log_table(pred_df, "predictions.json")
                print("Evaluation tables logged to MLflow")
                
    except ImportError:
        print("MLflow not available. Skipping table logging.")
    except Exception as e:
        print(f"Error logging tables to MLflow: {e}")
        
    return


def main():
    """Main function to evaluate the trained model."""
    # Define paths
    test_data_dir = "data/testing"
    model_path = "models/bin/GradientBoosting_model.pkl"
    encoder_path = "models/bin/label_encoder.pkl"
    output_dir = "visualizations"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and encoder
    print("Loading model...")
    model, encoder = load_model(model_path, encoder_path)
    
    # Extract features from test data
    print("Extracting features from test images...")
    features_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                              'data', 'features', 'potato_features.csv')
    if os.path.exists(features_file):
        data = extract_features_from_directory(features_file)
    else:
        data = extract_features_from_directory(test_data_dir)
    
    # Evaluate model
    print("Evaluating model...")
    evaluation_results = evaluate_model(model, encoder, data['features'], data['labels'])
    
    # Print evaluation metrics
    print(f"\nModel Accuracy: {evaluation_results['accuracy']:.4f}")
    print("\nClassification Report:")
    report = evaluation_results['classification_report']
    for cls in encoder.classes_:
        print(f"{cls}:")
        print(f"  Precision: {report[cls]['precision']:.4f}")
        print(f"  Recall: {report[cls]['recall']:.4f}")
        print(f"  F1-score: {report[cls]['f1-score']:.4f}")
    
    # Generate visualizations
    print("Generating visualizations...")
    visualize_results(evaluation_results, encoder, output_dir)
    
    try:
        # Try to import mlflow to check if it's available
        import mlflow
        # Log evaluation tables to MLflow
        print("Logging evaluation tables to MLflow...")
        log_evaluation_tables_to_mlflow(evaluation_results, encoder)
    except ImportError:
        print("MLflow not available. Skipping MLflow logging.")
    except Exception as e:
        print(f"Error logging to MLflow: {str(e)}")
    
    print(f"Evaluation complete. Visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
