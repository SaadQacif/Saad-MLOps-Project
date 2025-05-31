#!/usr/bin/env python
"""
Enhanced Frontend for Potato Disease Classification
Features:
- Model selection from available trained models
- Local file-based experiment tracking
- Model comparison visualization
- Prediction history & analytics
"""

import os
import sys
import json
import pickle
import logging
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from PIL import Image

# Logging config (do early)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Handle optional imports
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    class StandardScaler:
        def transform(self, X): return X
        def fit(self, X): return X
        def fit_transform(self, X): return X
    class LabelEncoder:
        def __init__(self):
            self.classes_ = np.array(['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy'])
        def inverse_transform(self, y): 
            return [self.classes_[i] for i in y]

# Use local file-based tracking only - no external dependencies
PREDICTION_HISTORY_FILE = "/app/tmp/prediction_history.json"
MODEL_COMPARISON_FILE = "/app/models/model_comparison.csv"

def load_prediction_history():
    """Load prediction history from local file"""
    try:
        if os.path.exists(PREDICTION_HISTORY_FILE):
            with open(PREDICTION_HISTORY_FILE, 'r') as f:
                return json.load(f)
        return []
    except Exception as e:
        logger.warning(f"Failed to load prediction history: {e}")
        return []

# Add project root to sys.path for custom imports
project_root = Path(__file__).parent.parent if "__file__" in locals() else Path.cwd()
sys.path.extend([str(project_root), str(project_root / "scripts"), str(project_root / "src")])

try:
    from src.utils import calcul_dev
    UTILS_AVAILABLE = True
except Exception:
    UTILS_AVAILABLE = False

try:
    from scripts.extract_potato_features import extract_features, extract_all_features_from_array
    from scripts.extract_potato_features import extract_all_features_from_array as extract_potato_features_script
    FEATURE_EXTRACTION_AVAILABLE = True
except Exception:
    FEATURE_EXTRACTION_AVAILABLE = False
    def extract_features(image): return np.random.rand(100)
    def extract_all_features_from_array(image_array): return np.random.rand(100)
    def extract_potato_features_script(image_array): return {f'feature_{i}': np.random.rand() for i in range(30)}

try:
    from src.mlflow_integration import MLflowManager
except ImportError:
    MLflowManager = None

st.set_page_config(
    page_title="🥔 Potato Disease Classifier - Enhanced", 
    page_icon="🥔",
    layout="wide",
    initial_sidebar_state="expanded"
)

class EnhancedModelManager:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.bin_dir = self.models_dir / "bin"
        self.preprocessing_dir = self.models_dir / "preprocessing"
        # Use local file storage only - no MLflow
        self.use_local_tracking = True

    def discover_available_models(self) -> Dict[str, Dict[str, Any]]:
        models = {}
        
        # Bin models (feature-based, 30 features)
        if self.bin_dir.exists():
            model_files = list(self.bin_dir.glob("*_model.pkl"))
            for model_file in model_files:
                model_name = model_file.stem.replace("_model", "")
                models[f"{model_name} (Feature-based)"] = {
                    "model_path": str(model_file),
                    "model_name": model_name,
                    "model_type": "feature-based",
                    "source": "bin",
                    "timestamp": datetime.fromtimestamp(model_file.stat().st_mtime),
                    "encoder_path": self._find_file(self.bin_dir, ["label_encoder.pkl", "encoder.pkl"]),
                    "scaler_path": self._find_file(self.bin_dir, ["scaler.pkl", "standard_scaler.pkl"]),
                    "metadata_path": self._find_file(self.bin_dir, ["model_metadata.json", "pipeline_summary.json"])
                }
        
        # Root models (feature-based, 30 features)
        model_files = list(self.models_dir.glob("*_model.pkl"))
        for model_file in model_files:
            model_name = model_file.stem.replace("_model", "")
            key = f"{model_name} (Feature-based)"
            if key not in models:
                models[key] = {
                    "model_path": str(model_file),
                    "model_name": model_name,
                    "model_type": "feature-based",
                    "source": "root",
                    "timestamp": datetime.fromtimestamp(model_file.stat().st_mtime),
                    "encoder_path": self._find_file(self.models_dir, ["label_encoder.pkl", "encoder.pkl"]),
                    "scaler_path": self._find_file(self.models_dir, ["scaler.pkl", "standard_scaler.pkl"]),
                    "metadata_path": self._find_file(self.models_dir, ["model_metadata.json", "pipeline_summary.json"])
                }
        
        return models

    def _find_file(self, directory: Path, filenames: list) -> Optional[str]:
        for filename in filenames:
            file_path = directory / filename
            if file_path.exists():
                return str(file_path)
        return None

    def load_model_artifacts(self, model_info: Dict[str, Any]) -> Tuple[Any, Any, Any, Dict]:
        try:
            with open(model_info["model_path"], 'rb') as f:
                model = pickle.load(f)
            
            encoder_path = model_info.get("encoder_path")
            if encoder_path and os.path.exists(encoder_path):
                with open(encoder_path, 'rb') as f:
                    encoder = pickle.load(f)
            else:
                encoder = LabelEncoder()
            
            scaler_path = model_info.get("scaler_path")
            if scaler_path and os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
            else:
                scaler = StandardScaler()
            
            metadata = {}
            metadata_path = model_info.get("metadata_path")
            if metadata_path and os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                except Exception as e:
                    logger.warning(f"Could not load metadata: {e}")
            
            return model, encoder, scaler, metadata
        except Exception as e:
            logger.error(f"Failed to load model artifacts: {e}")
            raise

    def get_local_tracking_experiments(self) -> List[Dict]:
        """Get recent local tracking experiments"""
        if not self.use_local_tracking:
            return []
        
        try:
            # For local tracking, we can return prediction history as experiments
            history = load_prediction_history()
            experiment_data = []
            
            for i, pred in enumerate(history[-10:]):  # Last 10 predictions as experiments
                experiment_data.append({
                    "run_id": f"local_run_{i}",
                    "run_name": f"Prediction_{pred.get('image_name', 'Unknown')}",
                    "status": "FINISHED",
                    "start_time": pred.get('timestamp', 'Unknown'),
                    "metrics": {"confidence": pred.get('confidence', 0.0)},
                    "params": {"model": pred.get('model', 'Unknown')}
                })
            
            return experiment_data
        except Exception as e:
            logger.error(f"Failed to get local tracking experiments: {e}")
            return []

# ----------- Utility Functions -----------

def init_session_state():
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = None

def extract_image_features(image: Image.Image, model_info: Dict[str, Any] = None) -> Optional[pd.DataFrame]:
    """Extract features from image using the standard 30-feature extraction for all models"""
    if not FEATURE_EXTRACTION_AVAILABLE:
        st.error("Feature extraction not available. Please check the setup.")
        return None
    
    try:
        image_array = np.array(image)
        
        # Handle different image formats
        if len(image_array.shape) == 3 and image_array.shape[2] == 4:
            image_array = image_array[:, :, :3]  # Remove alpha channel
        elif len(image_array.shape) == 2:
            if CV2_AVAILABLE:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
            else:
                image_array = np.stack([image_array] * 3, axis=-1)
        
        if image_array.dtype != np.uint8:
            image_array = (image_array * 255).astype(np.uint8)
        
        if model_info is None:
            model_info = st.session_state.get('selected_model_info', {})
        
        model_source = model_info.get('source', 'root')
        model_type = model_info.get('model_type', 'unknown')
        
        logger.debug(f"Extracting features for model source: {model_source}, type: {model_type}")

        # Use the standard 30-feature extraction for ALL models (fixed scaling issue)
        logger.info(f"Using standard 30-feature extraction script for feature-based model (source: {model_source}, type: {model_type}).")
          # Use the standard 30-feature extraction for all models
        features_dict = extract_potato_features_script(image_array)  # This returns a dict
        features_df = pd.DataFrame([features_dict])
        
        # Ensure columns are in the same order as during training
        # This should match the feature_order in extract_potato_features.py
        feature_order = [
            'r_dev', 'g_dev', 'b_dev', 'red_mean', 'red_std', 'red_kurtosis', 'red_skew',
            'green_mean', 'green_std', 'green_kurtosis', 'green_skew', 'blue_mean', 'blue_std',
            'blue_kurtosis', 'blue_skew', 'green_ratio', 'diseased_ratio', 'entropy_mean',
            'entropy_std', 'entropy_variation_rhythm', 'texture_contrast', 'avg_convexity',
            'min_convexity', 'contour_area', 'contour_perimeter', 'circularity', 'eccentricity',
            'solidity', 'aspect_ratio', 'area_bbox_ratio'
        ]
        features_df = features_df[feature_order]

        logger.info(f"Extracted features shape: {features_df.shape}")
        return features_df
        
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        logger.error(f"Feature extraction error: {e}")
        return None

def predict_with_model(image: Image.Image, model_info: Dict[str, Any]) -> Dict[str, Any]:
    try:
        model, encoder, scaler, metadata = model_manager.load_model_artifacts(model_info)
        features_df = extract_image_features(image, model_info)
        
        if features_df is None:
            return {"error": "Failed to extract features from image"}
        
        # Check scaler/feature shape match
        if hasattr(scaler, 'mean_') and scaler.mean_.shape[0] != features_df.shape[1]:
            return {"error": f"Scaler expects {scaler.mean_.shape[0]} features, but got {features_df.shape[1]}. Please retrain and save the scaler with the correct number of features."}
        
        features_scaled = scaler.transform(features_df.values)
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0] if hasattr(model, 'predict_proba') else None
        
        predicted_class = encoder.inverse_transform([prediction])[0] if hasattr(encoder, 'inverse_transform') else encoder.classes_[prediction]
        confidence = float(np.max(probabilities)) if probabilities is not None else 0.0
        
        class_probs = {}
        if probabilities is not None:
            for i, prob in enumerate(probabilities):
                class_name = encoder.classes_[i]
                class_probs[class_name] = float(prob)
        
        return {
            "prediction": predicted_class,
            "confidence": confidence,
            "probabilities": class_probs,
            "model_info": model_info
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"error": f"Prediction failed: {str(e)}"}

def display_model_comparison():
    st.subheader("📊 Model Performance Comparison")
    comparison_file = Path("models/model_comparison.csv")
    
    if comparison_file.exists():
        try:
            df = pd.read_csv(comparison_file)
            st.dataframe(df, use_container_width=True)
            
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            available_metrics = [m for m in metrics if m in df.columns]
            
            if available_metrics and PLOTLY_AVAILABLE:
                fig = make_subplots(rows=2, cols=2,
                                   subplot_titles=available_metrics,
                                   specs=[[{"type": "bar"}, {"type": "bar"}], [{"type": "bar"}, {"type": "bar"}]])
                
                for i, metric in enumerate(available_metrics):
                    row, col = (i // 2) + 1, (i % 2) + 1
                    fig.add_trace(go.Bar(
                        x=df['Model'], y=df[metric], name=metric, text=df[metric].round(3),
                        textposition='auto', showlegend=False),
                        row=row, col=col)
                
                fig.update_layout(height=600, title_text="Model Performance Comparison", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                best_accuracy_idx = df['Accuracy'].idxmax()
                best_model = df.loc[best_accuracy_idx, 'Model']
                best_accuracy = df.loc[best_accuracy_idx, 'Accuracy']
                st.success(f"🏆 Best performing model: **{best_model}** with {best_accuracy:.3f} accuracy")
        except Exception as e:
            st.error(f"Error loading model comparison: {e}")
    else:
        st.info("No model comparison data available. Train models first using the ML pipeline.")

def display_prediction_analytics():
    st.subheader("📈 Prediction Analytics")
    
    if st.session_state.prediction_history:
        history_df = pd.DataFrame(st.session_state.prediction_history)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Predictions", len(history_df))
        with col2:
            avg_confidence = history_df['confidence'].mean()
            st.metric("Average Confidence", f"{avg_confidence:.2%}")
        with col3:
            most_common = history_df['prediction'].mode()[0] if not history_df.empty else "N/A"
            st.metric("Most Common Prediction", most_common)
        
        # Prediction distribution
        if PLOTLY_AVAILABLE:
            st.subheader("Prediction Distribution")
            pred_counts = history_df['prediction'].value_counts()
            fig = px.pie(values=pred_counts.values, names=pred_counts.index, title="Distribution of Predictions")
            st.plotly_chart(fig, use_container_width=True)
            
            # Confidence over time
            st.subheader("Confidence Over Time")
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
            fig = px.line(history_df.sort_values('timestamp'), x='timestamp', y='confidence', 
                         title="Prediction Confidence Over Time")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No prediction history available. Make some predictions to see analytics.")

# ----------- Main Application -----------

def main():
    st.title("🥔 Enhanced Potato Disease Classifier")
    st.markdown("Upload a potato leaf image to detect diseases using trained ML models.")
    
    init_session_state()
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Model selection
    st.sidebar.subheader("🤖 Model Selection")
    models = model_manager.discover_available_models()
    
    if not models:
        st.error("No trained models found. Please train models first using the ML pipeline.")
        return
    
    model_names = list(models.keys())
    selected_model_name = st.sidebar.selectbox("Choose a model:", model_names)
    
    if selected_model_name:
        st.session_state.selected_model = models[selected_model_name]
        st.session_state.selected_model_info = models[selected_model_name]
        
        # Display model info
        with st.sidebar.expander("📋 Model Details"):
            model_info = models[selected_model_name]
            st.text(f"Name: {model_info['model_name']}")
            st.text(f"Type: {model_info['model_type']}")
            st.text(f"Source: {model_info['source']}")
            st.text(f"Last Modified: {model_info['timestamp'].strftime('%Y-%m-%d %H:%M')}")
    
    # Display options
    st.sidebar.subheader("🎨 Display Options")
    show_probabilities = st.sidebar.checkbox("Show class probabilities", value=True)
    save_predictions = st.sidebar.checkbox("Save predictions to history", value=True)
    show_analytics = st.sidebar.checkbox("Enable prediction analytics", value=True)
    
    # Main interface
    tab1, tab2, tab3, tab4 = st.tabs(["🔮 Prediction", "📊 Model Comparison", "💾 Local Storage", "📈 Analytics"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("📤 Upload Image")
            uploaded_file = st.file_uploader(
                "Choose a potato leaf image...", 
                type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
                help="Upload a clear image of a potato leaf for disease detection"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)
                
                if st.session_state.selected_model and st.button("🔮 Predict Disease", type="primary"):
                    with st.spinner("Analyzing image..."):
                        result = predict_with_model(image, st.session_state.selected_model)
                    
                    if "error" in result:
                        st.error(f"❌ {result['error']}")
                    else:
                        st.success(f"**Prediction:** {result['prediction']}")
                        st.metric("Confidence", f"{result['confidence']:.2%}")
                        
                        if show_probabilities and result.get('probabilities') and PLOTLY_AVAILABLE:
                            st.markdown("### 📊 Class Probabilities")
                            prob_df = pd.DataFrame(list(result['probabilities'].items()), columns=['Class', 'Probability'])
                            fig = px.bar(prob_df, x='Class', y='Probability', color='Probability', color_continuous_scale='viridis')
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        if save_predictions:
                            prediction_entry = {
                                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                "model": st.session_state.selected_model["model_name"],
                                "prediction": result["prediction"],
                                "confidence": result["confidence"],
                                "image_name": uploaded_file.name
                            }
                            st.session_state.prediction_history.append(prediction_entry)
            elif uploaded_file is not None:
                st.warning("Please select a model first!")
        
        with col2:
            st.header("📜 Recent Predictions")
            if st.session_state.prediction_history:
                recent_predictions = st.session_state.prediction_history[-5:]
                for i, pred in enumerate(reversed(recent_predictions)):
                    with st.expander(f"{pred['image_name']} - {pred['prediction'][:20]}..."):
                        st.write(f"**Model:** {pred['model']}")
                        st.write(f"**Prediction:** {pred['prediction']}")
                        st.write(f"**Confidence:** {pred['confidence']:.2%}")
                        st.write(f"**Time:** {pred['timestamp']}")
                
                if st.button("🗑️ Clear History"):
                    st.session_state.prediction_history = []
                    st.rerun()
            else:
                st.info("No predictions yet. Upload an image to get started!")
    
    with tab2:
        display_model_comparison()
    
    with tab3:
        st.subheader("📁 Local Storage Information")
        
        # Prediction History Storage
        st.markdown("### 📊 Prediction History")
        history = load_prediction_history()
        if history:
            st.success(f"✅ {len(history)} predictions stored locally")
            st.text(f"Storage file: {PREDICTION_HISTORY_FILE}")
            
            # Show recent predictions in a table
            if len(history) > 0:
                recent_history = pd.DataFrame(history[-10:])  # Last 10 predictions
                st.dataframe(recent_history, use_container_width=True)
                
                if st.button("📥 Download Prediction History"):
                    csv = pd.DataFrame(history).to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        else:
            st.info("No prediction history found. Make some predictions to populate this section.")
        
        # Model Files Information
        st.markdown("### 🤖 Available Model Files")
        models = model_manager.discover_available_models()
        if models:
            for model_name, model_info in models.items():
                with st.expander(f"📄 {model_name}"):
                    st.json({
                        "Model Path": model_info["model_path"],
                        "Model Type": model_info["model_type"],
                        "Source Directory": model_info["source"],
                        "Last Modified": str(model_info["timestamp"]),
                        "Encoder Available": bool(model_info["encoder_path"]),
                        "Scaler Available": bool(model_info["scaler_path"]),
                        "Metadata Available": bool(model_info["metadata_path"])
                    })
        
        # Storage Statistics
        st.markdown("### 📈 Storage Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            model_count = len(models)
            st.metric("Available Models", model_count)
        
        with col2:
            prediction_count = len(history) if history else 0
            st.metric("Total Predictions", prediction_count)
            
        with col3:
            # Calculate storage usage (approximate)
            storage_mb = 0
            try:
                if os.path.exists(PREDICTION_HISTORY_FILE):
                    storage_mb += os.path.getsize(PREDICTION_HISTORY_FILE) / (1024 * 1024)
                for model_info in models.values():
                    if os.path.exists(model_info["model_path"]):
                        storage_mb += os.path.getsize(model_info["model_path"]) / (1024 * 1024)
                st.metric("Storage Used", f"{storage_mb:.1f} MB")
            except Exception:
                st.metric("Storage Used", "N/A")
    
    with tab4:
        if show_analytics:
            display_prediction_analytics()
        else:
            st.info("Enable analytics in the sidebar to view prediction statistics.")

# Global model manager instance
model_manager = EnhancedModelManager()

if __name__ == "__main__":
    main()
