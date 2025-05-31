"""
Flask API for serving the ML model.
Uses Flask-RESTx for API documentation and includes monitoring.
"""

import os
import io
import json
import pickle
import time
import numpy as np
from PIL import Image
import cv2
from flask import Flask, request, jsonify, send_from_directory
from flask_restx import Api, Resource, fields, Namespace
from werkzeug.utils import secure_filename
from werkzeug.exceptions import BadRequest, InternalServerError
import logging
from datetime import datetime

from segmentation import contour_detection, segment_image
from image_processing import contour, apply_clahe
from utils import calcul_dev
from monitoring import APIMonitor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Configure API with documentation
authorizations = {
    'apikey': {
        'type': 'apiKey',
        'in': 'header',
        'name': 'X-API-KEY'
    }
}

api = Api(
    app,
    version='1.0',
    title='MLOps Image Classification API',
    description='API for image segmentation and classification',
    doc='/docs',
    authorizations=authorizations
)

# Define namespaces
ns_predict = Namespace('predict', description='Prediction operations')
ns_health = Namespace('health', description='Health check operations')
ns_admin = Namespace('admin', description='Admin operations')

api.add_namespace(ns_predict)
api.add_namespace(ns_health)
api.add_namespace(ns_admin)

# Load configuration
config_path = os.path.join(os.getcwd(), "configs", "config.json")
try:
    with open(config_path, "r") as f:
        config = json.load(f)
except FileNotFoundError:
    logger.warning(f"Config file not found at {config_path}")
    config = {
        "paths": {
            "data": {
                "input": "data/inputs",
                "output": "data/outputs",
                "upload": "data/uploads"
            },
            "models": {
                "directory": "models/bin",
                "model_file": "models_by_features.pkl",
                "encoder_file": "encoder.pkl"
            }
        },
        "api": {
            "host": "0.0.0.0",
            "port": 5000,
            "debug": False
        }
    }

# Load paths from config
MODEL_DIR = os.path.join(os.getcwd(), config["paths"]["models"]["directory"])
MODEL_PATH = os.path.join(MODEL_DIR, config["paths"]["models"]["model_file"])
UPLOAD_FOLDER = os.path.join(os.getcwd(), config["paths"]["data"]["upload"])
TMP_FOLDER = os.path.join(os.getcwd(), "tmp")

# Create required directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TMP_FOLDER, exist_ok=True)
os.makedirs(os.path.join(os.getcwd(), "logs"), exist_ok=True)

# Initialize API monitor
monitor = APIMonitor()

# Load models
try:
    with open(MODEL_PATH, 'rb') as f:
        models_by_features = pickle.load(f)
    model_loaded = True
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    logger.error(f"Model files not found at {MODEL_PATH}")
    model_loaded = False
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model_loaded = False


def process_image(image_path):
    """
    Process an image and extract features.
    
    Args:
        image_path: Path to the image
        
    Returns:
        Dictionary with features and processed image path
    """
    start_time = time.time()
    logger.info(f"Processing image: {image_path}")
    
    try:
        # Create output directory for temporary files
        tmp_dir = os.path.join(TMP_FOLDER, datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(tmp_dir, exist_ok=True)
        
        # Open the image
        with Image.open(image_path) as im:
            im_arr = np.asarray(im)
        
        # Filter the image to detect contours
        contours = contour(im_arr)
        filtered_path = os.path.join(tmp_dir, "filtered.png")
        cv2.imwrite(filtered_path, contours)
        
        # Segment the image
        cropped_path = os.path.join(tmp_dir, "cropped.png")
        segment_image(filtered_path, cropped_path, image_path)
        
        # Open the cropped image
        with Image.open(cropped_path) as im:
            im_arr = np.asarray(im)
        
        # Apply CLAHE for contrast enhancement
        enhanced_arr = apply_clahe(im_arr)
        enhanced_path = os.path.join(tmp_dir, "enhanced.png")
        cv2.imwrite(enhanced_path, enhanced_arr)
        
        # Calculate features
        features = calcul_dev(enhanced_arr)
        
        processing_time = time.time() - start_time
        logger.info(f"Image processed successfully in {processing_time:.2f}s")
        
        return {
            "features": features,
            "filtered_path": filtered_path,
            "cropped_path": cropped_path,
            "enhanced_path": enhanced_path,
            "processing_time": processing_time
        }
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        monitor.log_error("process_image", type(e).__name__, str(e))
        raise


def predict_class(features):
    """
    Predict class based on features.
    
    Args:
        features: Image features
        
    Returns:
        Prediction and confidence
    """
    if not model_loaded:
        logger.error("Model not loaded")
        raise RuntimeError("Model not loaded")
    
    try:
        # Get model and encoder
        model = models_by_features["model"]
        encoder = models_by_features["encoder"]
        
        # Reshape features for prediction
        features_array = np.array(features).reshape(1, -1)
        
        # Get prediction probability
        probs = model.predict_proba(features_array)[0]
        
        # Get the predicted class
        class_idx = np.argmax(probs)
        confidence = float(probs[class_idx])
        
        # Get class name
        class_name = encoder.inverse_transform([class_idx])[0]
        
        # Get top 3 predictions with probabilities
        top_indices = np.argsort(probs)[::-1][:3]
        top_predictions = [
            {
                "class": encoder.inverse_transform([idx])[0],
                "probability": float(probs[idx])
            }
            for idx in top_indices
        ]
        
        return {
            "prediction": class_name,
            "confidence": confidence,
            "top_predictions": top_predictions
        }
    
    except Exception as e:
        logger.error(f"Error predicting class: {str(e)}")
        monitor.log_error("predict_class", type(e).__name__, str(e))
        raise


# Define request and response models for API documentation
upload_parser = api.parser()
upload_parser.add_argument('file', location='files', type='file', required=True, help='Image file')

prediction_model = api.model('Prediction', {
    'prediction': fields.String(description='Predicted class'),
    'confidence': fields.Float(description='Confidence score'),
    'top_predictions': fields.List(fields.Nested(api.model('TopPrediction', {
        'class': fields.String(description='Class name'),
        'probability': fields.Float(description='Probability')
    })))
})

error_model = api.model('Error', {
    'error': fields.String(description='Error message')
})

@ns_health.route('/status')
class HealthCheck(Resource):
    @api.doc('health_check')
    def get(self):
        """Check API health status"""
        start_time = time.time()
        
        response = {
            "status": "ok" if model_loaded else "not_ready",
            "model_loaded": model_loaded,
            "timestamp": datetime.now().isoformat()
        }
        
        latency = time.time() - start_time
        monitor.log_request("/health/status", 200, latency)
        
        return response


@ns_health.route('/metrics')
class MetricsResource(Resource):
    @api.doc('get_metrics', security='apikey')
    def get(self):
        """Get API metrics"""
        # Check API key for admin endpoints
        api_key = request.headers.get('X-API-KEY')
        if api_key != os.environ.get('API_ADMIN_KEY', 'admin-secret-key'):
            monitor.log_request("/health/metrics", 401, 0)
            return {"error": "Unauthorized"}, 401
        
        start_time = time.time()
        metrics = monitor.get_metrics_summary()
        
        latency = time.time() - start_time
        monitor.log_request("/health/metrics", 200, latency)
        
        return metrics


@ns_predict.route('/image')
class PredictImage(Resource):
    @api.doc('predict_image', parser=upload_parser, responses={
        200: ('Success', prediction_model),
        400: ('Bad Request', error_model),
        500: ('Server Error', error_model)
    })
    def post(self):
        """Predict class from image"""
        start_time = time.time()
        
        try:
            # Check if file was included in request
            if 'file' not in request.files:
                monitor.log_request("/predict/image", 400, time.time() - start_time)
                return {"error": "No file part"}, 400
            
            file = request.files['file']
            
            # Check if file was selected
            if file.filename == '':
                monitor.log_request("/predict/image", 400, time.time() - start_time)
                return {"error": "No file selected"}, 400
            
            # Save uploaded file
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(UPLOAD_FOLDER, f"{timestamp}_{filename}")
            file.save(save_path)
            
            # Process image
            processed = process_image(save_path)
            
            # Predict class
            prediction = predict_class(processed["features"])
            
            # Add processing time
            prediction["processing_time"] = processed["processing_time"]
            
            # Add file paths
            prediction["image_paths"] = {
                "original": save_path,
                "filtered": processed["filtered_path"],
                "cropped": processed["cropped_path"],
                "enhanced": processed["enhanced_path"]
            }
            
            latency = time.time() - start_time
            monitor.log_request("/predict/image", 200, latency, {"prediction": prediction["prediction"]})
            
            return prediction
        
        except BadRequest as e:
            logger.error(f"Bad request: {str(e)}")
            monitor.log_request("/predict/image", 400, time.time() - start_time)
            return {"error": str(e)}, 400
        
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            monitor.log_request("/predict/image", 500, time.time() - start_time)
            return {"error": "Internal server error"}, 500


@ns_predict.route('/batch')
class BatchPredict(Resource):
    @api.doc('batch_predict', security='apikey')
    def post(self):
        """Batch predict from a directory"""
        start_time = time.time()
        
        try:
            # Validate request
            if not request.is_json:
                monitor.log_request("/predict/batch", 400, time.time() - start_time)
                return {"error": "Request must be JSON"}, 400
            
            data = request.get_json()
            
            if "directory" not in data:
                monitor.log_request("/predict/batch", 400, time.time() - start_time)
                return {"error": "Directory not specified"}, 400
            
            input_dir = data["directory"]
            
            if not os.path.isdir(input_dir):
                monitor.log_request("/predict/batch", 400, time.time() - start_time)
                return {"error": f"Directory not found: {input_dir}"}, 400
            
            # Process images
            results = []
            image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for image_file in image_files:
                image_path = os.path.join(input_dir, image_file)
                
                try:
                    # Process image
                    processed = process_image(image_path)
                    
                    # Predict class
                    prediction = predict_class(processed["features"])
                    
                    # Add to results
                    results.append({
                        "file": image_file,
                        "prediction": prediction["prediction"],
                        "confidence": prediction["confidence"]
                    })
                    
                except Exception as e:
                    results.append({
                        "file": image_file,
                        "error": str(e)
                    })
            
            response = {
                "results": results,
                "processed": len(results),
                "successful": sum(1 for r in results if "prediction" in r),
                "failed": sum(1 for r in results if "error" in r),
                "processing_time": time.time() - start_time
            }
            
            latency = time.time() - start_time
            monitor.log_request("/predict/batch", 200, latency)
            
            return response
        
        except Exception as e:
            logger.error(f"Error processing batch request: {str(e)}")
            monitor.log_request("/predict/batch", 500, time.time() - start_time)
            return {"error": "Internal server error"}, 500


@ns_admin.route('/visualizations/<path:path>')
class Visualizations(Resource):
    @api.doc('get_visualization', security='apikey')
    def get(self, path):
        """Get visualization files"""
        # Check API key
        api_key = request.headers.get('X-API-KEY')
        if api_key != os.environ.get('API_ADMIN_KEY', 'admin-secret-key'):
            monitor.log_request(f"/admin/visualizations/{path}", 401, 0)
            return {"error": "Unauthorized"}, 401
            
        start_time = time.time()
        
        try:
            vis_dir = os.path.join(os.getcwd(), "visualizations")
            
            if not os.path.exists(os.path.join(vis_dir, path)):
                monitor.log_request(f"/admin/visualizations/{path}", 404, time.time() - start_time)
                return {"error": "File not found"}, 404
            
            latency = time.time() - start_time
            monitor.log_request(f"/admin/visualizations/{path}", 200, latency)
            
            return send_from_directory(vis_dir, path)
        
        except Exception as e:
            logger.error(f"Error serving visualization: {str(e)}")
            monitor.log_request(f"/admin/visualizations/{path}", 500, time.time() - start_time)
            return {"error": "Internal server error"}, 500


if __name__ == "__main__":
    host = config["api"]["host"]
    port = config["api"]["port"]
    debug = config["api"]["debug"]
    
    logger.info(f"Starting API server on {host}:{port}")
    app.run(host=host, port=port, debug=debug)
