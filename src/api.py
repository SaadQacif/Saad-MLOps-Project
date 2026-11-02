"""Flask REST API for potato leaf disease classification.

Endpoints (Swagger UI at /docs):
  GET  /health/status      readiness: whether a model is loaded, which one, its held-out accuracy
  POST /predict/image      multipart upload `file` -> predicted class, confidence, probabilities
  GET  /monitoring/drift   drift of the features of served images against the training features
  GET  /health/metrics     request metrics (needs X-API-KEY = $API_ADMIN_KEY)

Predictions go through src.serving, the same feature and model path the training
script and the frontend use. Every prediction's features are appended to the
prediction log ($PREDICTION_LOG, default data/monitoring/predictions.csv), which is
what /monitoring/drift compares with data/features/reference_features.csv.

Environment: MODEL_DIR (default models/bin), MODEL_NAME (default: best held-out
accuracy), PREDICTION_LOG, API_ADMIN_KEY (admin endpoints are disabled when unset).
"""

import csv
import io
import os
import sys
import threading
import time
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from flask import Flask, request
from flask_restx import Api, Resource, fields, Namespace
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src import serving  # noqa: E402
from src.monitoring import APIMonitor  # noqa: E402
from scripts.drift_detection import DriftDetector  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("api")

MODEL_DIR = Path(os.environ.get("MODEL_DIR", ROOT / "models" / "bin"))
MODEL_NAME = os.environ.get("MODEL_NAME") or None
PREDICTION_LOG = Path(os.environ.get("PREDICTION_LOG", ROOT / "data" / "monitoring" / "predictions.csv"))
REFERENCE_CSV = ROOT / "data" / "features" / "reference_features.csv"
MAX_UPLOAD_MB = 10

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

api = Api(app, version='2.0', title='Potato Disease Classification API',
          description='Classifies potato leaf images (healthy, early blight, late blight)',
          doc='/docs',
          authorizations={'apikey': {'type': 'apiKey', 'in': 'header', 'name': 'X-API-KEY'}})
ns_predict = Namespace('predict', description='Prediction')
ns_health = Namespace('health', description='Health and metrics')
ns_monitoring = Namespace('monitoring', description='Data drift monitoring')
for ns in (ns_predict, ns_health, ns_monitoring):
    api.add_namespace(ns)

monitor = APIMonitor(log_dir=os.environ.get("API_LOG_DIR", str(ROOT / "logs")))
_log_lock = threading.Lock()

try:
    predictor = serving.load_predictor(MODEL_NAME, MODEL_DIR)
    model_error = None
    logger.info("Loaded model %s from %s", predictor.name, MODEL_DIR)
except Exception as e:  # the API still starts and reports not_ready
    predictor, model_error = None, f"{type(e).__name__}: {e}"
    logger.error("Model not loaded: %s", model_error)


def _admin_authorised():
    key = os.environ.get("API_ADMIN_KEY")
    return bool(key) and request.headers.get("X-API-KEY") == key


def log_prediction(result):
    """Append one served prediction (features, class, confidence) to the prediction log."""
    row = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "model": result["model"], "class": result["prediction"],
           "confidence": result["confidence"], **result["features"]}
    with _log_lock:
        PREDICTION_LOG.parent.mkdir(parents=True, exist_ok=True)
        new = not PREDICTION_LOG.exists()
        with open(PREDICTION_LOG, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)


upload_parser = api.parser()
upload_parser.add_argument('file', location='files', type='file', required=True, help='Leaf image')

prediction_model = api.model('Prediction', {
    'prediction': fields.String(description='Predicted class'),
    'confidence': fields.Float(description='Probability of the predicted class'),
    'probabilities': fields.Raw(description='Probability per class'),
    'model': fields.String(description='Model that produced the prediction'),
    'processing_time': fields.Float(description='Seconds'),
})
error_model = api.model('Error', {'error': fields.String})


@ns_health.route('/status')
class HealthCheck(Resource):
    def get(self):
        """Readiness and the loaded model"""
        body = {"status": "ok" if predictor else "not_ready",
                "model_loaded": predictor is not None,
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        if predictor:
            body.update(model=predictor.name,
                        held_out_accuracy=predictor.metadata["model"]["held_out_accuracy"],
                        trained_at=predictor.metadata.get("created_at"))
        else:
            body["error"] = model_error
        monitor.log_request("/health/status", 200, 0)
        return body, (200 if predictor else 503)


@ns_health.route('/metrics')
class MetricsResource(Resource):
    @api.doc(security='apikey')
    def get(self):
        """Request metrics (admin)"""
        if not _admin_authorised():
            return {"error": "Unauthorized (set API_ADMIN_KEY and send it as X-API-KEY)"}, 401
        return monitor.get_metrics_summary()


@ns_predict.route('/image')
class PredictImage(Resource):
    @api.doc(parser=upload_parser, responses={200: ('Success', prediction_model),
                                              400: ('Bad request', error_model),
                                              413: 'Upload too large',
                                              503: ('Model not loaded', error_model)})
    def post(self):
        """Classify one leaf image"""
        t0 = time.time()
        if predictor is None:
            return {"error": f"model not loaded: {model_error}"}, 503
        file = request.files.get('file')
        if file is None or file.filename == '':
            monitor.log_request("/predict/image", 400, time.time() - t0)
            return {"error": "send the image as multipart form field 'file'"}, 400
        try:
            image = Image.open(io.BytesIO(file.read()))
            image.load()
        except (UnidentifiedImageError, OSError):
            monitor.log_request("/predict/image", 400, time.time() - t0)
            return {"error": "the uploaded file is not a readable image"}, 400
        try:
            result = predictor.predict_image(image)
        except Exception as e:
            logger.exception("prediction failed")
            monitor.log_error("/predict/image", type(e).__name__, str(e))
            monitor.log_request("/predict/image", 500, time.time() - t0)
            return {"error": "prediction failed"}, 500
        log_prediction(result)
        result.pop("features")
        result["processing_time"] = time.time() - t0
        monitor.log_request("/predict/image", 200, result["processing_time"],
                            {"prediction": result["prediction"]})
        return result


@ns_monitoring.route('/drift')
class Drift(Resource):
    @api.doc(params={'window': 'Number of most recent predictions to compare (default 500)'})
    def get(self):
        """Drift of recent served features against the training features"""
        if not PREDICTION_LOG.exists():
            return {"status": "no_predictions_logged", "prediction_log": str(PREDICTION_LOG)}
        if not REFERENCE_CSV.exists():
            return {"error": f"reference features not found: {REFERENCE_CSV}"}, 503
        window = request.args.get("window", default=500, type=int)
        current = pd.read_csv(PREDICTION_LOG).tail(max(window, 1))
        reference = pd.read_csv(REFERENCE_CSV)
        cols = serving.FEATURE_NAMES + ["class"]
        report = DriftDetector().analyse(reference[cols], current[cols])
        top = sorted(report["feature_drift"].items(), key=lambda kv: kv[1]["ks_p_value"])[:5]
        return {"summary": report["summary"], "recommendations": report["recommendations"],
                "most_shifted_features": {k: {"ks_p_value": v["ks_p_value"], "psi": float(v["psi"]),
                                              "ref_mean": float(v["ref_mean"]),
                                              "cur_mean": float(v["cur_mean"])} for k, v in top}}


@app.errorhandler(413)
def too_large(_):
    return {"error": f"upload larger than {MAX_UPLOAD_MB} MB"}, 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
