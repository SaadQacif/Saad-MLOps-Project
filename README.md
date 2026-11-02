# Potato Leaf Disease Classification — MLOps Pipeline

Classifies potato leaf images as **healthy**, **early blight** or **late blight**, and
wraps the model in the parts of an MLOps workflow that are tested on every push: one
shared training/serving feature path, a REST API, a Streamlit app, data-drift
monitoring, locked dependencies, Docker images and CI/CD on GitHub Actions.

## Results

Held-out accuracy on 431 images (stratified 20% split, seed 42) that the served models
never saw during training. These are the files in `models/bin/`; the numbers are
written by the training script to `models/bin/model_metadata.json`, and CI re-trains
and checks that it reproduces them exactly.

| model | accuracy | macro F1 | healthy recall |
|---|---|---|---|
| SVM (served by default) | **0.886** | 0.830 | 0.55 |
| Random Forest | 0.884 | 0.815 | 0.48 |
| MLP | 0.870 | 0.814 | 0.55 |

The weak spot is the **healthy** class: it has 152 images against 1,000 for each
disease, and about half of the healthy test leaves are classified as diseased. See
[Limitations](#limitations).

## Architecture

```
images ──► src/serving.py: features_from_image ──► scaler ──► model ──► class
              ▲ used identically by                                  │
              ├─ scripts/train_and_save_all_models.py  (training)    │
              ├─ src/api.py                            (REST API) ───┼─► prediction log ──► /monitoring/drift
              └─ src/enhanced_complete_frontend_fixed.py (Streamlit)  │        (KS test vs training features)
```

* **One feature path.** Training, the API and the frontend all call
  `src.serving.features_from_image`, so a model is always served the same features it
  was trained on. A test recomputes features from the images and compares them with
  the training data (train/serve skew check).
* **Training** (`scripts/train_and_save_all_models.py`) extracts features from
  `data/processed/`, makes the split, fits scaler and models on the 80% only, scores
  them on the 20%, and records metrics, split, data hash and library versions in
  `model_metadata.json`.
* **REST API** (`src/api.py`, Flask + Swagger at `/docs`, gunicorn in Docker).
* **Frontend** (Streamlit) lists the trained models with their held-out accuracy.
* **Drift monitoring.** The API logs the features of every image it classifies; the
  drift endpoint compares the most recent window with the training features.
* **MLflow** tracking server runs in Docker Compose for the experiment-tracking
  pipeline in `scripts/complete_mlflow_pipeline.py`.

## Quick start

```bash
docker compose up --build
```

| service | URL |
|---|---|
| REST API (Swagger UI) | http://localhost:8000/docs |
| Streamlit app | http://localhost:8501 |
| MLflow UI | http://localhost:5000 |

Without Docker (Python 3.10):

```bash
pip install -r requirements.txt
export PYTHONPATH=$PWD:$PWD/scripts:$PWD/src
python scripts/train_and_save_all_models.py      # ~35 s on 8 CPU cores
gunicorn --threads 4 -b 0.0.0.0:8000 src.api:app
streamlit run src/enhanced_complete_frontend_fixed.py
```

## REST API

| method | path | |
|---|---|---|
| GET | `/health/status` | readiness, loaded model, its held-out accuracy (503 if no model) |
| POST | `/predict/image` | multipart field `file` (≤ 10 MB) → class, confidence, per-class probabilities |
| GET | `/monitoring/drift?window=500` | drift of recent served features against the training features |
| GET | `/health/metrics` | request metrics; needs header `X-API-KEY` equal to `$API_ADMIN_KEY` (always 401 when the variable is unset) |

```bash
curl -F "file=@data/split/test/Potato___Late_blight/<image>.jpg" http://localhost:8000/predict/image
```

Response shape (values illustrative):

```json
{"prediction": "Potato___Late_blight", "confidence": 0.97,
 "probabilities": {"Potato___Late_blight": 0.97, "Potato___Early_blight": 0.02, "Potato___healthy": 0.01},
 "model": "svm", "processing_time": 0.04}
```

Configuration: `MODEL_DIR` (default `models/bin`), `MODEL_NAME` (default: best
held-out accuracy), `PREDICTION_LOG` (default `data/monitoring/predictions.csv`),
`API_ADMIN_KEY`.

## Drift monitoring

`scripts/drift_detection.py` compares the current window with
`data/features/reference_features.csv` (the training split):

* a feature has drifted when its two-sample Kolmogorov–Smirnov test is significant
  after a Bonferroni correction over the features (p < 0.05 / n);
* the window is flagged as drift when more than 10% of the features drifted;
* fewer than 30 samples is reported as *insufficient data*, not as "no drift";
* PSI, Mann–Whitney U, mean shift and std ratio are reported per feature as context.

The tests check that held-out images from the training distribution show no drift,
that a shift in one feature is attributed to that feature only, and, end to end
through the API, that the same leaves photographed 60% brighter are flagged.
The same check runs from the command line on any feature table:
`python scripts/drift_detection.py --reference-data-path ... --current-data-path ... --output-path ...`
(this variant also logs the report to MLflow).

## Testing and CI/CD

`pytest` runs on every push to `main` and on pull requests
(`.github/workflows/ci.yml`, Python 3.10, locked dependencies):

| file | checks |
|---|---|
| `test_model_quality.py` | relative image paths; train/serve skew; committed models load with the pinned scikit-learn; every model the frontend lists predicts; re-training reproduces `model_metadata.json` and reaches ≥ 0.85 held-out accuracy; the Streamlit script runs without exceptions |
| `test_api.py` | health; predictions on held-out images (≥ 0.8 correct); rejects missing and non-image uploads; admin key; drift endpoint end to end |
| `test_drift.py` | no false alarm on held-out data; correct attribution of a single-feature shift; broad shift flagged; small windows reported as insufficient |
| `test_mlflow_dataset.py`, `test_preprocessing.py`, `test_training.py`, `test_predictions.py` | the MLflow experiment pipeline in `scripts/complete_mlflow_pipeline.py` |

After CI passes on `main`, CD (`.github/workflows/cd.yml`) builds the three images
with Docker Compose on the runner, waits for their health checks, classifies one
held-out image of each class through the running API, queries the drift endpoint,
and tears the stack down. The same job runs on any branch that changes a
Dockerfile, the compose file or the requirements. Nothing is deployed or exposed
outside the runner.

**Dependencies.** `requirements.in` lists the direct dependencies;
`requirements.txt` is the lock compiled from it for Linux / Python 3.10 and is what
CI and every image install. scikit-learn is pinned to 1.5.2, the version the models
are serialised with.

## Repository layout

```
src/serving.py                        shared feature + model path
src/api.py                            REST API
src/enhanced_complete_frontend_fixed.py   Streamlit app
scripts/train_and_save_all_models.py  feature extraction, training, held-out evaluation
scripts/extract_potato_features.py    feature definitions
scripts/drift_detection.py            drift statistics and report
scripts/complete_mlflow_pipeline.py   MLflow experiment pipeline (tested in CI)
models/bin/                           served models, scaler, encoder, model_metadata.json
data/processed/                       masked leaf images by class (training input)
data/features/                        extracted features; reference_features.csv = training split
tests/                                pytest suite
Dockerfile, Dockerfile.frontend, Dockerfile.mlflow, docker-compose.yml
.github/workflows/                    CI and CD
```

`kubernetes/`, `docs/`, `notebooks/` and several scripts in `scripts/` come from
earlier iterations of the project. They are kept for reference and are **not**
exercised by CI; the Kubernetes manifests reference images that are not published.

## Limitations

* **17 of the 30 features are placeholders.** `extract_all_features_from_array`
  returns a constant for the channel kurtosis and skew, green and diseased ratios,
  entropy mean and std, texture contrast, minimum convexity and the contour features,
  and three more duplicate other features (`red_std` = `r_dev`, …). The models
  effectively use about 10 features. Implementing the missing ones is the most likely
  way to improve accuracy, and in particular healthy recall.
* **Class imbalance.** 152 healthy images against 1,000 per disease; the models are
  trained without class weighting or resampling.
* **One split.** Metrics come from a single stratified 80/20 split; there is no
  cross-validation or confidence interval, so differences of ~0.01 between models are
  not meaningful.
* **Drift needs real traffic.** The monitoring is demonstrated with held-out and
  artificially brightened images. The prediction log lives inside the API container;
  in a real deployment it should go to persistent storage.
* **The image data is committed to git** (about 36 MB under `data/`), not versioned
  with DVC; there is no DVC remote.
