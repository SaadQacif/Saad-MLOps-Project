#!/usr/bin/env python
"""Extract features, train the served models on a held-out split, and record metrics.

1. Features are computed for every image in data/processed with
   src.serving.features_from_image, the exact function the API and the frontend
   use, and written to data/features/potato_features.csv (relative image paths).
2. A stratified 80/20 split (seed 42) is made. The scaler and the SVM, MLP and
   Random Forest are fit on the 80% only; the training split is also written to
   data/features/reference_features.csv as the reference for drift detection.
3. Each model is scored on the 20% it never saw. The models that are saved are
   the evaluated ones, so the accuracy in models/bin/model_metadata.json is the
   held-out accuracy of the files being served.

Usage: python scripts/train_and_save_all_models.py [--skip-extraction]
"""

import argparse
import hashlib
import json
import pickle
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from src.serving import FEATURE_NAMES, METADATA_FILE, features_from_image  # noqa: E402

IMAGES = ROOT / "data" / "processed"
FEATURES_CSV = ROOT / "data" / "features" / "potato_features.csv"
REFERENCE_CSV = ROOT / "data" / "features" / "reference_features.csv"
MODEL_DIR = ROOT / "models" / "bin"
SEED = 42
TEST_SIZE = 0.2

MODELS = {
    "svm": SVC(probability=True, random_state=SEED),
    "mlp": MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, early_stopping=True,
                         random_state=SEED),
    "randomforest": RandomForestClassifier(n_estimators=100, random_state=SEED),
}


def extract_features() -> pd.DataFrame:
    rows = []
    for class_dir in sorted(p for p in IMAGES.iterdir() if p.is_dir()):
        files = sorted(f for f in class_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
        print(f"{class_dir.name}: {len(files)} images")
        for f in files:
            row = features_from_image(f)
            row["image_path"] = f.relative_to(ROOT).as_posix()
            row["class"] = class_dir.name
            rows.append(row)
    df = pd.DataFrame(rows, columns=FEATURE_NAMES + ["image_path", "class"])
    FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FEATURES_CSV, index=False)
    return df


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--skip-extraction", action="store_true",
                    help="reuse data/features/potato_features.csv")
    args = ap.parse_args()

    t0 = time.time()
    df = pd.read_csv(FEATURES_CSV) if args.skip_extraction else extract_features()
    X, labels = df[FEATURE_NAMES].to_numpy(dtype=float), df["class"].to_numpy()
    encoder = LabelEncoder().fit(labels)
    y = encoder.transform(labels)
    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=TEST_SIZE, random_state=SEED,
                                      stratify=y)
    df.iloc[idx_tr].to_csv(REFERENCE_CSV, index=False)
    scaler = StandardScaler().fit(X[idx_tr])
    Xtr, Xte = scaler.transform(X[idx_tr]), scaler.transform(X[idx_te])

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    meta_models = {}
    for name, model in MODELS.items():
        with warnings.catch_warnings():
            warnings.simplefilter("error", category=ConvergenceWarning)
            model.fit(Xtr, y[idx_tr])
        pred = model.predict(Xte)
        fname = f"{name}_model.pkl"
        with open(MODEL_DIR / fname, "wb") as f:
            pickle.dump(model, f)
        meta_models[name] = {
            "file": fname,
            "estimator": type(model).__name__,
            "params": {k: v for k, v in model.get_params().items()
                       if isinstance(v, (int, float, str, bool, type(None), tuple))},
            "held_out_accuracy": float(accuracy_score(y[idx_te], pred)),
            "held_out_macro_f1": float(f1_score(y[idx_te], pred, average="macro")),
            "confusion_matrix": confusion_matrix(y[idx_te], pred).tolist(),
            "per_class": classification_report(y[idx_te], pred, target_names=encoder.classes_,
                                               output_dict=True),
        }
        print(f"{name:13s} held-out accuracy {meta_models[name]['held_out_accuracy']:.3f}  "
              f"macro F1 {meta_models[name]['held_out_macro_f1']:.3f}")
    for obj, fname in ((scaler, "scaler.pkl"), (encoder, "label_encoder.pkl")):
        with open(MODEL_DIR / fname, "wb") as f:
            pickle.dump(obj, f)

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "training_script": "scripts/train_and_save_all_models.py",
        "features": FEATURE_NAMES,
        "classes": encoder.classes_.tolist(),
        "split": {"method": "stratified train_test_split", "test_size": TEST_SIZE, "seed": SEED,
                  "n_train": int(len(idx_tr)), "n_test": int(len(idx_te)),
                  "class_counts": df["class"].value_counts().sort_index().to_dict()},
        "data": {"features_csv": FEATURES_CSV.relative_to(ROOT).as_posix(),
                 "features_sha256": sha256(FEATURES_CSV),
                 "reference_csv": REFERENCE_CSV.relative_to(ROOT).as_posix()},
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                        "numpy": np.__version__, "pandas": pd.__version__},
        "scaler_file": "scaler.pkl",
        "encoder_file": "label_encoder.pkl",
        "models": meta_models,
    }
    with open(MODEL_DIR / METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    # the frontend's comparison chart reads this file (held-out, macro-averaged)
    pd.DataFrame([{"Model": n, "Accuracy": m["held_out_accuracy"],
                   "Precision": m["per_class"]["macro avg"]["precision"],
                   "Recall": m["per_class"]["macro avg"]["recall"],
                   "F1-Score": m["held_out_macro_f1"]} for n, m in meta_models.items()]
                 ).to_csv(ROOT / "models" / "model_comparison.csv", index=False)
    print(f"saved to {MODEL_DIR} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
