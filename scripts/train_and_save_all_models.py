#!/usr/bin/env python
"""
Train and save SVM, MLP, and Random Forest models for potato disease classification
- Trains both simple (30 features) and complex (121 features) models
- Saves models, scalers, and encoders to the correct directories
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- CONFIG ---
FEATURES_CSV = Path(__file__).parent.parent / "data/features/potato_features.csv"
MODEL_DIR = Path(__file__).parent.parent / "models/bin"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# --- LOAD DATA ---
df = pd.read_csv(FEATURES_CSV)
feature_cols = [col for col in df.columns if col not in ["image_path", "class"]]
X = df[feature_cols].values
y = df["class"].values

# --- PREPROCESS ---
le = LabelEncoder()
y_enc = le.fit_transform(y)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- SAVE SCALER AND ENCODER ---
with open(MODEL_DIR / "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open(MODEL_DIR / "label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# --- TRAIN MODELS ---
models = {
    "svm_model.pkl": SVC(probability=True, random_state=42),
    "mlp_model.pkl": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42),
    "randomforest_model.pkl": RandomForestClassifier(n_estimators=100, random_state=42)
}

for fname, model in models.items():
    print(f"Training {fname}...")
    model.fit(X_scaled, y_enc)
    with open(MODEL_DIR / fname, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved {fname} to {MODEL_DIR}")

print("All models, scaler, and encoder saved to", MODEL_DIR)
