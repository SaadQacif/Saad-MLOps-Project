"""The single serving path: image -> 30 features -> scaler -> model -> class.

The training script (scripts/train_and_save_all_models.py), the Flask API
(src/api.py) and the Streamlit frontend all compute features through
``features_from_image``, so training and serving see identical inputs.
"""

from __future__ import annotations

import json
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:  # the API, frontend and tests start from different places
    sys.path.insert(0, str(_ROOT))

from scripts.extract_potato_features import extract_all_features_from_array, get_feature_names  # noqa: E402

FEATURE_NAMES: List[str] = get_feature_names()
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "bin"
METADATA_FILE = "model_metadata.json"


def to_rgb_array(image: Union[Image.Image, np.ndarray, str, Path]) -> np.ndarray:
    """Load or convert an image to an RGB uint8 array (alpha dropped, grey expanded)."""
    if isinstance(image, (str, Path)):
        with Image.open(image) as im:
            return np.array(im.convert("RGB"))
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255 if arr.max() <= 1.0 else arr, 0, 255).astype(np.uint8)
    return arr


def features_from_image(image) -> Dict[str, float]:
    """The 30 named features of one image, in FEATURE_NAMES order."""
    feats = extract_all_features_from_array(to_rgb_array(image))
    if feats is None:
        raise ValueError("feature extraction failed for this image")
    return {name: float(feats[name]) for name in FEATURE_NAMES}


@dataclass
class Predictor:
    name: str
    model: object
    scaler: object
    encoder: object
    metadata: Dict = field(default_factory=dict)

    def predict_features(self, features: Dict[str, float]) -> Dict:
        x = np.array([[features[n] for n in FEATURE_NAMES]], dtype=float)
        probs = self.model.predict_proba(self.scaler.transform(x))[0]
        order = np.argsort(probs)[::-1]
        classes = self.encoder.inverse_transform(order)
        return {
            "prediction": str(classes[0]),
            "confidence": float(probs[order[0]]),
            "probabilities": {str(c): float(probs[i]) for c, i in zip(classes, order)},
            "model": self.name,
        }

    def predict_image(self, image) -> Dict:
        feats = features_from_image(image)
        out = self.predict_features(feats)
        out["features"] = feats
        return out


def load_metadata(model_dir: Union[str, Path] = DEFAULT_MODEL_DIR) -> Dict:
    path = Path(model_dir) / METADATA_FILE
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def available_models(model_dir: Union[str, Path] = DEFAULT_MODEL_DIR) -> List[str]:
    """Models listed in the metadata written by the training script."""
    return list(load_metadata(model_dir).get("models", {}))


def load_predictor(name: Optional[str] = None,
                   model_dir: Union[str, Path] = DEFAULT_MODEL_DIR) -> Predictor:
    """Load one trained model with its scaler and label encoder.

    ``name`` defaults to the model with the best held-out accuracy in the metadata.
    """
    model_dir = Path(model_dir)
    meta = load_metadata(model_dir)
    models = meta.get("models", {})
    if not models:
        raise FileNotFoundError(f"no {METADATA_FILE} with trained models in {model_dir}")
    if name is None:
        name = max(models, key=lambda m: models[m]["held_out_accuracy"])
    if name not in models:
        raise KeyError(f"unknown model {name!r}; available: {sorted(models)}")

    def load(fname):
        with open(model_dir / fname, "rb") as f:
            return pickle.load(f)

    return Predictor(name=name, model=load(models[name]["file"]), scaler=load(meta["scaler_file"]),
                     encoder=load(meta["encoder_file"]),
                     metadata={k: v for k, v in meta.items() if k != "models"} | {"model": models[name]})
