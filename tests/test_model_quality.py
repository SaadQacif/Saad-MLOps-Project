"""Quality gates on the code paths the API and the Streamlit frontend actually use.

* train/serve skew: features recomputed from the images through src.serving must
  equal the features the models were trained on (data/features/potato_features.csv);
* the committed models must load with the pinned scikit-learn and be served by the
  frontend's own model discovery and prediction code;
* held-out accuracy: the training recipe, refit on its 80/20 split, must reproduce
  the metadata of the committed models and reach MIN_HELD_OUT_ACCURACY;
* the frontend script runs without raising under the pinned Streamlit.
"""

import importlib.util
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from sklearn.base import clone
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src import serving

ROOT = Path(__file__).resolve().parent.parent
FEATURES_CSV = ROOT / "data" / "features" / "potato_features.csv"
BIN = ROOT / "models" / "bin"
FRONTEND = ROOT / "src" / "enhanced_complete_frontend_fixed.py"
TRAIN_SCRIPT = ROOT / "scripts" / "train_and_save_all_models.py"

MIN_HELD_OUT_ACCURACY = 0.85
SKEW_PER_CLASS = 10
SKEW_RTOL = 1e-9


@pytest.fixture(scope="module")
def features():
    df = pd.read_csv(FEATURES_CSV)
    assert list(df.columns[:30]) == serving.FEATURE_NAMES
    return df


@pytest.fixture(scope="module")
def recipe():
    spec = importlib.util.spec_from_file_location("train_script", TRAIN_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # main() is guarded; this only defines the recipe
    return mod


def _sample(df, n_per_class, seed=0):
    return pd.concat([g.sample(n=n_per_class, random_state=seed) for _, g in df.groupby("class")])


def test_image_paths_are_relative_and_exist(features):
    assert not features["image_path"].str.contains(r"^[A-Za-z]:|\\", regex=True).any()
    missing = [p for p in features["image_path"] if not (ROOT / p).exists()]
    assert not missing, missing[:5]


def test_serving_features_match_training_features(features):
    for _, row in _sample(features, SKEW_PER_CLASS).iterrows():
        got = serving.features_from_image(ROOT / row["image_path"])
        want = row[serving.FEATURE_NAMES].to_numpy(dtype=float)
        np.testing.assert_allclose([got[k] for k in serving.FEATURE_NAMES], want,
                                   rtol=SKEW_RTOL, err_msg=row["image_path"])


@pytest.mark.parametrize("path", sorted(BIN.glob("*.pkl")), ids=lambda p: p.name)
def test_committed_artifacts_load_with_pinned_sklearn(path):
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        with open(path, "rb") as f:
            pickle.load(f)


def test_frontend_serves_every_model_it_lists(features):
    """Every model the frontend offers must predict a valid class for a real image."""
    import enhanced_complete_frontend_fixed as fe  # imported, not run: main() is guarded
    models = fe.EnhancedModelManager(str(ROOT / "models")).discover_available_models()
    assert sorted(m["model_name"] for m in models.values()) == sorted(serving.available_models(BIN))
    image = Image.open(ROOT / _sample(features, 1).iloc[0]["image_path"])
    for label, info in models.items():
        out = fe.predict_with_model(image, info)
        assert "error" not in out, (label, out)
        assert out["prediction"] in set(features["class"])


def test_retraining_reproduces_metadata_and_clears_accuracy_gate(features, recipe):
    meta = serving.load_metadata(BIN)
    X = features[serving.FEATURE_NAMES].to_numpy(dtype=float)
    y = LabelEncoder().fit_transform(features["class"])
    idx_tr, idx_te = train_test_split(np.arange(len(features)), test_size=recipe.TEST_SIZE,
                                      random_state=recipe.SEED, stratify=y)
    assert (len(idx_tr), len(idx_te)) == (meta["split"]["n_train"], meta["split"]["n_test"])
    scaler = StandardScaler().fit(X[idx_tr])
    scores = {}
    for name, est in recipe.MODELS.items():
        model = clone(est).fit(scaler.transform(X[idx_tr]), y[idx_tr])
        scores[name] = float((model.predict(scaler.transform(X[idx_te])) == y[idx_te]).mean())
    print("\nheld-out accuracy:", {k: round(v, 3) for k, v in scores.items()})
    for name, acc in scores.items():
        assert acc == pytest.approx(meta["models"][name]["held_out_accuracy"], abs=1e-12), name
        assert acc >= MIN_HELD_OUT_ACCURACY, (name, acc)


def test_frontend_script_runs_without_exceptions():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(FRONTEND), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
