"""Drift detection decides on the right feature, does not cry wolf on held-out data
from the training distribution, and does flag a real shift."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.drift_detection import MIN_SAMPLES, DriftDetector
from src import serving

ROOT = Path(__file__).resolve().parent.parent
FEATURES = ROOT / "data" / "features" / "potato_features.csv"
REFERENCE = ROOT / "data" / "features" / "reference_features.csv"
COLS = serving.FEATURE_NAMES + ["class"]


@pytest.fixture(scope="module")
def data():
    ref = pd.read_csv(REFERENCE)
    held_out = pd.read_csv(FEATURES)
    held_out = held_out[~held_out["image_path"].isin(ref["image_path"])]
    return ref[COLS], held_out[COLS]


@pytest.fixture(scope="module")
def varying(data):
    """Features that vary in the training data (the extractor still returns constant
    placeholders for some; a shift of a constant column is not a shift)."""
    ref, _ = data
    return [c for c in serving.FEATURE_NAMES if ref[c].std() > 0]


def test_held_out_images_show_no_drift(data):
    ref, held_out = data
    report = DriftDetector().analyse(ref, held_out)
    assert report["summary"]["status"] == "no_drift", report["summary"]


@pytest.mark.parametrize("feature", ["eccentricity", "green_mean", "entropy_variation_rhythm"])
def test_shift_in_one_feature_is_attributed_to_that_feature(data, feature):
    ref, held_out = data
    shifted = held_out.copy()
    shifted[feature] += ref[feature].std()
    report = DriftDetector().analyse(ref, shifted)
    assert report["summary"]["drifted_features"] == [feature]


def test_broad_shift_is_flagged(data, varying):
    ref, held_out = data
    shifted = held_out.copy()
    for col in varying:
        shifted[col] = shifted[col] * 1.5
    report = DriftDetector().analyse(ref, shifted)
    assert report["summary"]["drift_detected"] is True
    assert set(report["summary"]["drifted_features"]) == set(varying)


def test_small_window_is_insufficient_not_clean(data):
    ref, held_out = data
    report = DriftDetector().analyse(ref, held_out.head(MIN_SAMPLES - 1))
    assert report["summary"]["drift_detected"] is None
    assert report["summary"]["status"].startswith("insufficient_data")
