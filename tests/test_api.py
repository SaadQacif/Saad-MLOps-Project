"""The REST API, exercised through Flask's test client with real held-out images."""

import io
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image, ImageEnhance

import api as api_module
from scripts.drift_detection import MIN_SAMPLES

ROOT = Path(__file__).resolve().parent.parent
FEATURES = ROOT / "data" / "features" / "potato_features.csv"
REFERENCE = ROOT / "data" / "features" / "reference_features.csv"
N_DRIFT = 40  # images posted per drift scenario (>= MIN_SAMPLES)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(api_module, "PREDICTION_LOG", tmp_path / "predictions.csv")
    api_module.app.config["TESTING"] = True
    return api_module.app.test_client()


@pytest.fixture(scope="module")
def held_out():
    """Images the served model never saw in training."""
    ref = set(pd.read_csv(REFERENCE)["image_path"])
    df = pd.read_csv(FEATURES)
    df = df[~df["image_path"].isin(ref)]
    return pd.concat([g.sample(n=N_DRIFT // 3 + 1, random_state=1) for _, g in df.groupby("class")])


def _post(client, image: Image.Image, name="leaf.jpg"):
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=95)
    buf.seek(0)
    return client.post("/predict/image", data={"file": (buf, name)},
                       content_type="multipart/form-data")


def _post_file(client, path: Path):
    """Upload the original file bytes, as a client would."""
    with open(path, "rb") as f:
        return client.post("/predict/image", data={"file": (io.BytesIO(f.read()), path.name)},
                           content_type="multipart/form-data")


def test_health_reports_the_loaded_model(client):
    r = client.get("/health/status")
    assert r.status_code == 200, r.json
    assert r.json["model_loaded"] and r.json["held_out_accuracy"] >= 0.85


def test_predicts_held_out_images(client, held_out):
    correct = 0
    for _, row in held_out.iterrows():
        with open(ROOT / row["image_path"], "rb") as f:
            r = client.post("/predict/image", data={"file": (f, Path(row["image_path"]).name)},
                            content_type="multipart/form-data")
        assert r.status_code == 200, r.json
        body = r.json
        assert set(body["probabilities"]) == set(held_out["class"])
        assert abs(sum(body["probabilities"].values()) - 1) < 1e-6
        correct += body["prediction"] == row["class"]
    assert correct / len(held_out) >= 0.8, f"{correct}/{len(held_out)}"


def test_rejects_bad_requests(client):
    assert client.post("/predict/image", data={}, content_type="multipart/form-data").status_code == 400
    r = client.post("/predict/image", data={"file": (io.BytesIO(b"not an image"), "x.jpg")},
                    content_type="multipart/form-data")
    assert r.status_code == 400 and "not a readable image" in r.json["error"]


def test_admin_endpoint_needs_a_configured_key(client, monkeypatch):
    monkeypatch.delenv("API_ADMIN_KEY", raising=False)
    assert client.get("/health/metrics", headers={"X-API-KEY": "admin-secret-key"}).status_code == 401
    monkeypatch.setenv("API_ADMIN_KEY", "k-test")
    assert client.get("/health/metrics", headers={"X-API-KEY": "wrong"}).status_code == 401
    assert client.get("/health/metrics", headers={"X-API-KEY": "k-test"}).status_code == 200


def test_drift_endpoint_end_to_end(client, held_out):
    assert client.get("/monitoring/drift").json["status"] == "no_predictions_logged"
    rows = list(held_out.itertuples())
    # a few predictions: not enough to judge
    for row in rows[:5]:
        assert _post_file(client, ROOT / row.image_path).status_code == 200
    assert client.get("/monitoring/drift").json["summary"]["status"].startswith("insufficient_data")
    # held-out images from the training distribution: no drift
    for row in rows[5:N_DRIFT]:
        assert _post_file(client, ROOT / row.image_path).status_code == 200
    report = client.get(f"/monitoring/drift?window={N_DRIFT}").json
    assert report["summary"]["current_size"] >= MIN_SAMPLES
    assert report["summary"]["status"] == "no_drift", report
    # the same images, overexposed (e.g. a new camera setting): drift
    for row in rows[:N_DRIFT]:
        bright = ImageEnhance.Brightness(Image.open(ROOT / row.image_path)).enhance(1.6)
        assert _post(client, bright).status_code == 200
    report = client.get(f"/monitoring/drift?window={N_DRIFT}").json
    assert report["summary"]["status"] == "drift", report["summary"]
