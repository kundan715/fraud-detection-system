import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

MODELS = Path(__file__).resolve().parent.parent / "models"
pytestmark = pytest.mark.skipif(not (MODELS / "model.joblib").exists(), reason="train the model first")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_metrics_has_models(client):
    meta = client.get("/metrics").json()
    assert {"Logistic Regression", "Random Forest", "XGBoost"} <= set(meta["models"])
    assert len(meta["features"]) == 29


def test_fraud_sample_scores_higher_than_legit(client):
    fraud = client.get("/sample/fraud").json()["features"]
    legit = client.get("/sample/legit").json()["features"]
    p_f = client.post("/predict", json={"features": fraud}).json()
    p_l = client.post("/predict", json={"features": legit}).json()
    assert p_f["fraud_probability"] > p_l["fraud_probability"]
    assert len(p_f["explanation"]) == 8


def test_missing_features_rejected(client):
    r = client.post("/predict", json={"features": {"Amount": 10}})
    assert r.status_code == 422


def test_negative_amount_rejected(client):
    feats = client.get("/sample/legit").json()["features"]
    feats["Amount"] = -5
    assert client.post("/predict", json={"features": feats}).status_code == 422


def test_bad_sample_kind(client):
    assert client.get("/sample/nope").status_code == 404
