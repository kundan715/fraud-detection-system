"""FastAPI service that serves the trained fraud model with per-prediction explanations."""
import json
import os
import random
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE = Path(__file__).resolve().parent.parent
MODELS = BASE / "models"
TOP_N = 8


class Transaction(BaseModel):
    features: dict[str, float] = Field(
        ..., description="Map of feature name -> value. Needs Amount and V1..V28."
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ready = (MODELS / "model.joblib").exists() and (MODELS / "metadata.json").exists()
    if app.state.ready:
        app.state.model = joblib.load(MODELS / "model.joblib")
        app.state.meta = json.loads((MODELS / "metadata.json").read_text())
        app.state.samples = json.loads((MODELS / "samples.json").read_text())
    yield


app = FastAPI(title="Fraud Detection API", version="1.0.0", lifespan=lifespan)

origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])


def require_model():
    if not app.state.ready:
        raise HTTPException(503, "Model not trained yet. Run `python -m ml.train` in backend/.")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": app.state.ready}


@app.get("/metrics")
def metrics():
    require_model()
    return app.state.meta


@app.get("/sample/{kind}")
def sample(kind: str):
    require_model()
    if kind not in ("fraud", "legit"):
        raise HTTPException(404, "kind must be 'fraud' or 'legit'")
    return {"kind": kind, "features": random.choice(app.state.samples[kind])}


@app.post("/predict")
def predict(tx: Transaction):
    require_model()
    feats = app.state.meta["features"]
    missing = [f for f in feats if f not in tx.features]
    if missing:
        raise HTTPException(422, f"Missing features: {missing}")
    if tx.features["Amount"] < 0:
        raise HTTPException(422, "Amount must be >= 0")

    X = pd.DataFrame([[tx.features[f] for f in feats]], columns=feats)
    proba = float(app.state.model.predict_proba(X)[0, 1])

    # Native XGBoost SHAP values (log-odds contribution of each feature).
    contribs = app.state.model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[0]
    pairs = sorted(zip(feats, contribs[:-1]), key=lambda t: -abs(t[1]))[:TOP_N]

    threshold = app.state.meta["threshold"]
    level = "high" if proba >= threshold else "medium" if proba >= threshold / 2 else "low"
    return {
        "fraud_probability": round(proba, 6),
        "is_fraud": proba >= threshold,
        "threshold": threshold,
        "risk_level": level,
        "explanation": [
            {
                "feature": f,
                "value": float(tx.features[f]),
                "contribution": round(float(c), 4),
                "direction": "fraud" if c > 0 else "legit",
            }
            for f, c in pairs
        ],
    }
