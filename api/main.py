"""
api/main.py
============
FastAPI service exposing the trained churn and uplift models.

Run locally:
    uvicorn api.main:app --reload --port 8000

Endpoints:
    GET  /health
    POST /predict/churn        -> churn probability for a customer
    POST /predict/uplift       -> predicted uplift (retention offer effect)
    POST /predict/recommend    -> churn + uplift + a targeting recommendation
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features import build_feature_matrix, CATEGORICAL_COLS, NUMERIC_COLS

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

app = FastAPI(
    title="Churn + Retention-Uplift API",
    description="Serves churn-risk and retention-offer uplift predictions for the "
                 "customer-churn ROI simulator project.",
    version="1.0.0",
)

_churn_model = None
_churn_encoder = None
_uplift_model = None
_uplift_encoder = None


def _lazy_load():
    global _churn_model, _churn_encoder, _uplift_model, _uplift_encoder
    if _churn_model is None:
        churn_path = MODELS_DIR / "churn_model.joblib"
        encoder_path = MODELS_DIR / "churn_encoder.joblib"
        if not churn_path.exists():
            raise HTTPException(
                status_code=503,
                detail="Churn model not found. Run `python -m src.churn_model` to train it first."
            )
        _churn_model = joblib.load(churn_path)
        _churn_encoder = joblib.load(encoder_path)
    if _uplift_model is None:
        uplift_path = MODELS_DIR / "uplift_model.joblib"
        u_encoder_path = MODELS_DIR / "uplift_encoder.joblib"
        if not uplift_path.exists():
            raise HTTPException(
                status_code=503,
                detail="Uplift model not found. Run `python -m src.uplift_model` to train it first."
            )
        _uplift_model = joblib.load(uplift_path)
        _uplift_encoder = joblib.load(u_encoder_path)


class CustomerRecord(BaseModel):
    gender: str = Field(..., examples=["Female"])
    SeniorCitizen: int = Field(..., examples=[0])
    Partner: str = Field(..., examples=["Yes"])
    Dependents: str = Field(..., examples=["No"])
    tenure: int = Field(..., examples=[5])
    PhoneService: str = Field(..., examples=["Yes"])
    MultipleLines: str = Field(..., examples=["No"])
    InternetService: str = Field(..., examples=["Fiber optic"])
    OnlineSecurity: str = Field(..., examples=["No"])
    OnlineBackup: str = Field(..., examples=["No"])
    DeviceProtection: str = Field(..., examples=["No"])
    TechSupport: str = Field(..., examples=["No"])
    StreamingTV: str = Field(..., examples=["Yes"])
    StreamingMovies: str = Field(..., examples=["Yes"])
    Contract: str = Field(..., examples=["Month-to-month"])
    PaperlessBilling: str = Field(..., examples=["Yes"])
    PaymentMethod: str = Field(..., examples=["Electronic check"])
    MonthlyCharges: float = Field(..., examples=[95.50])
    TotalCharges: float = Field(..., examples=[477.50])


def _to_frame(record: CustomerRecord) -> pd.DataFrame:
    return pd.DataFrame([record.model_dump()])


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict/churn")
def predict_churn(record: CustomerRecord):
    _lazy_load()
    df = _to_frame(record)
    X, _ = build_feature_matrix(df, encoder=_churn_encoder, fit=False)
    proba = float(_churn_model.predict_proba(X)[0, 1])
    return {"churn_probability": round(proba, 4)}


@app.post("/predict/uplift")
def predict_uplift(record: CustomerRecord):
    _lazy_load()
    df = _to_frame(record)
    X, _ = build_feature_matrix(df, encoder=_uplift_encoder, fit=False)
    uplift = float(_uplift_model.predict_uplift(X)[0])
    return {
        "predicted_uplift": round(uplift, 4),
        "interpretation": (
            "Positive = the retention offer is predicted to REDUCE this customer's "
            "churn probability. Negative = the offer may increase churn risk "
            "('sleeping dog') — do not target."
        ),
    }


@app.post("/predict/recommend")
def predict_recommend(record: CustomerRecord):
    _lazy_load()
    df = _to_frame(record)

    X_churn, _ = build_feature_matrix(df, encoder=_churn_encoder, fit=False)
    churn_proba = float(_churn_model.predict_proba(X_churn)[0, 1])

    X_uplift, _ = build_feature_matrix(df, encoder=_uplift_encoder, fit=False)
    uplift = float(_uplift_model.predict_uplift(X_uplift)[0])

    if uplift < 0:
        recommendation = "DO NOT TARGET — offer likely to backfire (sleeping dog)"
    elif uplift < 0.03:
        recommendation = "LOW PRIORITY — minimal predicted incremental effect"
    elif churn_proba > 0.6 and uplift < 0.05:
        recommendation = "LOST CAUSE — high churn risk but offer unlikely to help; consider a different intervention"
    else:
        recommendation = "TARGET — strong predicted incremental retention from offer"

    return {
        "churn_probability": round(churn_proba, 4),
        "predicted_uplift": round(uplift, 4),
        "recommendation": recommendation,
    }
