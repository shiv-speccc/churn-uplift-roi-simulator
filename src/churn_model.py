"""
churn_model.py
===============
Baseline supervised churn classifier (XGBoost). This answers "who is
likely to churn?" — the question most portfolio churn projects stop at.
This script exists mainly as a comparison point for the uplift models in
uplift_model.py, which answer the more useful question: "who will churn
*and* can be saved by an intervention?"
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier

from src.data_prep import load_and_prepare
from src.features import build_feature_matrix

MODEL_PATH = Path("models/churn_model.joblib")
ENCODER_PATH = Path("models/churn_encoder.joblib")


def train_churn_model(df: pd.DataFrame):
    y = (df["churn_observed"] == 1).astype(int)
    X, encoder = build_feature_matrix(df, fit=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="auc",
        random_state=42,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    print(f"[churn_model] Test ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, proba > 0.5))

    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    print(f"[churn_model] Saved model -> {MODEL_PATH}")
    return model, encoder, auc


if __name__ == "__main__":
    data = load_and_prepare("data/telco_churn.csv")
    train_churn_model(data)
