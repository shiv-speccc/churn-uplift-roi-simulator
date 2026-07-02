"""
features.py
============
Shared feature engineering for both the baseline churn model and the
uplift models. Keeping this in one place guarantees the API, the training
scripts, and the Streamlit app all transform raw customer records
identically.
"""

import pandas as pd

CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
NUMERIC_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS


def build_feature_matrix(df: pd.DataFrame, encoder=None, fit: bool = False):
    """
    One-hot encodes categoricals + passes numerics through.
    If `encoder` (a fitted sklearn OneHotEncoder) is provided, reuses it
    (inference-time). If `fit=True`, fits a new one (training-time).
    Returns (X_df, encoder).
    """
    from sklearn.preprocessing import OneHotEncoder
    import numpy as np

    cat = df[CATEGORICAL_COLS].astype(str)
    num = df[NUMERIC_COLS].astype(float)

    if fit or encoder is None:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        cat_enc = encoder.fit_transform(cat)
    else:
        cat_enc = encoder.transform(cat)

    cat_names = encoder.get_feature_names_out(CATEGORICAL_COLS)
    cat_df = pd.DataFrame(cat_enc, columns=cat_names, index=df.index)

    X = pd.concat([num.reset_index(drop=True), cat_df.reset_index(drop=True)], axis=1)
    return X, encoder
