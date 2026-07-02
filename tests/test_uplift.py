"""
tests/test_uplift.py
======================
A few targeted regression tests — notably one that pins down the
T-Learner / X-Learner sign convention (positive = offer reduces churn),
since a flipped sign there is a silent, high-consequence bug: it would
have the model recommend targeting the customers LEAST likely to be
helped by an offer.

Run: pytest tests/
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data_prep import load_and_prepare
from src.uplift_model import TLearner, XLearner, qini_curve
from src.features import build_feature_matrix


def _small_dataset(n=800):
    df = load_and_prepare(None)
    return df.sample(n=n, random_state=0).reset_index(drop=True)


def test_uplift_sign_convention_t_learner():
    df = _small_dataset()
    X, _ = build_feature_matrix(df, fit=True)
    treatment = df["treatment"].to_numpy()
    y = df["churn_observed"].to_numpy()

    model = TLearner().fit(X, treatment, y)
    uplift = model.predict_uplift(X)

    # Persuadable segment should have clearly positive mean predicted uplift
    persuadable_mask = (df["segment"] == "Persuadable").to_numpy()
    assert uplift[persuadable_mask].mean() > 0.05, (
        "T-Learner uplift should be positive (offer reduces churn) for Persuadables"
    )


def test_uplift_sign_convention_x_learner():
    df = _small_dataset()
    X, _ = build_feature_matrix(df, fit=True)
    treatment = df["treatment"].to_numpy()
    y = df["churn_observed"].to_numpy()

    model = XLearner().fit(X, treatment, y)
    uplift = model.predict_uplift(X)

    persuadable_mask = (df["segment"] == "Persuadable").to_numpy()
    sure_thing_mask = (df["segment"] == "Sure Thing").to_numpy()

    # Persuadables should score meaningfully higher uplift than Sure Things
    assert uplift[persuadable_mask].mean() > uplift[sure_thing_mask].mean(), (
        "X-Learner should rank Persuadables above Sure Things — if this fails, "
        "check for a sign flip in XLearner.predict_uplift()"
    )


def test_qini_curve_shape():
    rng = np.random.default_rng(0)
    n = 500
    treatment = rng.binomial(1, 0.5, n)
    outcome = rng.binomial(1, 0.3, n)
    scores = rng.random(n)

    fractions, qini_values, qini_coef = qini_curve(scores, treatment, outcome)
    assert len(fractions) == len(qini_values) == n
    assert isinstance(qini_coef, float)


def test_lost_cause_gets_low_uplift_relative_to_churn_risk():
    """The core business claim of this project: Lost Cause customers have
    HIGH churn risk but LOW uplift — this is what makes uplift targeting
    beat naive churn-risk targeting. Regression-test that this holds."""
    df = _small_dataset(n=2000)
    X, encoder = build_feature_matrix(df, fit=True)
    treatment = df["treatment"].to_numpy()
    y = df["churn_observed"].to_numpy()

    model = TLearner().fit(X, treatment, y)
    uplift = model.predict_uplift(X)

    lost_cause = (df["segment"] == "Lost Cause").to_numpy()
    persuadable = (df["segment"] == "Persuadable").to_numpy()

    assert uplift[lost_cause].mean() < uplift[persuadable].mean(), (
        "Lost Cause customers should have lower predicted uplift than Persuadables "
        "despite higher churn risk — this is the project's core business insight."
    )
