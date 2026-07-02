"""
uplift_model.py
================
Implements two meta-learner uplift models from first principles on top of
sklearn/XGBoost (no dependency on the `causalml` package, which has heavy
native-build requirements that often fail to install — these meta-learners
are exactly what causalml implements under the hood, just made explicit and
portable here):

  T-Learner   : train two independent models — one on treated units, one on
                control units — and take the difference in their predicted
                churn probabilities as the estimated uplift.

  X-Learner   : an improvement on the T-learner for imbalanced or noisy
                treatment effects. Trains outcome models per arm, imputes
                per-unit treatment effects using the OPPOSITE arm's model,
                trains a second stage on those imputed effects, and blends
                the two stage-2 models using the propensity score.
                Generally reduces variance vs. the T-learner in real-world
                observational-ish settings.

Both are evaluated with the Qini curve / Qini coefficient — the standard
uplift-modeling analogue of the ROC/AUC for classification: it measures
how well the model's ranking concentrates true incremental conversions
(here, incremental *non*-churn) in the top-ranked customers.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.linear_model import LogisticRegression

from src.data_prep import load_and_prepare
from src.features import build_feature_matrix

MODELS_DIR = Path("models")


class TLearner:
    """Two independent regressors, one per treatment arm."""

    def __init__(self, **xgb_kwargs):
        params = dict(n_estimators=250, max_depth=4, learning_rate=0.05, random_state=42)
        params.update(xgb_kwargs)
        self.model_treated = XGBRegressor(**params)
        self.model_control = XGBRegressor(**params)

    def fit(self, X, treatment, y):
        t_mask = treatment == 1
        self.model_treated.fit(X[t_mask], y[t_mask])
        self.model_control.fit(X[~t_mask], y[~t_mask])
        return self

    def predict_uplift(self, X):
        """Uplift = P(churn | control) - P(churn | treated).
        Positive => the offer REDUCES churn probability for this customer."""
        p_control = self.model_control.predict(X)
        p_treated = self.model_treated.predict(X)
        return p_control - p_treated


class XLearner:
    """Two-stage meta-learner; generally lower variance than the T-learner."""

    def __init__(self, **xgb_kwargs):
        params = dict(n_estimators=250, max_depth=4, learning_rate=0.05, random_state=42)
        params.update(xgb_kwargs)
        self.mu_treated = XGBRegressor(**params)
        self.mu_control = XGBRegressor(**params)
        self.tau_treated = XGBRegressor(**params)
        self.tau_control = XGBRegressor(**params)
        self.propensity_model = LogisticRegression(max_iter=1000)

    def fit(self, X, treatment, y):
        X_arr = X.values if hasattr(X, "values") else X
        t_mask = treatment == 1

        # Stage 1: outcome models per arm (same as T-learner)
        self.mu_treated.fit(X_arr[t_mask], y[t_mask])
        self.mu_control.fit(X_arr[~t_mask], y[~t_mask])

        # Imputed treatment effects using the OPPOSITE arm's model
        d_treated = y[t_mask] - self.mu_control.predict(X_arr[t_mask])
        d_control = self.mu_treated.predict(X_arr[~t_mask]) - y[~t_mask]

        # Stage 2: model the imputed effects directly
        self.tau_treated.fit(X_arr[t_mask], d_treated)
        self.tau_control.fit(X_arr[~t_mask], d_control)

        # Propensity model to blend stage-2 predictions
        self.propensity_model.fit(X_arr, treatment)
        return self

    def predict_uplift(self, X):
        X_arr = X.values if hasattr(X, "values") else X
        g = self.propensity_model.predict_proba(X_arr)[:, 1]  # P(treated)
        tau_t = self.tau_treated.predict(X_arr)
        tau_c = self.tau_control.predict(X_arr)
        # tau_t/tau_c are fit on (Y_treated - Y_control_imputed), i.e. the
        # "treated minus control" convention (negative = treatment helped,
        # since Y = churn and we want churn to go DOWN). Our project-wide
        # convention is the opposite: positive uplift = treatment REDUCES
        # churn. Negate the blended estimate to match that convention.
        return -(g * tau_c + (1 - g) * tau_t)


def qini_curve(uplift_scores: np.ndarray, treatment: np.ndarray, outcome: np.ndarray):
    """
    Builds the Qini curve. `outcome` here should be 1 = churned, 0 = retained.
    We rank customers by predicted uplift (descending) and, at each cut point,
    compute the cumulative incremental RETENTION (non-churn) the model would
    have captured had the offer been targeted only at customers above that
    rank, versus what random targeting would achieve.

    Returns (fractions, qini_values, qini_coefficient)
    """
    order = np.argsort(-uplift_scores)
    treatment = treatment[order]
    outcome = outcome[order]
    n = len(outcome)

    retained = 1 - outcome  # 1 = retained (good outcome)
    cum_t_retained = np.cumsum(retained * treatment)
    cum_c_retained = np.cumsum(retained * (1 - treatment))
    cum_t_n = np.cumsum(treatment)
    cum_c_n = np.cumsum(1 - treatment)

    # avoid div-by-zero on the first few rows
    cum_c_n_safe = np.where(cum_c_n == 0, 1, cum_c_n)
    qini = cum_t_retained - cum_c_retained * (cum_t_n / cum_c_n_safe)

    fractions = np.arange(1, n + 1) / n
    qini_norm = qini / n

    # Qini coefficient = area between model curve and random-targeting line
    random_line = qini_norm[-1] * fractions
    trapz_fn = getattr(np, "trapezoid", None) or np.trapz
    qini_coefficient = float(trapz_fn(qini_norm - random_line, fractions))
    return fractions, qini_norm, qini_coefficient


def train_uplift_models(df: pd.DataFrame):
    y = df["churn_observed"].to_numpy()
    treatment = df["treatment"].to_numpy()
    X, encoder = build_feature_matrix(df, fit=True)

    from sklearn.model_selection import train_test_split
    idx_train, idx_test = train_test_split(
        np.arange(len(df)), test_size=0.25, random_state=42, stratify=treatment
    )

    X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]
    t_train, t_test = treatment[idx_train], treatment[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]

    t_learner = TLearner().fit(X_train, t_train, y_train)
    x_learner = XLearner().fit(X_train, t_train, y_train)

    uplift_t = t_learner.predict_uplift(X_test)
    uplift_x = x_learner.predict_uplift(X_test)

    _, _, qini_t = qini_curve(uplift_t, t_test, y_test)
    _, _, qini_x = qini_curve(uplift_x, t_test, y_test)
    print(f"[uplift_model] T-Learner Qini coefficient: {qini_t:.4f}")
    print(f"[uplift_model] X-Learner Qini coefficient: {qini_x:.4f}")

    # sanity check vs. ground-truth simulated uplift (only possible because
    # this is simulated data — in real deployments you'd rely on the Qini
    # curve above, since true ITE is never observable in reality)
    true_uplift_test = df["true_uplift"].to_numpy()[idx_test]
    corr_t = np.corrcoef(uplift_t, true_uplift_test)[0, 1]
    corr_x = np.corrcoef(uplift_x, true_uplift_test)[0, 1]
    print(f"[uplift_model] T-Learner corr. with ground-truth ITE: {corr_t:.4f}")
    print(f"[uplift_model] X-Learner corr. with ground-truth ITE: {corr_x:.4f}")

    MODELS_DIR.mkdir(exist_ok=True)
    best_model = x_learner if qini_x >= qini_t else t_learner
    best_name = "x_learner" if qini_x >= qini_t else "t_learner"
    joblib.dump(best_model, MODELS_DIR / "uplift_model.joblib")
    joblib.dump(encoder, MODELS_DIR / "uplift_encoder.joblib")
    print(f"[uplift_model] Saved best model ({best_name}) -> models/uplift_model.joblib")

    return {
        "t_learner": t_learner, "x_learner": x_learner, "encoder": encoder,
        "qini_t": qini_t, "qini_x": qini_x,
        "X_test": X_test, "t_test": t_test, "y_test": y_test,
        "uplift_t": uplift_t, "uplift_x": uplift_x,
    }


if __name__ == "__main__":
    data = load_and_prepare("data/telco_churn.csv")
    train_uplift_models(data)
