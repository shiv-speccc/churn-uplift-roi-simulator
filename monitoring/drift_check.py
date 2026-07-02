"""
monitoring/drift_check.py
===========================
Generates a data-drift report comparing a "reference" window (the data the
models were trained on) against a "current" window (freshly-scored
production data), using Evidently.

This is the piece most student portfolios skip entirely: a model that
performs well at launch silently degrades as the customer base and their
behavior shift over time. This script simulates that by injecting a
realistic distribution shift (e.g. rising fiber-optic adoption, rising
monthly charges post price-increase) into a "current" snapshot and
detects it.

Run:
    python -m monitoring.drift_check
Outputs:
    monitoring/drift_report.html
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.data_prep import load_and_prepare
from src.features import FEATURE_COLS


def _simulate_current_snapshot(reference: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """Simulates a 'current' production snapshot with realistic drift:
    a price increase (higher MonthlyCharges) and higher fiber-optic
    adoption, plus a slightly younger tenure mix (more new signups)."""
    rng = np.random.default_rng(seed)
    current = reference.sample(n=len(reference), replace=True, random_state=seed).copy()

    current["MonthlyCharges"] = (current["MonthlyCharges"] * 1.12 + rng.normal(0, 3, len(current))).clip(18, 150)
    flip_mask = rng.random(len(current)) < 0.15
    current.loc[flip_mask, "InternetService"] = "Fiber optic"
    current["tenure"] = (current["tenure"] * 0.8).astype(int).clip(0, 72)

    return current


def run_drift_report(output_path: str = "monitoring/drift_report.html"):
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
    except ImportError:
        print(
            "[drift_check] `evidently` is not installed. Install it with:\n"
            "    pip install evidently\n"
            "Skipping HTML report generation — falling back to a lightweight "
            "manual drift summary instead.\n"
        )
        _manual_fallback()
        return

    df = load_and_prepare("data/telco_churn.csv")
    reference = df[FEATURE_COLS]
    current = _simulate_current_snapshot(df)[FEATURE_COLS]

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    Path(output_path).parent.mkdir(exist_ok=True)
    report.save_html(output_path)
    print(f"[drift_check] Drift report saved -> {output_path}")


def _manual_fallback():
    """Lightweight drift summary (mean shift + KS test) requiring only
    scipy/pandas, used when evidently isn't installed."""
    from scipy.stats import ks_2samp

    df = load_and_prepare("data/telco_churn.csv")
    reference = df
    current = _simulate_current_snapshot(df)

    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    print("\n[drift_check] Manual drift summary (KS test, numeric features)")
    print("-" * 60)
    for col in numeric_cols:
        stat, p_value = ks_2samp(reference[col], current[col])
        drift_flag = "DRIFT DETECTED" if p_value < 0.05 else "no significant drift"
        print(f"{col:20s} | KS stat={stat:.4f} | p={p_value:.4g} | {drift_flag}")

    print("-" * 60)
    for col in ["InternetService", "Contract"]:
        ref_dist = reference[col].value_counts(normalize=True).round(3).to_dict()
        cur_dist = current[col].value_counts(normalize=True).round(3).to_dict()
        print(f"{col}:\n  reference={ref_dist}\n  current  ={cur_dist}")


if __name__ == "__main__":
    run_drift_report()
