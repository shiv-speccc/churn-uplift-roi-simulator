"""
data_prep.py
============
Loads the IBM/Kaggle Telco Customer Churn dataset and layers a SIMULATED
randomized retention-offer experiment on top of it.

Why simulated treatment?
-------------------------
The public Telco Churn dataset is observational — it has no record of any
retention intervention (discount, call, email) ever being run. Real uplift
modeling requires a randomized (or quasi-randomized) treatment/control split
with an observed outcome under both arms. Since that data doesn't exist
publicly for this dataset, we simulate a randomized 50/50 "retention offer"
rollout with a HETEROGENEOUS treatment effect (i.e. the offer works better
on some customer segments than others — this heterogeneity is exactly what
uplift models are built to discover).

This is a standard, transparent technique for uplift-modeling portfolio
projects when no real experimental data is available. The simulation logic
is fully documented below so it never masquerades as real experimental
results.

Usage
-----
    from src.data_prep import load_and_prepare
    df = load_and_prepare("data/telco_churn.csv")   # real Kaggle file
    df = load_and_prepare(None)                      # synthetic fallback
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG_SEED = 42


def _load_raw(csv_path: str | None) -> pd.DataFrame:
    """Load the real Kaggle Telco Churn CSV if present, else synthesize
    a dataset with an identical schema so the pipeline is runnable
    out-of-the-box before you've downloaded the real file."""
    if csv_path and Path(csv_path).exists():
        expected_cols = {"customerID", "MonthlyCharges", "tenure", "TotalCharges", "Churn"}
        df = None
        last_err = None
        last_parsed_columns = None

        for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
            for kwargs in ({}, {"sep": None, "engine": "python"}):
                try:
                    candidate = pd.read_csv(csv_path, encoding=encoding, **kwargs)
                except (UnicodeDecodeError, pd.errors.ParserError) as e:
                    last_err = e
                    continue

                last_parsed_columns = list(candidate.columns)
                if expected_cols.issubset(set(candidate.columns)):
                    df = candidate
                    note = f"encoding='{encoding}'"
                    if kwargs:
                        note += ", auto-detected delimiter"
                    print(f"[data_prep] Loaded '{csv_path}' ({note}).")
                    break
            if df is not None:
                break

        if df is None:
            if last_parsed_columns is not None:
                raise ValueError(
                    f"'{csv_path}' was read, but is missing expected columns.\n"
                    f"Expected (at least): {sorted(expected_cols)}\n"
                    f"Columns found: {last_parsed_columns}\n"
                    f"This usually means the file isn't the Kaggle 'Telco Customer "
                    f"Churn' CSV, or got altered during download/opening. Re-download "
                    f"from https://www.kaggle.com/datasets/blastchar/telco-customer-churn "
                    f"and place it at data/telco_churn.csv without opening/saving it in "
                    f"Excel first."
                )
            if last_err is not None:
                raise last_err
            raise ValueError(f"Could not parse '{csv_path}' — file may be empty or corrupted.")

        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"])
        return df
    
    print(f"[data_prep] '{csv_path}' not found — generating a synthetic "
          f"Telco-schema dataset (n=7043) for development/testing.\n"
          f"Download the real dataset from Kaggle "
          f"('Telco Customer Churn' by blastchar) and place it at "
          f"data/telco_churn.csv to use real data.")
    return _synthesize_telco(n=7043)


def _synthesize_telco(n: int = 7043) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)

    gender = rng.choice(["Male", "Female"], n)
    senior = rng.choice([0, 1], n, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], n)
    dependents = rng.choice(["Yes", "No"], n, p=[0.3, 0.7])
    tenure = rng.integers(0, 73, n)
    phone = rng.choice(["Yes", "No"], n, p=[0.9, 0.1])
    multiple_lines = rng.choice(["Yes", "No", "No phone service"], n)
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24])
    paperless = rng.choice(["Yes", "No"], n, p=[0.59, 0.41])
    payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], n
    )
    monthly_charges = np.round(rng.normal(64, 30, n).clip(18, 120), 2)
    total_charges = np.round(monthly_charges * tenure + rng.normal(0, 50, n), 2).clip(0)

    online_security = rng.choice(["Yes", "No", "No internet service"], n)
    tech_support = rng.choice(["Yes", "No", "No internet service"], n)

    # Base churn probability driven by realistic risk factors
    logit = (
        -1.2
        + 1.4 * (contract == "Month-to-month")
        - 0.9 * (contract == "Two year")
        + 0.015 * (monthly_charges - 64)
        - 0.03 * tenure
        + 0.4 * (internet == "Fiber optic")
        - 0.3 * (online_security == "Yes")
        - 0.3 * (tech_support == "Yes")
        + 0.3 * (paperless == "Yes")
        - 0.2 * (partner == "Yes")
    )
    churn_prob = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, churn_prob)

    df = pd.DataFrame({
        "customerID": [f"{i:05d}-SYNTH" for i in range(n)],
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], n),
        "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], n),
        "TechSupport": tech_support,
        "StreamingTV": rng.choice(["Yes", "No", "No internet service"], n),
        "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], n),
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Churn": np.where(churn == 1, "Yes", "No"),
    })
    return df


def add_synthetic_retention_experiment(df: pd.DataFrame, seed: int = RNG_SEED) -> pd.DataFrame:
    """
    Simulates a randomized 50/50 rollout of a retention offer (e.g. a
    3-month discount) using the classic uplift-modeling customer taxonomy,
    so the project has a clean, explainable story rather than diffuse noise:

      Sure Things   (~20%) : long-tenure / two-year-contract customers who
                              were going to stay regardless. Low baseline
                              churn, ~zero treatment effect. Offering them
                              a discount is WASTED SPEND.
      Lost Causes   (~15%) : month-to-month, fiber-optic customers with no
                              tech support / security add-ons. High
                              baseline churn AND unresponsive to the offer
                              — their dissatisfaction isn't about price.
                              A pure "highest churn risk" policy wastes
                              a lot of budget here.
      Sleeping Dogs (~8%)  : loyal, longer-tenure customers for whom being
                              offered a discount backfires slightly (draws
                              attention to price) — a real, well-documented
                              uplift-modeling phenomenon.
      Persuadables  (~57%) : moderate baseline churn risk, but a LARGE
                              positive treatment effect — these are the
                              customers the offer actually saves, and the
                              ones a pure churn-risk model tends to under-
                              rank relative to Lost Causes.

    This produces genuine divergence between "target by churn risk" and
    "target by uplift" — which is the entire point of building an uplift
    model instead of a plain classifier.

    Adds columns: segment, treatment, churn_observed, true_churn_control,
    true_churn_treated, true_uplift  (see module docstring for semantics).
    """
    rng = np.random.default_rng(seed)
    n = len(df)

    contract = df["Contract"].to_numpy()
    tenure = df["tenure"].to_numpy()
    internet = df["InternetService"].to_numpy()
    tech_support = df["TechSupport"].to_numpy()
    online_security = df["OnlineSecurity"].to_numpy()
    monthly_charges = df["MonthlyCharges"].to_numpy()

    is_m2m = contract == "Month-to-month"
    is_long_commit = (contract == "Two year") | (tenure > 48)
    fiber_no_support = (internet == "Fiber optic") & (tech_support != "Yes") & (online_security != "Yes")
    high_charge = monthly_charges > np.median(monthly_charges)

    segment = np.full(n, "Persuadable", dtype=object)
    segment[is_long_commit & ~fiber_no_support] = "Sure Thing"
    segment[is_m2m & fiber_no_support & high_charge] = "Lost Cause"
    # Sleeping dogs: a slice of otherwise-loyal customers (assign after the above,
    # carved out of whoever is left as "Sure Thing"-ish but not already tagged)
    sleeping_dog_pool = (segment == "Sure Thing")
    sleeping_dog_mask = sleeping_dog_pool & (rng.random(n) < 0.35)
    segment[sleeping_dog_mask] = "Sleeping Dog"

    # Baseline churn probability (P(churn | control)) and treatment effect (ITE),
    # drawn per-archetype with individual noise for realism
    true_churn_control = np.zeros(n)
    ite = np.zeros(n)

    m = segment == "Sure Thing"
    true_churn_control[m] = rng.normal(0.04, 0.02, m.sum()).clip(0, 1)
    ite[m] = rng.normal(0.01, 0.01, m.sum())

    m = segment == "Lost Cause"
    true_churn_control[m] = rng.normal(0.78, 0.08, m.sum()).clip(0, 1)
    ite[m] = rng.normal(0.02, 0.02, m.sum())

    m = segment == "Sleeping Dog"
    true_churn_control[m] = rng.normal(0.10, 0.04, m.sum()).clip(0, 1)
    ite[m] = rng.normal(-0.06, 0.02, m.sum())

    m = segment == "Persuadable"
    true_churn_control[m] = rng.normal(0.45, 0.10, m.sum()).clip(0, 1)
    ite[m] = rng.normal(0.24, 0.06, m.sum())

    true_churn_control = true_churn_control.clip(0, 1)
    true_churn_treated = np.clip(true_churn_control - ite, 0, 1)

    treatment = rng.binomial(1, 0.5, n)
    churn_prob_observed = np.where(treatment == 1, true_churn_treated, true_churn_control)
    churn_observed = rng.binomial(1, churn_prob_observed)

    out = df.copy()
    out["segment"] = segment  # kept for analysis/dashboard only — NEVER a model feature
    out["treatment"] = treatment
    out["churn_observed"] = churn_observed
    out["Churn"] = np.where(churn_observed == 1, "Yes", "No")  # keep label consistent w/ observed outcome
    out["true_churn_control"] = true_churn_control
    out["true_churn_treated"] = true_churn_treated
    out["true_uplift"] = true_churn_control - true_churn_treated
    return out


def load_and_prepare(csv_path: str | None = "data/telco_churn.csv") -> pd.DataFrame:
    df = _load_raw(csv_path)
    df = add_synthetic_retention_experiment(df)
    return df


if __name__ == "__main__":
    data = load_and_prepare("data/telco_churn.csv")
    print(data.shape)
    print(data[["treatment", "churn_observed", "true_uplift"]].describe())
    Path("data").mkdir(exist_ok=True)
    data.to_parquet("data/prepared.parquet", index=False)
    print("Saved -> data/prepared.parquet")
