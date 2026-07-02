"""
business_simulation.py
========================
Translates model rankings into a business decision: given a fixed
retention-offer budget, how much revenue do you save by targeting with
each strategy?

Strategies compared:
  1. Random targeting            (no model — baseline)
  2. Churn-risk targeting        (offer the highest churn-probability
                                   customers — the "obvious" but naive
                                   strategy most churn-only projects stop at)
  3. Uplift targeting            (offer customers with the highest
                                   predicted INCREMENTAL retention from
                                   the offer — this project's contribution)

Because we have simulated ground-truth potential outcomes (true_churn_control
/ true_churn_treated), we can compute the REAL revenue impact of each
targeting policy exactly — something you cannot do with only observational
data, but which is standard for validating a simulation/portfolio project.
"""

import numpy as np
import pandas as pd


def simulate_policy_value(
    df: pd.DataFrame,
    rank_scores: np.ndarray,
    offer_cost: float,
    customer_ltv: float,
    budget_fraction: float,
) -> dict:
    """
    Ranks customers by `rank_scores` (descending = highest priority for the
    offer) and evaluates the revenue outcome of offering to the
    budget_fraction% highest-ranked customers.

    Revenue accounting per customer offered:
        - pays offer_cost
        - customer retained (true_churn_treated == 0) -> keeps customer_ltv
        - customer churns anyway (true_churn_treated == 1) -> $0 revenue

    Customers NOT offered are evaluated under true_churn_control.
    """
    n = len(df)
    n_offer = int(n * budget_fraction)

    order = np.argsort(-rank_scores)
    offered_idx = order[:n_offer]
    not_offered_idx = order[n_offer:]

    tct = df["true_churn_treated"].to_numpy()
    tcc = df["true_churn_control"].to_numpy()

    revenue_offered = np.sum((1 - tct[offered_idx]) * customer_ltv) - n_offer * offer_cost
    revenue_not_offered = np.sum((1 - tcc[not_offered_idx]) * customer_ltv)

    total_revenue = revenue_offered + revenue_not_offered
    total_spend = n_offer * offer_cost
    customers_retained = (1 - tct[offered_idx]).sum() + (1 - tcc[not_offered_idx]).sum()

    return {
        "budget_fraction": budget_fraction,
        "n_offered": n_offer,
        "total_spend": total_spend,
        "total_revenue": total_revenue,
        "customers_retained": int(customers_retained),
    }


def compare_policies(
    df: pd.DataFrame,
    churn_risk_scores: np.ndarray,
    uplift_scores: np.ndarray,
    offer_cost: float = 50.0,
    customer_ltv: float = 1200.0,
    budget_fractions=(0.05, 0.1, 0.15, 0.2, 0.3, 0.5),
) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    random_scores = rng.random(len(df))

    rows = []
    for frac in budget_fractions:
        for name, scores in [
            ("Random", random_scores),
            ("Churn-risk ranking", churn_risk_scores),
            ("Uplift ranking", uplift_scores),
        ]:
            result = simulate_policy_value(df, scores, offer_cost, customer_ltv, frac)
            result["strategy"] = name
            rows.append(result)

    return pd.DataFrame(rows)


def summarize_best_budget(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """Revenue uplift of the Uplift strategy vs. Random and vs. Churn-risk,
    at each budget level — the headline numbers for a business audience."""
    pivot = comparison_df.pivot(index="budget_fraction", columns="strategy", values="total_revenue")
    pivot["Uplift vs Random ($)"] = pivot["Uplift ranking"] - pivot["Random"]
    pivot["Uplift vs Churn-risk ($)"] = pivot["Uplift ranking"] - pivot["Churn-risk ranking"]
    pivot["Uplift vs Random (%)"] = (pivot["Uplift vs Random ($)"] / pivot["Random"].abs()) * 100
    return pivot.round(2)


if __name__ == "__main__":
    from src.data_prep import load_and_prepare
    from src.churn_model import train_churn_model
    from src.uplift_model import train_uplift_models
    from src.features import build_feature_matrix

    df = load_and_prepare("data/telco_churn.csv")
    churn_model, churn_encoder, _ = train_churn_model(df)
    uplift_results = train_uplift_models(df)

    X_all, _ = build_feature_matrix(df, encoder=churn_encoder, fit=False)
    churn_risk_scores = churn_model.predict_proba(X_all)[:, 1]

    X_all_up, _ = build_feature_matrix(df, encoder=uplift_results["encoder"], fit=False)
    best_uplift_model = (
        uplift_results["x_learner"] if uplift_results["qini_x"] >= uplift_results["qini_t"]
        else uplift_results["t_learner"]
    )
    uplift_scores = best_uplift_model.predict_uplift(X_all_up)

    comparison = compare_policies(df, churn_risk_scores, uplift_scores)
    print(comparison)
    print("\n--- Business summary ---")
    print(summarize_best_budget(comparison))
