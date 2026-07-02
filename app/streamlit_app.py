"""
app/streamlit_app.py
=====================
Business-facing dashboard: lets a non-technical stakeholder explore the
ROI of different retention-targeting strategies and score individual
customers, without touching the API or notebooks directly.

Run:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_prep import load_and_prepare
from src.features import build_feature_matrix
from src.business_simulation import compare_policies, summarize_best_budget

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

st.set_page_config(page_title="Retention ROI Simulator", layout="wide")


@st.cache_data
def get_data():
    return load_and_prepare("data/telco_churn.csv")


@st.cache_resource
def get_models():
    churn_model = joblib.load(MODELS_DIR / "churn_model.joblib")
    churn_encoder = joblib.load(MODELS_DIR / "churn_encoder.joblib")
    uplift_model = joblib.load(MODELS_DIR / "uplift_model.joblib")
    uplift_encoder = joblib.load(MODELS_DIR / "uplift_encoder.joblib")
    return churn_model, churn_encoder, uplift_model, uplift_encoder


st.title("📉 Customer Churn → Retention-Offer ROI Simulator")
st.caption(
    "Compares naive churn-risk targeting against uplift-based targeting for a "
    "retention offer, and quantifies the revenue difference under a fixed budget."
)

if not (MODELS_DIR / "churn_model.joblib").exists() or not (MODELS_DIR / "uplift_model.joblib").exists():
    st.error(
        "Models not found. Run these first from the project root:\n\n"
        "```\npython -m src.churn_model\npython -m src.uplift_model\n```"
    )
    st.stop()

df = get_data()
churn_model, churn_encoder, uplift_model, uplift_encoder = get_models()

X_churn, _ = build_feature_matrix(df, encoder=churn_encoder, fit=False)
churn_scores = churn_model.predict_proba(X_churn)[:, 1]

X_uplift, _ = build_feature_matrix(df, encoder=uplift_encoder, fit=False)
uplift_scores = uplift_model.predict_uplift(X_uplift)

tab1, tab2, tab3 = st.tabs(["💰 Business ROI Simulator", "🧭 Customer Segments", "🔎 Score a Customer"])

# ----------------------------------------------------------------------
with tab1:
    st.subheader("Budget-constrained targeting comparison")
    c1, c2, c3 = st.columns(3)
    offer_cost = c1.number_input("Retention offer cost per customer ($)", 5.0, 500.0, 50.0, step=5.0)
    customer_ltv = c2.number_input("Customer LTV if retained ($)", 100.0, 10000.0, 1200.0, step=50.0)
    budget_fraction = c3.slider("Budget: % of customer base to target", 0.05, 0.75, 0.20, step=0.05)

    comparison = compare_policies(
        df, churn_scores, uplift_scores,
        offer_cost=offer_cost, customer_ltv=customer_ltv,
        budget_fractions=sorted(set([0.05, 0.1, 0.15, 0.2, 0.3, 0.5, budget_fraction])),
    )

    fig = px.line(
        comparison, x="budget_fraction", y="total_revenue", color="strategy",
        markers=True, labels={"budget_fraction": "Fraction of customers offered the deal",
                               "total_revenue": "Total projected revenue ($)"},
        title="Projected revenue by targeting strategy and budget",
    )
    fig.add_vline(x=budget_fraction, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)

    at_budget = comparison[comparison["budget_fraction"] == budget_fraction]
    random_rev = at_budget.loc[at_budget.strategy == "Random", "total_revenue"].values[0]
    risk_rev = at_budget.loc[at_budget.strategy == "Churn-risk ranking", "total_revenue"].values[0]
    uplift_rev = at_budget.loc[at_budget.strategy == "Uplift ranking", "total_revenue"].values[0]

    m1, m2, m3 = st.columns(3)
    m1.metric("Uplift strategy revenue", f"${uplift_rev:,.0f}")
    m2.metric("vs. Random targeting", f"${uplift_rev - random_rev:,.0f}", f"{(uplift_rev-random_rev)/abs(random_rev)*100:.1f}%")
    m3.metric("vs. Naive churn-risk targeting", f"${uplift_rev - risk_rev:,.0f}", f"{(uplift_rev-risk_rev)/abs(risk_rev)*100:.1f}%")

    st.markdown("#### Revenue summary across all budget levels")
    st.dataframe(summarize_best_budget(comparison), use_container_width=True)

    st.info(
        "💡 **Why uplift beats churn-risk targeting:** a churn-risk model spends budget "
        "on high-risk customers regardless of whether the offer would actually change "
        "their behavior — including 'Lost Cause' customers who churn no matter what. "
        "The uplift model instead prioritizes 'Persuadable' customers: moderate-risk "
        "customers the offer can genuinely save."
    )

# ----------------------------------------------------------------------
with tab2:
    st.subheader("Customer segment breakdown (ground-truth, for validation)")
    st.caption(
        "In this simulated experiment we know the true customer archetype. In a real "
        "deployment you would never observe this directly — it's shown here only to "
        "validate that the uplift model recovers meaningful structure."
    )
    seg_counts = df["segment"].value_counts().reset_index()
    seg_counts.columns = ["segment", "count"]
    fig_seg = px.bar(seg_counts, x="segment", y="count", color="segment", title="Customer segment sizes")
    st.plotly_chart(fig_seg, use_container_width=True)

    df_plot = df.copy()
    df_plot["predicted_uplift"] = uplift_scores
    df_plot["churn_risk"] = churn_scores
    fig_scatter = px.scatter(
        df_plot.sample(min(2000, len(df_plot)), random_state=1),
        x="churn_risk", y="predicted_uplift", color="segment", opacity=0.6,
        labels={"churn_risk": "Predicted churn risk", "predicted_uplift": "Predicted uplift"},
        title="Churn risk vs. predicted uplift, colored by true segment",
    )
    fig_scatter.add_hline(y=0, line_dash="dot", line_color="gray")
    st.plotly_chart(fig_scatter, use_container_width=True)

# ----------------------------------------------------------------------
with tab3:
    st.subheader("Score an individual customer")
    col1, col2, col3 = st.columns(3)
    with col1:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 6)
        monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 90.0)
    with col2:
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check",
                                                    "Bank transfer (automatic)", "Credit card (automatic)"])
    with col3:
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        gender = st.selectbox("Gender", ["Male", "Female"])

    record = pd.DataFrame([{
        "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
        "tenure": tenure, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": internet, "OnlineSecurity": online_security, "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": tech_support, "StreamingTV": "Yes", "StreamingMovies": "Yes",
        "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges, "TotalCharges": monthly_charges * max(tenure, 1),
    }])

    X_c, _ = build_feature_matrix(record, encoder=churn_encoder, fit=False)
    churn_p = float(churn_model.predict_proba(X_c)[0, 1])
    X_u, _ = build_feature_matrix(record, encoder=uplift_encoder, fit=False)
    uplift_p = float(uplift_model.predict_uplift(X_u)[0])

    st.markdown("---")
    m1, m2 = st.columns(2)
    m1.metric("Predicted churn probability", f"{churn_p:.1%}")
    m2.metric("Predicted uplift from retention offer", f"{uplift_p:+.1%}")

    if uplift_p < 0:
        st.error("🚫 DO NOT TARGET — offer predicted to backfire ('sleeping dog').")
    elif uplift_p < 0.03:
        st.warning("⚪ LOW PRIORITY — minimal predicted incremental effect.")
    elif churn_p > 0.6 and uplift_p < 0.05:
        st.warning("⚠️ LOST CAUSE — high churn risk but the offer is unlikely to change the outcome.")
    else:
        st.success("✅ TARGET — this customer is a strong candidate for the retention offer.")
