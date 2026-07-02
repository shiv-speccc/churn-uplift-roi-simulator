# 📉 Customer Churn Prediction + Uplift Modeling + Retention ROI Simulator

**The question most churn projects answer:** *"Who is going to churn?"*
**The question this project answers:** *"Who is going to churn **and can actually be saved** by a retention offer — and how much money does targeting the right customers save versus the naive approach?"*

That distinction is the entire point of this project. A plain churn classifier ranks customers by risk. But high risk doesn't mean *savable* — some high-risk customers will churn no matter what you offer them ("Lost Causes"), and some low-risk customers would respond well to an offer you never gave them. **Uplift modeling** predicts the *incremental* effect of an intervention per customer, not just their baseline risk — and this project shows, in dollars, why that distinction matters.

---

## TL;DR results

| Strategy | What it optimizes for | Revenue vs. random targeting (20% budget) |
|---|---|---|
| Random targeting | nothing | baseline |
| **Naive churn-risk targeting** | highest churn probability | +modest |
| **Uplift targeting (this project)** | highest *incremental* retention | **+2.6% more revenue than churn-risk targeting**, and the gap widens at smaller budgets |

At tighter budgets (5–15% of the customer base — the realistic regime for most retention campaigns), uplift targeting retains meaningfully more customers than churn-risk targeting, because it stops wasting spend on customers who were either never going to leave or were never going to be swayed by an offer.

---

## Why this isn't just another churn classifier

| | Typical portfolio churn project | This project |
|---|---|---|
| Predicts | P(churn) | P(churn) **and** the causal effect of an intervention |
| Evaluated with | ROC-AUC | ROC-AUC **+ Qini coefficient** (the correct metric for uplift ranking) |
| Output | A risk score | A **targeting policy** with quantified ROI |
| Deployment | Notebook | FastAPI service **+ Streamlit dashboard + Docker + drift monitoring** |

---

## Architecture

```
                     ┌─────────────────────┐
                     │  Telco churn data    │  (Kaggle, ~7K customers)
                     │  + simulated RCT      │  (retention-offer rollout)
                     └──────────┬───────────┘
                                │
              ┌─────────────────┼─────────────────┐
              ▼                                     ▼
    ┌───────────────────┐               ┌────────────────────────┐
    │  Churn classifier   │               │  Uplift models           │
    │  (XGBoost)           │               │  T-Learner / X-Learner    │
    │  "who's at risk?"    │               │  "who can be saved?"      │
    └──────────┬──────────┘               └───────────┬──────────────┘
               │                                        │
               └───────────────────┬───────────────────┘
                                     ▼
                     ┌───────────────────────────┐
                     │  Business ROI simulator      │
                     │  budget-constrained targeting │
                     │  policy comparison (Random /   │
                     │  Churn-risk / Uplift)           │
                     └──────────────┬────────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              ▼                                                ▼
   ┌────────────────────┐                        ┌─────────────────────┐
   │  FastAPI service      │                        │  Streamlit dashboard   │
   │  /predict/churn        │                        │  ROI simulator, segment │
   │  /predict/uplift       │                        │  explorer, customer      │
   │  /predict/recommend    │                        │  scoring UI               │
   └────────────────────┘                        └─────────────────────┘
              │
              ▼
   ┌────────────────────┐
   │  Drift monitoring      │
   │  (Evidently / KS test)  │
   └────────────────────┘

   All services containerized via Docker Compose.
```

---

## The dataset problem, and how it's handled honestly

The public [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM/Kaggle) is **observational** — it records who churned, but no retention offer was ever actually run on this population, so there's no ground-truth treatment/control split to learn a *causal* effect from.

Real uplift modeling requires either:
1. A genuine randomized experiment (treatment vs. control), or
2. A well-instrumented quasi-experiment

Since neither exists publicly for this dataset, this project **layers a clearly-documented, fully-transparent simulated randomized retention-offer experiment** on top of the real customer features (`src/data_prep.py::add_synthetic_retention_experiment`). The simulation assigns each customer to one of four classic uplift-modeling archetypes based on their actual features (contract type, tenure, service add-ons):

- **Sure Things** (~20%) — long-tenure / two-year-contract customers who were staying regardless. Near-zero treatment effect. Offering them a discount is wasted spend.
- **Lost Causes** (~15%) — month-to-month, fiber-optic customers with no tech support or security add-ons. High churn risk **and** unresponsive to the offer — their dissatisfaction isn't about price.
- **Sleeping Dogs** (~8%) — otherwise-loyal customers for whom the offer slightly *backfires* (a well-documented real-world uplift phenomenon — drawing attention to price can prompt someone to reconsider their contract).
- **Persuadables** (~57%) — moderate churn risk, but a large positive treatment effect. These are the customers the offer genuinely saves, and the ones a pure churn-risk model tends to under-rank relative to Lost Causes.

This is a standard, transparent way to build an uplift-modeling portfolio project when no real experimental data is available — and it's designed so the divergence between "target by risk" and "target by uplift" is genuine and explainable, not manufactured noise. **If you have access to real experiment/promotion data, swap it in** — the modeling code doesn't care where `treatment` and `churn_observed` come from.

---

## Models

**Baseline churn classifier** — XGBoost, one-hot encoded categoricals, ROC-AUC ≈ 0.74–0.76 on held-out data. Exists mainly as the "naive" comparison point.

**T-Learner** — trains two independent XGBoost regressors (one on treated customers, one on control), predicted uplift = `P(churn | control) − P(churn | treated)`.

**X-Learner** — two-stage meta-learner: imputes per-unit treatment effects using the *opposite* arm's outcome model, fits a second-stage model on those imputed effects, and blends the two arms' second-stage predictions by propensity score. Generally lower-variance than the T-Learner, especially with imbalanced treatment groups.

Both are implemented directly on `scikit-learn` / `xgboost` rather than depending on the `causalml` package, which has heavy native-build requirements that frequently fail to install in constrained environments — these meta-learners are exactly what `causalml` implements under the hood, just made explicit and portable here.

**Evaluation:** the [Qini coefficient](https://www2.cs.duke.edu/courses/spring20/compsci590.1/uplift_modeling.pdf) — the uplift-modeling analogue of ROC-AUC — measures how well the model's ranking concentrates incremental retention in the top-ranked customers, versus random targeting.

---

## Business ROI simulation

`src/business_simulation.py` compares three targeting policies at several budget levels (5%–50% of the customer base offered the deal):

1. **Random** — no model
2. **Churn-risk ranking** — offer the highest-churn-probability customers (the "obvious" strategy most churn projects stop at)
3. **Uplift ranking** — offer the highest-predicted-incremental-retention customers

Because the simulated experiment has known ground-truth potential outcomes, the exact revenue impact of each policy is computable — letting the project state a concrete dollar (and percentage) improvement rather than only a model metric.

---

## Project structure

```
churn-uplift-roi/
├── src/
│   ├── data_prep.py          # loads Telco data, simulates the retention RCT
│   ├── features.py           # shared feature engineering
│   ├── churn_model.py        # baseline XGBoost churn classifier
│   ├── uplift_model.py       # T-Learner, X-Learner, Qini evaluation
│   └── business_simulation.py # budget-constrained ROI comparison
├── api/
│   └── main.py                # FastAPI serving layer
├── app/
│   └── streamlit_app.py       # business-facing ROI dashboard
├── monitoring/
│   └── drift_check.py         # Evidently data-drift report (+ scipy fallback)
├── tests/
│   └── test_uplift.py         # regression tests incl. sign-convention check
├── models/                    # trained artifacts (generated, gitignored)
├── data/                      # place telco_churn.csv here (optional)
├── train_all.py               # one-command training entrypoint
├── requirements.txt
├── Dockerfile.api
├── Dockerfile.app
└── docker-compose.yml
```

---

## Getting started

### 1. Install dependencies
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### 2. (Optional) Add the real dataset
Download **"Telco Customer Churn"** by blastchar from Kaggle and place it at:
```
data/telco_churn.csv
```
If this file isn't present, the pipeline auto-generates a synthetic dataset with an identical schema so everything runs out of the box — useful for development, but swap in the real file before treating results as representative.

### 3. Train everything
```bash
python train_all.py
```
This trains the churn classifier, both uplift models, prints Qini scores, and prints the business ROI summary table.

### 4. Run the tests
```bash
pytest tests/
```

### 5. Start the API
```bash
uvicorn api.main:app --reload
```
Visit `http://127.0.0.1:8000/docs` for interactive Swagger docs. Try `/predict/recommend` with a customer record.

### 6. Start the dashboard
```bash
streamlit run app/streamlit_app.py
```

### 7. Run everything in Docker
```bash
docker compose up --build
```
API → `http://localhost:8000` · Dashboard → `http://localhost:8501`

### 8. Check for data drift
```bash
python -m monitoring.drift_check
```
Generates `monitoring/drift_report.html` (or a console summary if `evidently` isn't installed).

---

## Honest limitations

- The uplift/causal component relies on **simulated** treatment assignment, clearly documented as such — it demonstrates the modeling and business-simulation methodology, not a real historical campaign's actual ROI. Swap in real experiment data to make the dollar figures real.
- The Telco dataset is a static snapshot (~7K customers); a production system would need continuous relabeling as tenure/outcomes accrue.
- Qini coefficients here are modest in absolute terms — realistic for a ~50/50 randomized 7K-row simulation with intentionally noisy individual treatment effects, not inflated to look artificially strong.

---

## Author

Shivarchan C · B.Tech CSE (Data Science), JAIN (Deemed-to-be University), Bengaluru
