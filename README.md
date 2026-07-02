# Customer Churn → Uplift Modeling → Retention ROI Simulator

**🔗 Live demo:** [churn-uplift-roi.streamlit.app](https://churn-uplift-roi.streamlit.app/)

**Business problem:** Retention teams have a fixed budget and a churn model that ranks customers by risk. The standard playbook — offer discounts to the highest-risk customers — routinely wastes a large share of that budget, because *high risk of churning* and *responsive to an offer* are not the same thing. This project builds the model that tells the difference, and quantifies exactly how much that difference is worth in revenue.

---

## Results (trained on the real Kaggle Telco Customer Churn dataset, n=7,043)

**Baseline churn classifier (XGBoost)**

| Metric | Value |
|---|---|
| ROC-AUC (held-out test set) | **0.786** |
| Precision / Recall (churn class) | 0.71 / 0.44 |
| Accuracy | 0.76 |

**Uplift models — T-Learner vs. X-Learner**

| Model | Qini coefficient | Correlation with ground-truth ITE* |
|---|---|---|
| T-Learner | 0.0098 | 0.629 |
| X-Learner | 0.0095 | **0.677** |

*\*Ground-truth individual treatment effect is only knowable because the treatment/outcome layer is a documented simulation on top of real customer features — see [Methodology](#methodology-and-honest-limitations) below. This correlation is the validation signal a real deployment would not have access to.*

**Business impact — uplift-based targeting vs. naive churn-risk targeting**

| Budget (% of customers offered) | Revenue: churn-risk targeting | Revenue: random targeting | Uplift targeting vs. churn-risk | Uplift targeting vs. random |
|---|---|---|---|---|
| 5% | $5,187,951 | $5,229,181 | **+$91,557** | +0.96% |
| 10% | $5,177,985 | $5,260,226 | **+$181,436** | +1.89% |
| 15% | $5,169,255 | $5,295,190 | **+$272,158** | +2.76% |
| 20% | $5,179,351 | $5,331,298 | **+$342,929** | +3.58% |
| 30% | $5,337,007 | $5,400,659 | **+$326,529** | +4.87% |
| 50% | $5,668,255 | $5,534,085 | **+$169,075** | +5.48% |

**Reading this table the way a retention-ops stakeholder would:** at a realistic campaign budget (10–20% of the customer base), switching from "offer the highest-risk customers" to "offer the customers the offer will actually save" is worth **$180K–$343K in additional projected revenue** on this customer base, before spending a single extra dollar on the campaign. The gap is largest at tight budgets — exactly where targeting precision matters most — and narrows as budget grows toward covering the whole base, which is the expected, sanity-checking shape for this kind of curve.

---

## Why this isn't just another churn classifier

| | Typical churn portfolio project | This project |
|---|---|---|
| Predicts | P(churn) | P(churn) **and** the causal effect of a specific intervention |
| Evaluated with | ROC-AUC only | ROC-AUC **+ Qini coefficient** (the correct metric for ranking-by-incremental-effect) |
| Output | A risk score | A **targeting policy** with a quantified dollar impact |
| Deployment | Notebook | FastAPI service + Streamlit dashboard + Docker + drift monitoring |

A churn-risk model answers "who is likely to leave." That's necessary but insufficient for a targeting decision — it conflates two very different customers: someone who will churn *no matter what you do* (a wasted offer), and someone whose decision genuinely hinges on the offer (the one worth spending on). **Uplift modeling separates these**, which is why it's standard practice at companies running retention/marketing spend at scale, and why it's a meaningfully harder and more interview-relevant skill than a plain classifier.

---

## Architecture

```
                     ┌─────────────────────┐
                     │  Telco churn data     │  Kaggle, 7,043 customers
                     │  + simulated retention │  randomized offer rollout
                     │    offer experiment     │  (documented simulation)
                     └──────────┬───────────┘
                                │
              ┌─────────────────┼─────────────────┐
              ▼                                     ▼
    ┌───────────────────┐               ┌────────────────────────┐
    │  Churn classifier   │               │  Uplift models            │
    │  XGBoost              │               │  T-Learner / X-Learner    │
    │  "who's at risk?"     │               │  "who can be saved?"      │
    └──────────┬──────────┘               └───────────┬──────────────┘
               │                                        │
               └───────────────────┬───────────────────┘
                                     ▼
                     ┌───────────────────────────┐
                     │  Business ROI simulator       │
                     │  budget-constrained policy      │
                     │  comparison: Random / Churn-risk │
                     │  / Uplift targeting                │
                     └──────────────┬────────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              ▼                                                ▼
   ┌────────────────────┐                        ┌─────────────────────┐
   │  FastAPI service       │                        │  Streamlit dashboard   │
   │  /predict/churn         │                        │  ROI simulator, segment │
   │  /predict/uplift        │                        │  explorer, live scoring │
   │  /predict/recommend     │                        │  UI                       │
   └────────────────────┘                        └─────────────────────┘
              │
              ▼
   ┌────────────────────┐
   │  Drift monitoring       │
   │  Evidently / KS test     │
   └────────────────────┘

   Both services containerized independently (Dockerfile.api, Dockerfile.app)
   and orchestrated via docker-compose.
```

---

## Methodology and honest limitations

**The dataset problem.** The public [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM/Kaggle) is observational — it records who churned, but no retention offer was ever actually run on this population. There is no real treatment/control split to learn a causal effect from, and no public Telco dataset that has one.

**The fix, applied transparently.** `src/data_prep.py::add_synthetic_retention_experiment` layers a documented, randomized 50/50 retention-offer simulation on top of the real customer features. Each customer is assigned to one of four canonical uplift-modeling archetypes *based on their actual contract/tenure/service features*, not arbitrary noise:

| Segment | Share | Baseline churn risk | Effect of the offer |
|---|---|---|---|
| **Sure Things** | ~20% | Low | ~0 — staying regardless; offering them is wasted spend |
| **Lost Causes** | ~15% | High | ~0 — churning regardless; the highest-risk group a naive model over-targets |
| **Sleeping Dogs** | ~8% | Low–moderate | Negative — offer slightly *increases* churn risk (a documented real-world phenomenon) |
| **Persuadables** | ~57% | Moderate | Strongly positive — the group the offer actually saves |

This is a standard, disclosed technique for building an uplift-modeling portfolio project without access to real experimental data — it exists to demonstrate the modeling and business-simulation methodology correctly, and the code is written so that swapping in a real `treatment` / `outcome` column from an actual campaign requires no changes to the modeling logic.

**What's real vs. simulated in the numbers above:** the customer features, the churn classifier, and its 0.786 AUC are trained entirely on real data. The uplift/ROI numbers depend on the simulated treatment layer described above — real in mechanism and methodology, illustrative rather than historical in the specific dollar figures.

**Other limitations:**
- Static ~7K-row snapshot; a production system would need continuous relabeling as outcomes accrue over time.
- Qini coefficients here are modest in absolute terms — realistic for a randomized 50/50 simulation with deliberately noisy individual-level effects, not inflated to look artificially strong.

---

## Models

**Baseline churn classifier** — XGBoost on one-hot encoded features. Exists as the "naive" comparison point the whole project is built to outperform on targeting decisions, not just on AUC.

**T-Learner** — two independent XGBoost regressors (treated arm, control arm); uplift = `P(churn | control) − P(churn | treated)`.

**X-Learner** — two-stage meta-learner: imputes per-unit treatment effects using the *opposite* arm's outcome model, fits a second-stage model on the imputed effects, and blends both arms' second-stage predictions by propensity score. Lower variance than the T-Learner, particularly useful with imbalanced treatment/control groups.

Both are implemented directly on `scikit-learn`/`xgboost` rather than the `causalml` package, which has heavy native-build requirements that frequently fail outside curated environments — these meta-learners are exactly what `causalml` implements internally, made explicit and portable here.

**Evaluation metric:** the [Qini coefficient](https://www2.cs.duke.edu/courses/spring20/compsci590.1/uplift_modeling.pdf), the uplift-modeling analogue of ROC-AUC — it measures how well a ranking concentrates incremental (not just correlated) positive outcomes in the top-ranked customers versus random targeting.

---

## Project structure

```
churn-uplift-roi/
├── src/
│   ├── data_prep.py            # loads Telco data, builds the retention-offer simulation
│   ├── features.py             # shared feature engineering
│   ├── churn_model.py          # baseline XGBoost churn classifier
│   ├── uplift_model.py         # T-Learner, X-Learner, Qini evaluation
│   └── business_simulation.py  # budget-constrained ROI comparison
├── api/
│   └── main.py                 # FastAPI serving layer
├── app/
│   └── streamlit_app.py        # business-facing ROI dashboard
├── monitoring/
│   └── drift_check.py          # Evidently drift report (+ scipy fallback)
├── tests/
│   └── test_uplift.py          # regression tests, incl. uplift sign-convention check
├── models/                     # trained artifacts
├── data/                       # telco_churn.csv goes here
├── train_all.py                # one-command training entrypoint
├── requirements.txt
├── Dockerfile.api
├── Dockerfile.app
└── docker-compose.yml
```

---

## Getting started

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

Download [**Telco Customer Churn**](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (blastchar, Kaggle) and place it at `data/telco_churn.csv`. If absent, the pipeline auto-generates a schema-matching synthetic dataset so the project still runs end-to-end — useful for development, but the results above are from the real file.

```bash
python train_all.py          # trains churn model + both uplift models, prints ROI summary
pytest tests/                # 4 tests, incl. sign-convention regression check
uvicorn api.main:app --reload           # API → http://127.0.0.1:8000/docs
streamlit run app/streamlit_app.py      # dashboard → http://localhost:8501
docker compose up --build               # both services, containerized
python -m monitoring.drift_check        # data drift report
```

---

## Author

Shivarchan C · B.Tech CSE (Data Science), JAIN (Deemed-to-be University), Bengaluru
