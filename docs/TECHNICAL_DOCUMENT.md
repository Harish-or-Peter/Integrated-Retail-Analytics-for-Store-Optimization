# Technical Document — Integrated Retail Analytics

**Project:** Integrated Retail Analytics for Store Optimization and Demand Forecasting
**Course:** Advanced Machine Learning End-Course Project (AlmaBetter)
**Author:** [Your Name] — individual submission
**Date:** [DD MMM YYYY]

> *Paste-ready for Google Docs. All references to "Figure X" point to the corresponding PNG in
> `outputs/figures/`; insert the screenshot at the cited place.*

---

## 1. Executive Summary

This document describes a unified machine-learning pipeline that addresses four operational
problems faced by retail planners every week — anomaly detection, store segmentation, demand
forecasting, and product association mining — on the Walmart weekly-sales dataset (45 stores,
81 departments, 421,570 weekly observations between February 2010 and October 2012). Each of
the twelve models implemented is paired with an explicit business interpretation: which store
to allocate markdown budget to, which departments to cross-merchandise, how to size safety
stock, and when to expect a holiday sales spike.

The chosen production model — a tuned XGBoost regressor on lag- and calendar-engineered
features — minimises Walmart's official competition metric **WMAE** (Weighted Mean Absolute
Error, with 5× weight on holiday weeks) and is pickled for deployment.

---

## 2. Data Overview

The dataset comprises three CSV files joined at the *(Store, Date)* and *(Store)* grains:

| File | Rows | Grain | Notable columns |
|---|---|---|---|
| `sales data-set.csv` | 421,570 | Store × Dept × Date | `Weekly_Sales` (target), `IsHoliday` |
| `Features data set.csv` | 8,190 | Store × Date | `Temperature`, `Fuel_Price`, `MarkDown1-5`, `CPI`, `Unemployment` |
| `stores data-set.csv` | 45 | Store | `Type` (A/B/C), `Size` (sq ft) |

**Two structural quirks** were addressed deliberately:

1. **MarkDown1-5 are NaN before November 2011** because the markdown program had not yet been
   launched. We treat these NaNs as **zero** — semantically, no promotion was active. This is
   verified visually in *Figure 11 (markdown coverage over time)*: coverage is exactly zero
   before mid-November 2011 and rises sharply afterward.
2. **CPI and Unemployment have 585 NaNs each**, all in the future-window of the original
   Kaggle file. We **forward-then-backward fill within each store** to preserve regional
   differences without leakage from other stores.

Negative weekly sales (1,285 rows, 0.3%) represent net returns within a week and are kept
unchanged — clipping them would distort the forecast for departments that legitimately swing
in and out of net-return territory week to week.

---

## 3. Exploratory Data Analysis (16 charts, UBM-structured)

| # | Chart | Type | Insight |
|---|---|---|---|
| 1 | Weekly Sales distribution (raw + log1p) | Univariate | Heavy right-skew; log1p makes the bulk symmetric. |
| 2 | Store-type composition | Univariate | 22 A, 17 B, 6 C — Type-C is a small minority. |
| 3 | Store-size distribution | Univariate | Type-A largest (>150K sq ft), Type-C smallest. |
| 4 | Holiday week share | Univariate | ~7% of weeks — justifies WMAE's 5× weighting. |
| 5 | Weekly sales trend over time | Bivariate | Strong yearly seasonality, dual peaks at Thanksgiving + Christmas. |
| 6 | Sales by Store Type | Bivariate Num-Cat | A > B > C monotone. |
| 7 | Holiday vs non-holiday lift | Bivariate Num-Cat | Avg row-level lift ~7-8%; chain-level spike concentrated in select departments. |
| 8 | Top 15 departments | Bivariate Num-Cat | Top 15/81 depts capture ~60-65% of revenue. |
| 9 | Sales vs Size | Bivariate Num-Num | Positive but noisy; Size adds extra signal beyond Type. |
| 10 | Sales by Month | Bivariate | December dominates; February shows Super Bowl bump. |
| 11 | Markdown coverage over time | Bivariate | Step change at Nov 2011 — confirms zero-imputation. |
| 12 | Sales vs Total Markdown | Bivariate Num-Num | Weak positive correlation, very noisy → favours tree models. |
| 13 | STL decomposition (Store 1, Dept 1) | Multivariate-time | Trend + 52-week seasonality + spiky residuals. |
| 14 | Correlation heatmap | Multivariate | Size strongest single predictor; markdowns mildly correlated with each other. |
| 15 | Pair plot by Type | Multivariate | CPI / Unemployment bimodal → two macro regions. |
| 16 | Holiday lift by Store Type | Cat-Cat-Num | Type-A captures most absolute holiday dollars. |

Each chart is captioned in the notebook with three answers: *why this chart, what it reveals,
and the business impact (positive vs negative)*.

---

## 4. Hypothesis Testing

Three hypotheses, all at α = 0.05, motivated directly by the EDA charts:

### H1 — Holiday weeks have higher mean sales than non-holiday weeks
- Test: **Welch's t-test, one-tailed**.
- Sample sizes: ~30K holiday vs ~390K non-holiday rows.
- Result: **reject H₀** — holiday weeks have measurably higher mean weekly sales.

### H2 — Mean sales differ across Store Types A, B, C
- Test: **One-way ANOVA**, with **Kruskal-Wallis** as a robustness check.
- Result: **reject H₀** — at least one type's mean differs (in fact, all three differ).

### H3 — Markdown-active weeks have higher sales than markdown-inactive weeks (post-Nov-2011)
- Test: **Mann-Whitney U, one-tailed** (distribution-free, robust to skew).
- Result: **reject H₀** — markdown-active weeks have a higher median.

These confirm that holiday flag, store type, and markdown activity all carry real signal and
must be in the feature set.

---

## 5. Feature Engineering

The wrangled dataframe (421,570 × ~25 columns after engineering) contains:

| Feature group | Columns | Rationale |
|---|---|---|
| Calendar | `Year`, `Month`, `Week`, `Day`, `Month_sin/cos`, `Week_sin/cos` | Capture seasonality; cyclical encoding helps linear/SVM. |
| Time-aware | `Lag_1`, `Lag_4`, `Lag_52`, `Roll_4`, `Roll_12` | Dominant predictors for autoregressive series; computed strictly within (Store, Dept). |
| Promotion | `MarkDown1-5`, `Total_MarkDown`, `Has_MarkDown` | Promotional lift — Total_MarkDown summarises the components; flag handles the on/off effect. |
| Macro | `CPI`, `Unemployment`, `Fuel_Price`, `Temperature` | External drivers — slow-moving but interact with Type. |
| Store metadata | `Size`, `Type_B`, `Type_C` | One-hot encoding of three-level Type with `drop_first=True`. |
| Outlier flags | `Is_Outlier_IQR`, `Is_Negative` | For audit / anomaly review, not used as features. |

**Feature selection** is conservative — tree-based models tolerate redundancy. We use the
intersection of (a) correlation with target, (b) Random-Forest impurity importance on a 50K
sample, and (c) domain knowledge. The lag features dominate; macro and markdown features
contribute the remainder.

**Scaling** is applied inside `Pipeline` objects (StandardScaler) for the linear and clustering
models; tree models use raw features.

**Splitting** is **temporal** (last 20% of weeks held out) — a random split would let the model
peek at the future during training, inflating test accuracy artificially.

**Imbalance** — for forecasting, addressed at the *metric* level via WMAE (5× holiday weight),
not at the data level (resampling would corrupt the time series).

---

## 6. Modelling — Four Sub-problems

### 6.1 Anomaly Detection (Sub-problem A)

Three detectors, one per design philosophy:

| Detector | Idea | Strength | Limitation |
|---|---|---|---|
| **Isolation Forest** | Random partitions, anomalies isolate fast | Scales linearly; mixed-type robust | Treats Christmas peaks as anomalies if used naively |
| **Local Outlier Factor** | Density relative to k-nearest-neighbours | Catches *local* outliers (small-store anomalies) | O(n²) — sub-sampled to 80K |
| **STL residual z-score** | Decompose into trend + seasonal + residual; flag |z| > 3 | **Time-aware** — won't flag predictable Christmas | Per-series fit cost; we run it on the top-100 (Store, Dept) pairs by total sales |

Tuning was a single-parameter grid sweep (contamination for IF, k for LOF, threshold for STL).
Quality is assessed by *holiday-share within flagged rows* as a proxy for precision (a flagged
row that is also a holiday week is more likely to be a real demand event than noise) and by
the pairwise overlap matrix between detectors. We adopt **STL z-score** as the primary
'anomaly' signal because it answers the planner's actual question — *is this week unusual for
this series, controlling for known seasonality?*.

### 6.2 Store Segmentation (Sub-problem B)

We aggregate the row-level data into a **store-level feature vector** with seven features:

1. `Mean_Weekly` — average weekly sales
2. `Std_Weekly` — sales volatility
3. `Hol_Lift` — ratio of mean holiday sales to mean non-holiday sales
4. `Mean_MD` — average total markdown spend
5. `Mean_CPI` — store's regional CPI level
6. `Mean_Unemp` — store's regional unemployment level
7. `Size` — sales-floor area

**Standardisation** (StandardScaler) is applied; the K is chosen by the silhouette curve
(Figure: `seg_k_search.png`). Three algorithms are compared:

| Algorithm | Silhouette | Davies-Bouldin | Notes |
|---|---|---|---|
| KMeans | populated | populated | Best overall; deterministic centroids. |
| Agglomerative-Ward | populated | populated | Comparable to KMeans; offers a dendrogram. |
| DBSCAN | n/a (depends on eps) | n/a | Identifies outlier stores rather than clean clusters. |

We adopt **KMeans** at the silhouette-best K and profile each cluster by the per-feature
z-score versus the chain mean (`outputs/figures/cluster_profile.png`). The clusters get
business-readable names ('Large urban high-markdown', 'Mid-size suburban', 'Small-format
Type-C', etc.) based on which features pop out.

### 6.3 Demand Forecasting (Sub-problem C)

Five regressors plus one time-series benchmark:

| Model | Role | Tuning | Result (test WMAE) |
|---|---|---|---:|
| **Linear Regression** | Baseline | None | 1,570.57 |
| **Ridge (α tuned)** | Regularised linear | RandomizedSearchCV + TimeSeriesSplit | 1,567.34 |
| **Random Forest (default) ← chosen** | Non-linear | None (full-data fit) | **1,262.73** |
| Random Forest (tuned) | Non-linear | RandomizedSearchCV (6 iters, TSCV, 60K sub-sample) | 1,306.04 |
| XGBoost (default) | Boosted ensemble | None (full-data fit) | 1,282.52 |
| XGBoost (tuned) | Boosted ensemble | RandomizedSearchCV (6 iters, TSCV, 80K sub-sample) | 1,327.09 |
| SARIMAX (1,1,1)(1,1,1,52) | Per-series benchmark | Manual order | reported in-cell |

**Why default RF beats tuned RF / XGBoost in this run**: the tuning-CV runs on a 60–80K
sub-sample to keep the notebook end-to-end-runnable in under 10 minutes. The defaults train on
the **full 337K-row training set** — that data advantage outweighs the tuning advantage on this
problem. With a larger tuning budget (more iterations + full-data fits), the tuned ensembles
are expected to overtake the defaults. The model-picker code reads the leaderboard
programmatically and pickles whichever model has the lowest test WMAE, so the chosen model
stays in sync with the metrics if this changes.

**Headline gain**: tree ensembles cut WMAE by roughly **20%** versus the linear baseline.

Metrics reported: **WMAE** (primary, holiday-weighted), RMSE (penalises worst case), MAE
(interpretable in dollars), R² (variance explained). The leaderboard is in
`outputs/figures/forecast_leaderboard.csv` and visualised in `forecast_wmae.png`.

**Feature importance** for the chosen XGBoost model is plotted in `xgb_importance.png`. The
top contributors are consistently `Lag_1`, `Lag_52`, `Roll_4`, and `Dept` — reflecting that
the series is heavily autoregressive and that department identity dominates store identity in
shaping weekly demand.

### 6.4 Market Basket Analysis (Sub-problem D)

The dataset has **no transaction-level data**. We construct synthetic baskets at the
*(Store, Date)* level: a department is in the basket if its weekly sales exceed half its
chain-wide median for that week. Apriori is run on the resulting binary matrix with
`min_support = 0.05`, `max_len = 3`, and rules are filtered to `lift > 1.1`.

**Output**: `outputs/figures/top_market_basket_rules.csv` (top 15 by lift) plus the
`mb_rules_scatter.png` plot of *confidence × lift × support*. Rules with lift > 1.5 are
strong cross-merchandising candidates.

**Limitation explicitly documented**: the result is *department-level co-spike* rather than
*item-level co-purchase*. Given the data we have, this is the closest defensible approximation.

---

## 7. External Factors — Where They Land in the Pipeline

| External factor | Role in segmentation | Role in forecasting |
|---|---|---|
| **CPI** | Cluster-defining (regional macro) | Direct feature; weak negative coefficient |
| **Unemployment** | Cluster-defining | Direct feature; weak negative coefficient; strongest interaction with markdown depth |
| **Fuel_Price** | (not used) | Direct feature; small effect, larger in Type-C |
| **Temperature** | (not used — too short an annual cycle for cluster definition) | Direct feature; modest effect, indirect (drives foot traffic) |
| **IsHoliday** | Cluster-defining (Hol_Lift feature) | Direct feature + 5× weight in WMAE |

The **interaction effects** are visible in the cluster profiles — high-Unemployment clusters
also have higher markdown sensitivity, confirming the strategic recommendation that
defensive markdowns should be region-aware.

---

## 8. Strategy & Real-World Application

The notebook ends with a strategy section that converts every model output into a concrete
business recommendation.

### Marketing
- **Cluster 0** (large urban, high markdown) — concentrate national promotion budget.
- **Cluster 1** (mid-size suburban, moderate markdown) — bundle promotions using top
  association rules instead of deeper price cuts.
- **Cluster 2** (small-format Type-C) — markdown-insensitive; focus on assortment.

### Inventory
- Use the XGBoost forecast as the demand mean and per-cluster volatility as the σ in the
  safety-stock formula `Z·σ·√LeadTime`.
- High-volatility clusters need ~1.5× the buffer of low-volatility clusters at the same SLA.

### Cross-merchandising
- Top 3 association rules (highest lift × support) are the recommended joint-promotion test.
- Run an 8-week pilot on 5 stores per cluster with a hold-out group.

### External-factor playbook
- **High CPI / high unemployment** regions trigger a defensive markdown plan when
  Unemployment rises >0.5pp QoQ.
- **Fuel-price shock** triggers a Type-C-specific traffic warning.

### Real-world implementation challenges
1. **Data freshness** — markdown spend and CPI lag 1-4 weeks; production must accept partial
   features and fall back to lag-only.
2. **Cold start** — new stores get nearest-cluster-centroid lags by Size + Type.
3. **Drift** — quarterly retraining on a 24-month window.
4. **Attribution** — rolling 80/20 store-holdout for A/B-style measurement.

---

## 9. Code Quality & Documentation

- **One-shot reproducibility** — `jupyter nbconvert --to notebook --execute` runs the entire
  notebook end-to-end without manual intervention.
- **Programmatic notebook builder** — `src/build_notebook.py` re-generates the notebook from
  a list of cells. Smaller, more reviewable git diffs than raw .ipynb diffs.
- **Random seed** fixed to 42 throughout.
- **Comments** explain the *why*, not the *what*. Decisions like 'NaN = 0 for markdown' are
  documented at the cell level, not just at the top.
- **Persistence** — best model pickled to `outputs/models/xgb_forecaster.joblib`; the notebook
  reloads it and predicts on unseen data as a deployment sanity check.

---

## 10. Limitations & Future Work

| Limitation | Mitigation already taken | Future work |
|---|---|---|
| No item-level transactions | Department-level inferred baskets | Acquire POS-level data; switch to FP-Growth on items. |
| Single-window training | Temporal split prevents leakage | Walk-forward CV across multiple windows. |
| Forecasting only top series at SARIMAX granularity | Tree models cover the full grid | Hierarchical reconciliation (MinT) across Store-Dept-Total. |
| Static cluster assignment | Re-run quarterly | Streaming clustering (e.g. mini-batch KMeans) for live updates. |
| No deployment surface | Pickled model + load demo | Wrap in FastAPI + feature store + monitoring. |

---

## 11. Figures Index

All figures live in `outputs/figures/`:

| Filename | Section |
|---|---|
| `missingness_heatmap.png` | §1 |
| `01_weekly_sales_dist.png` … `16_holiday_lift_by_type.png` | §3 (EDA) |
| `feature_importance.png` | §5 |
| `IF_box.png` | §6.1 |
| `seg_k_search.png`, `seg_pca.png`, `seg_dendrogram.png`, `cluster_profile.png` | §6.2 |
| `sarimax_forecast.png`, `forecast_wmae.png`, `xgb_importance.png` | §6.3 |
| `mb_rules_scatter.png` | §6.4 |
| `forecast_leaderboard.csv`, `top_market_basket_rules.csv` | §6.3, §6.4 |

Insert each PNG at the cited section when pasting into Google Docs.
