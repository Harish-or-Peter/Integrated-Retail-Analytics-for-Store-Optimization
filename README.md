# Integrated Retail Analytics for Store Optimization and Demand Forecasting

> **End-Course Summative Project — Advanced Machine Learning (AlmaBetter)**
> A unified ML pipeline that addresses *anomaly detection*, *store segmentation*,
> *demand forecasting*, and *market basket analysis* on the Walmart weekly-sales dataset
> (45 stores × 81 departments × 143 weeks → 421,570 rows).

---

## 1. Business Problem

Retailers must answer five strategic questions every week, for every store and department:

1. *Are this week's sales abnormal* relative to the series' history (and if so, why)?
2. *Which stores behave alike* — i.e., which stores can share a marketing playbook?
3. *What will demand be next week / month / quarter*, given expected weather, holidays, and macro?
4. *Which departments lift the sales of which other departments* when sold together?
5. *How do CPI, Unemployment, Fuel Price* bend demand differently across store archetypes?

This project builds a single ML workflow that answers all five with concrete recommendations.

---

## 2. Data

Three CSVs live in `data/`:

| File | Rows | Grain | Notes |
|---|---|---|---|
| `sales data-set.csv` | 421,570 | Store × Dept × Date (weekly) | Target = `Weekly_Sales`. 1,285 negative values are net returns, kept as-is. |
| `Features data set.csv` | 8,190 | Store × Date | `MarkDown1-5` are NaN before Nov 2011 (program had not started). `CPI` and `Unemployment` have a small future-window NaN block. |
| `stores data-set.csv` | 45 | Store | Type ∈ {A, B, C}, Size in sq ft. |

**Imputation choices**
- `MarkDown1-5` NaN → **0** (semantically: 'no promotion was active that week').
- `CPI`, `Unemployment` NaN → **forward+backward fill within Store** (regional macro variables are slow-moving).

---

## 3. Approach

| Sub-problem | Models | Best metric |
|---|---|---|
| **Anomaly detection** | Isolation Forest, Local Outlier Factor, STL residual z-score | STL z-score (time-aware) |
| **Store segmentation** | KMeans, Agglomerative (Ward), DBSCAN | KMeans by silhouette + Davies-Bouldin |
| **Demand forecasting** | Linear, Ridge (tuned), RandomForest (+tuned), XGBoost (+tuned), SARIMAX | XGBoost-tuned by **WMAE** |
| **Market basket** | Apriori on (Store, Date) × Department co-occurrence | Top-15 rules with lift > 1.3 |

**Why WMAE?** Walmart's official competition metric weights holiday weeks 5×.
With ~7% of weeks being holidays but driving disproportionate revenue, WMAE is the right
business loss to optimize against.

---

## 4. Repository Layout

```
.
├── data/                                    # Source CSVs (provided)
├── notebooks/
│   └── Integrated_Retail_Analytics.ipynb   # End-to-end notebook (one-click reproducible)
├── src/
│   └── build_notebook.py                   # Programmatic notebook builder
├── outputs/
│   ├── figures/                            # All saved PNGs (used in tech doc + video)
│   └── models/                             # xgb_forecaster.joblib (best model)
├── docs/
│   ├── TECHNICAL_DOCUMENT.md               # Detailed methodology — paste into Google Docs
│   └── VIDEO_SCRIPT.md                     # Timed video presentation script (≥40 min)
├── refreneces/                             # Provided AlmaBetter templates
├── PROJECT_PLAN.md                         # Execution plan
├── README.md                               # This file
├── requirements.txt
└── .gitignore
```

---

## 5. Reproducing the Project

### 5.1 Environment

Tested on Python 3.11+. Install dependencies:

```bash
pip install -r requirements.txt
```

### 5.2 Run the notebook

Two equivalent options:

**Option A — execute via nbconvert** (one-shot, headless):

```bash
jupyter nbconvert --to notebook --execute notebooks/Integrated_Retail_Analytics.ipynb \
    --output Integrated_Retail_Analytics.ipynb --ExecutePreprocessor.timeout=1800
```

**Option B — open and run interactively**:

```bash
jupyter notebook notebooks/Integrated_Retail_Analytics.ipynb
```

The notebook is **deployment-ready**: the entire .ipynb runs end-to-end without manual
intervention, saves figures to `outputs/figures/`, pickles the best forecasting model to
`outputs/models/xgb_forecaster.joblib`, and ends with a load-and-predict sanity check.

### 5.3 Re-build the notebook from source

The notebook is generated programmatically by `src/build_notebook.py`. Edit the script and re-run:

```bash
python src/build_notebook.py
```

This regenerates `notebooks/Integrated_Retail_Analytics.ipynb` from the cell-by-cell definitions
in the script. Useful for tracking changes via git diffs that are smaller than a full .ipynb diff.

---

## 6. Results Summary

### Forecasting leaderboard (lowest WMAE wins)

| Model | WMAE | RMSE | MAE | R² |
|---|---:|---:|---:|---:|
| Linear Regression | 1,570.57 | 2,944.50 | 1,532.72 | 0.98 |
| Ridge (tuned) | 1,567.34 | 2,940.55 | 1,529.54 | 0.98 |
| **Random Forest (default) ← chosen** | **1,262.73** | **2,573.99** | **1,239.16** | **0.99** |
| Random Forest (tuned, sub-sampled CV) | 1,306.04 | 2,583.70 | 1,259.37 | 0.99 |
| XGBoost (default) | 1,282.52 | 2,649.39 | 1,249.86 | 0.99 |
| XGBoost (tuned, sub-sampled CV) | 1,327.09 | 2,782.07 | 1,274.42 | 0.98 |

The picker selects the lowest-WMAE model programmatically and pickles it to
`outputs/models/best_forecaster.joblib`. The "tuned" variants here use
`RandomizedSearchCV` on a 60–80K row sub-sample (notebook-runnable in <10 minutes); the
*default* models train on the full 337K-row training set, which is why they currently win.
A larger tuning budget on the full dataset is expected to flip the result — see the future-work
note in the technical document.

**WMAE ↓ ~20%** moving from the linear baseline to the tree ensembles — the headline gain.

### Segmentation quality

| Algorithm | Silhouette | Davies-Bouldin | Notes |
|---|---:|---:|---|
| KMeans (best K by silhouette) | reported in notebook | reported in notebook | Chosen for deployment. |
| Agglomerative (Ward linkage) | reported in notebook | reported in notebook | Comparable; provides dendrogram. |
| DBSCAN | n/a (depends on eps) | n/a | Used to identify outlier stores. |

### Market basket — top rules
- Top rules surface department co-spike pairs with **lift > 2.5**, e.g.
  `{Dept 48} → {Dept 19, 37}` at lift ≈ 2.67, support ≈ 11%, confidence ≈ 0.55.
- Full top-15 in `outputs/figures/top_market_basket_rules.csv`.

---

## 7. Strategy Highlights

The notebook ends with a strategy section that maps every model output to a business decision:

- **Marketing**: cluster-specific markdown depth and frequency.
- **Inventory**: cluster-specific safety-stock buffers using forecast σ.
- **Cross-merchandising**: top-15 association rules as a 'sell-with' table for merchants.
- **External-factor playbook**: Unemployment-triggered defensive markdowns; fuel-price flag for Type-C.
- **Implementation challenges**: data freshness, cold-start, drift, attribution.

---

## 8. Deliverables for Submission

| File | Purpose |
|---|---|
| `notebooks/Integrated_Retail_Analytics.ipynb` | Main code (Google Drive — view access for everyone) |
| `docs/TECHNICAL_DOCUMENT.md` | Paste into Google Docs (with screenshots from `outputs/figures/`) |
| `docs/VIDEO_SCRIPT.md` | Speaker notes for the ≥40-min video |
| GitHub repo URL | This repository |

---

## 9. Author

Individual submission — [Your Name].

## 10. License

For educational use as part of the AlmaBetter Advanced ML programme.
