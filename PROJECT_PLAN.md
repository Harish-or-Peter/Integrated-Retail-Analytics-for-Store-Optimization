# Project Execution Plan — Integrated Retail Analytics

**Project Title:** Integrated Retail Analytics for Store Optimization and Demand Forecasting
**Course:** AlmaBetter — Advanced Machine Learning End Course Summative Assignment
**Mode:** Individual submission

---

## Dataset Confirmed

Walmart-style retail dataset, three CSVs in `data/`:

| File | Rows | Description |
|---|---|---|
| `sales data-set.csv` | 421,570 | Weekly sales per Store × Dept × Date |
| `Features data set.csv` | 8,190 | Per-store weekly features: Temperature, Fuel_Price, MarkDown1-5, CPI, Unemployment, IsHoliday |
| `stores data-set.csv` | 45 | Store metadata: Type (A/B/C), Size |

References folder contains the AlmaBetter standard EDA + ML notebook templates (15 charts, 3 hypothesis tests, 3 ML models, UBM rule).

---

## Deliverables

1. **Jupyter Notebook** — main code, follows AlmaBetter template, all 10 project components
2. **GitHub Repository** — unified repo with README, code, data, models, visualizations
3. **Technical Document** — Google Docs ready, methodology + insights + screenshots
4. **Video Presentation Script** — ≥40 minutes, 8 sections per rubric

---

## Project Structure

```
c:\AB_projects\
├── data/                                    # Source CSVs (provided)
├── refreneces/                              # AlmaBetter templates (provided)
├── notebooks/
│   └── Integrated_Retail_Analytics.ipynb   # Main deliverable
├── src/                                     # Optional modular helpers
├── outputs/
│   ├── figures/                            # Saved PNGs for docs/video
│   └── models/                             # Pickled best models
├── docs/
│   ├── TECHNICAL_DOCUMENT.md               # Google Docs source
│   └── VIDEO_SCRIPT.md                     # Timed presentation script
├── PROJECT_PLAN.md                         # This file
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Execution Phases

### Phase 1 — Notebook (Core, ~80% of work)

Single notebook covering all 10 project components, mapped onto the AlmaBetter template:

- **§1-3 Know data → wrangling**: load 3 CSVs, merge on Store+Date, sanity checks
- **§4 Visualizations (15+, UBM-structured)**: weekly sales over time, store-type box plots, holiday spikes, markdown coverage, CPI/unemployment/fuel scatter, dept heatmap, seasonality decomposition, correlation heatmap, pair plot
- **§5 Hypothesis tests** (3):
  - Holiday weeks vs non-holiday sales (t-test)
  - Store Type A vs B vs C mean sales (ANOVA)
  - Markdown-active weeks vs inactive (Mann-Whitney)
- **§6 Feature engineering**: markdown imputation (zeros = no markdown), lag features (1, 4, 52 weeks), rolling means, week-of-year, month, holiday flag expansions, store size bins, type one-hot
- **§7 Modeling — four sub-problems integrated:**
  - **Anomaly detection**: IsolationForest, LocalOutlierFactor, STL residual z-score; compare on labeled holiday weeks as proxy
  - **Segmentation**: KMeans + Hierarchical + DBSCAN on store-level feature vectors; silhouette + Davies-Bouldin
  - **Forecasting**: naive seasonal baseline → Linear Regression → Random Forest → XGBoost; SARIMAX/Prophet on representative series; metrics: WMAE, RMSE, MAPE; `RandomizedSearchCV` tuning
  - **Market basket**: Apriori on department co-occurrence per (Store, Date) basket
- **Strategy section** + pickled best model + load-and-predict sanity check + conclusion

### Phase 2 — GitHub Repo

- README.md (problem, data, structure, how to run, results table, screenshots)
- requirements.txt with pinned versions
- .gitignore for Python + data
- `git init` + initial commit
- **Pause before remote push** — needs user GitHub URL + confirmation

### Phase 3 — Technical Document (`docs/TECHNICAL_DOCUMENT.md`)

Mirrors notebook structure, written for Google Doc — methodology, decisions, figures referenced from `outputs/figures/`. Pasteable into Google Docs with PNG screenshots.

### Phase 4 — Video Script (`docs/VIDEO_SCRIPT.md`)

- 8 sections matching the rubric, timed to ~42 min (above the 40-min minimum)
- Speaker notes per section + slide-by-slide bullets + "what the interviewer is checking" callouts + answers to all 5 follow-up questions

---

## Key Technical Decisions

1. **MarkDown NaNs treated as zeros** — markdowns started Nov 2011, so NaN = no promotion was active. Documented as a deliberate choice, not lazy imputation.
2. **WMAE as primary forecasting metric** (5× weight on holiday weeks) — this is the official Walmart competition metric, shows domain awareness alongside RMSE/MAPE.
3. **Forecasting scope**: top-N high-volume Store×Dept pairs + an aggregate model. Full grid (3000+ series) won't run end-to-end in reasonable time within a notebook.
4. **Market basket limitation explicitly called out**: no transaction-level data, so basket = (Store, Date) and items = departments with sales > threshold. Documented as a constraint of the dataset.

---

## Evaluation Rubric Alignment

| Section | Weight | Where addressed in notebook |
|---|---|---|
| Data Analysis & Preprocessing | 20% | §1-3, §6 |
| ML Modeling & Techniques | 30% | §7 (anomaly, segmentation, forecasting) |
| Market Basket + Segmentation | 15% | §7b, §7d |
| External Factors Integration | 10% | §6 (features) + §7c (forecasting inputs) |
| Strategy & Real-World Application | 10% | Strategy section + Conclusion |
| Code Quality & Documentation | 10% | Throughout, modular cells, comments |
| Presentation & Reporting | 5% | Video script + Tech Doc |

---

## Checkpoints with User

1. **Before GitHub push** — get repo URL or confirm "create new"
2. **Before final wrap** — review of the notebook output before freeze
