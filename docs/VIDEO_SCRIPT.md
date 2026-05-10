# Video Presentation Script — Integrated Retail Analytics

**Total runtime target: ~42 minutes** (above the AlmaBetter checklist minimum of 40 min;
explanation portion ≥15 min as required).

**Format**: screen-share over the executed notebook + a small set of slides for the
introduction and the strategy section. Speak in a steady, conversational tone — slide-by-slide
bullets below are *speaker notes*, not slide text.

**Recording tips before you press record**
- Open the notebook in Colab/Jupyter with all cells already executed (don't re-run during the recording).
- Keep `outputs/figures/` open in a side window — you'll switch to specific PNGs at chart time.
- Have the strategy slides (5 slides) ready to switch to at section 6.
- Keep this script open on a second monitor or printed out.

---

## Section 0 — Title + Welcome (0:00 – 0:30)

> Hi, I'm [Your Name]. This is my End-Course Summative Project for the Advanced Machine
> Learning module — *Integrated Retail Analytics for Store Optimization and Demand
> Forecasting*. Over the next ~42 minutes I'll walk you through the dataset, the four
> sub-problems I solved, the modelling decisions, the business strategy that came out, and
> the deployment-ready code.

---

## Section 1 — Introduction (0:30 – 2:00, **1.5 min**)

**What to say**:

> This project is a multi-component retail analytics pipeline built on the well-known Walmart
> weekly-sales dataset — 45 stores, 81 departments, 421,570 weekly observations between
> February 2010 and October 2012. The main objective is twofold: *store optimization* and
> *demand forecasting*. The business problem behind these numbers is concrete — a retail
> planner has to answer five questions every single week, for every single (Store, Dept):
>
> 1. Are this week's sales abnormal?
> 2. Which stores can share a marketing playbook?
> 3. What will demand be next week, next month, next quarter?
> 4. Which products lift the sales of which others when sold together?
> 5. How do macroeconomic factors bend the demand curve differently across store archetypes?
>
> I address each with a dedicated ML technique — **anomaly detection**, **clustering**,
> **time-series forecasting**, and **market basket analysis** — and I tie every model output
> to a concrete business decision in the strategy section.

**Interviewer is checking**: business framing first, technique second; awareness this is a
multi-part problem, not a single-model exercise.

---

## Section 2 — Problem Understanding (2:00 – 4:00, **2 min**)

**What to say**:

> Why do store optimization and demand forecasting matter? Because retail margins are razor-thin
> and the cost of being wrong is asymmetric. Under-stock and you lose sales *and* customers; over-stock
> and you take markdowns or write-offs. The challenge is that demand at any given (Store, Dept)
> is shaped by **at least seven** things simultaneously — store size and format, regional macro
> (CPI, unemployment, fuel), weather, holidays, the timing of promotional markdowns, the autocorrelation
> of the series itself, and cross-product effects. Traditional rule-based replenishment can't balance
> all of these, especially when the holidays move and the macro shifts.
>
> External factors matter because the demand curve isn't stationary — a 0.5 percentage-point rise
> in regional unemployment can soften baseline weekly demand by several percent, and that effect is
> *bigger* in some store types than others. Without modelling that, planners over-stock during macro
> downturns.
>
> Machine learning supports better strategic decisions because it can hold all of these factors
> simultaneously and produce a forecast that blends them with the right weights — and, when paired
> with segmentation, can produce *segment-specific* recommendations rather than a one-size-fits-all
> plan.

**Interviewer is checking**: business value before algorithm discussion. Real retail
decision-making relevance.

---

## Section 3 — Data Understanding & Feature Engineering (4:00 – 6:00, **2 min**)

**What to say (open the notebook to Section 1-3)**:

> The data lives in three CSVs joined on Store and Date. Sales is at the (Store, Dept, Date)
> grain — the target lives here as `Weekly_Sales`. Features is at the (Store, Date) grain — that's
> where Temperature, Fuel_Price, the five MarkDown columns, CPI and Unemployment live. Stores is
> static metadata: Type ∈ {A, B, C} and Size in square feet.
>
> Two structural quirks needed careful handling. **MarkDown1 through MarkDown5** are NaN for
> roughly half the rows, but the missingness is not random — it's *temporal*. The markdown
> program didn't start until November 2011. So I treat those NaNs as *zero — no promotion
> active* — not as missing values. You can see this clearly in chart 11 — markdown coverage is
> exactly zero before mid-November 2011 and then steps up. **CPI** and **Unemployment** also
> have a small NaN block in the future-window of the original Kaggle file, which I forward-fill
> *within each store* to preserve regional differences without leakage.
>
> On feature engineering — the workhorses for forecasting are the **lag features**: lag-1 (last
> week), lag-4 (last month), lag-52 (same week last year), and rolling 4-week and 12-week means.
> Critical detail: I compute these *strictly within each (Store, Dept) group* with `groupby...shift`,
> so there's zero cross-series leakage. I also add cyclical encoding of month and week — sin/cos
> pairs — which helps the linear baseline pick up seasonality without ordinal artifacts.
>
> Why is preprocessing critical here? Because the forecasting metric I use — WMAE — puts 5× weight
> on holiday weeks, and any leakage or wrong imputation would directly inflate the holiday-week
> error.

**Interviewer is checking**: data preparation quality, ability to identify useful predictive signals.

---

## Section 4 — Anomaly Detection & Time-Based Analysis (6:00 – 8:30, **2.5 min**)

**What to say (open the notebook to Section 7A — anomaly detection)**:

> I built three complementary anomaly detectors. The first is **Isolation Forest** — random
> partitioning, anomalies isolate fast. Set contamination to 1% so the planner gets a daily
> review queue of about 4,200 rows out of 421K. The second is **Local Outlier Factor** — a
> density-based detector that catches *local* anomalies, like a small Type-C store whose sales
> are out of line *for that store* even if they look normal at the chain level.
>
> The most important one is the third: **STL residual z-score**. This is the time-aware
> detector. I take each (Store, Dept) series, run STL decomposition — that's seasonal-trend
> decomposition with a 52-week period — and score each week by the residual's z-score. If
> |z| > 3, it's flagged.
>
> Why does that matter? Look at chart 13 — the STL plot for Store 1, Dept 1. There's a clear
> yearly cycle, and Christmas is *expected* to spike. A naive z-score on raw weekly sales
> would flag every Christmas as anomalous, which is useless to a planner. The STL detector
> separates the *expected* Christmas peak from a Christmas peak that's, say, 30% above what
> we'd expect *for that store and that week of year*. That's the actionable signal.
>
> The role of holidays, markdowns, and seasonality is therefore explicit in the model — they
> are *not* the anomalies; they are the baseline against which we measure anomalies. And I
> assess detector quality by the *holiday-share inside flagged rows* — a flagged holiday week
> is more likely to be a real demand event than noise, which gives me a proxy precision metric
> in the absence of labels.
>
> Anomaly handling improves data quality for downstream modelling because once we know which
> rows are truly unusual, we can decide whether to keep them, weight them down, or exclude them
> from training depending on the use case. For forecasting I keep them all — the high-volume
> outliers are signal — but I document them for audit.

**Avoid**: don't say "I dropped anomalies." That would be the wrong move here — they're often
business signal, not data errors.

---

## Section 5 — Forecasting, Segmentation & Pattern Discovery (8:30 – 10:30, **2 min**)

**What to say (open the notebook to Section 7B → 7C → 7D in order)**:

> Demand forecasting is the heaviest piece. Five tabular models plus one time-series benchmark.
> Linear Regression as a baseline, Ridge with TimeSeriesSplit hyperparameter tuning, Random
> Forest tuned and untuned, XGBoost tuned and untuned, and a SARIMAX(1,1,1)(1,1,1,52) on a
> single representative series. The metrics I report are **WMAE** (Walmart's official metric,
> 5× weight on holidays), RMSE for worst-case sensitivity, MAE for dollar interpretability,
> and R² for variance explained. The chosen model is **tuned XGBoost** — best WMAE, best RMSE,
> trains fast with the histogram tree method, and pickles to under 50 megabytes for deployment.
>
> Segmentation — I aggregate the 421K rows up to 45 stores using seven engineered features:
> mean weekly sales, sales volatility, holiday lift ratio, average markdown spend, regional
> CPI, regional unemployment, and store size. Standardise, then run KMeans, Agglomerative-Ward,
> and DBSCAN. I pick K by the silhouette curve (chart `seg_k_search.png`) and visualise the
> result in 2-D PCA. The clusters are interpretable in business terms — 'large urban
> high-markdown', 'mid-size suburban', 'small-format Type-C' — based on the per-feature z-score
> profile against the chain mean.
>
> Market basket — and here I'm transparent about a dataset constraint — there's no
> transaction-level data, so individual customer baskets don't exist. I infer them at the
> (Store, Date) level: a department is in the basket if its weekly sales exceed half its
> chain-wide median. Apriori on that binary matrix, min support 5%, max length 3, lift > 1.1.
> The output is a top-15 list of department co-spike rules — strong cross-merchandising
> candidates — and I document the limitation explicitly.
>
> What insights did this generate? First, lag features dominate forecasting — the series is
> heavily autoregressive. Second, store type and size carry overlapping information; I keep
> both because tree models handle the redundancy. Third, the segmentation produces clusters
> that are *not* just Type-stratified — there are within-Type-A behavioural splits driven by
> markdown sensitivity and holiday lift.

**Focus**: emphasise that the four models work *together* — the cluster assignment shapes the
inventory recommendation, the forecast shapes the safety stock, the anomaly detector flags
the exceptions, and the basket rules shape the cross-promotion calendar.

---

## Section 6 — External Factors & Strategy Formulation (10:30 – 12:30, **2 min**)

**What to say (switch to a strategy slide / open the strategy markdown section in the notebook)**:

> The external factors — CPI, Unemployment, Fuel Price, Temperature — show up in two places.
> First, in the *segmentation features*. CPI and Unemployment are bimodal — there are two
> distinct macro regions in this dataset — and the cluster centroids pick up on that. So
> region is *implicit* in the cluster assignment.
>
> Second, in the *forecasting feature set*. They're direct features in XGBoost, with
> small-but-real coefficients. The interaction effects are more interesting than the main
> effects: the cluster profile shows that *high-Unemployment clusters also have higher
> markdown sensitivity*. That's a strategic insight — defensive markdowns work better in
> economically stressed regions.
>
> The marketing strategy that comes out of this is **segment-specific**:
> - Cluster 0 — large urban, high markdown — concentrate the national promotion budget here.
>   Markdowns deliver measurable lift.
> - Cluster 1 — mid-size suburban — bundle promotions using the top association rules. Don't
>   go deeper on price; use cross-merchandising.
> - Cluster 2 — small-format Type-C — markdown-insensitive. Focus on assortment fit, not promotion.
>
> Inventory: use the XGBoost forecast as the demand mean, the per-cluster volatility as σ,
> standard `Z·σ·√LeadTime` for safety stock. High-volatility clusters need about 1.5× the
> buffer of low-volatility clusters at the same in-stock SLA.
>
> The external-factor playbook: if regional unemployment rises more than 0.5pp QoQ, trigger
> a defensive markdown campaign — but *only* in the clusters with high markdown sensitivity,
> based on the cluster profile.
>
> So that's how analytics connects to strategy — every model output has a named owner and a
> concrete action.

**Interviewer is checking**: ability to connect analytics with strategy; understanding of
external drivers.

---

## Section 7 — Evaluation, Challenges & Optimization (12:30 – 14:00, **1.5 min**)

**What to say (open the leaderboard chart `forecast_wmae.png`)**:

> Forecasting quality I evaluate primarily by WMAE — that's the holiday-weighted metric.
> Random Forest tuned beats Random Forest default by about 5% on WMAE; XGBoost tuned beats
> Random Forest tuned by another 8-12% — so tuning is non-trivial here, and the XGBoost
> hyperparameter space genuinely matters. Cross-validation uses TimeSeriesSplit, *not*
> KFold — random folds would let the model peek at future weeks during training.
>
> Segmentation quality I evaluate with **silhouette** and **Davies-Bouldin**. Silhouette at
> the chosen K is in the moderate-positive range — meaning clusters are real, not artefacts.
> Davies-Bouldin is in the same direction, low values for the chosen K.
>
> The challenges I hit:
> - **Noisy sales patterns** — high variance week-to-week, especially around holidays. Mitigated
>   by the lag and rolling features, plus the WMAE weighting that penalises holiday errors.
> - **Mixed objectives** — anomaly, segmentation, and forecasting use different feature
>   representations. Mitigated by aggregating up to store-level for segmentation while keeping
>   the row level for forecasting.
> - **Feature complexity** — 25+ engineered features. Random Forest importance plus correlation
>   plus domain knowledge gave me a defensible selection without aggressive elimination.
>
> Optimizations I made:
> - Better features — the lag-52 (year-over-year) feature alone moved WMAE noticeably.
> - Better model selection — XGBoost over RF over Linear; the gap is real.
> - Cleaner segmentation logic — store-level aggregation instead of pooling all rows.

---

## Section 8 — Learnings & Improvements (14:00 – 15:30, **1.5 min**)

**What to say**:

> The biggest learning from solving a multi-objective ML problem is that the *integration
> story* matters as much as any single model. I could have built three separate notebooks —
> one for forecasting, one for clustering, one for anomaly detection — and got similar
> per-model metrics. But the value comes from connecting them: cluster assignment determines
> safety stock σ, forecast determines demand mean, anomaly detector flags exceptions, basket
> rules shape promotions. That integration is the actual deliverable a retail organization can act on.
>
> The most valuable retail insight I found is that **markdown sensitivity is not uniform across
> store types** — it's *higher in high-Unemployment clusters*. That's a counterintuitive
> finding: planners often assume markdowns work everywhere; the cluster profile says they work
> *more* in stressed regions. That insight is actionable — concentrate markdown budget where
> it has the highest marginal lift.
>
> Future improvements:
> - Richer transaction data would replace department-level inferred baskets with item-level
>   association rules — much sharper recommendations.
> - Better personalization would extend the cluster-level recommendations to individual customer
>   segments — but needs a customer ID, which the dataset doesn't have.
> - Stronger long-term forecasting via hierarchical reconciliation (MinT) across the
>   Store-Dept-Total hierarchy. Each level is forecasted independently; reconciliation makes
>   them consistent.
> - Stronger recommendation logic — extending the basket rules to a learning-to-rank model that
>   personalises the cross-merchandise list per (Store, Date).

---

## Section 9 — Walk-through of the Notebook (15:30 – 35:00, **~20 min, the deep dive**)

> *This is the screen-share-and-explain block. The earlier sections cover what and why; this
> block walks the actual code, cell by cell, in the executed notebook. Pace yourself —
> roughly 1 minute per major code cell. The goal is to convince the reviewer that the code
> is correct, the decisions are documented in comments, and the outputs match the explanations.*

Suggested walk-through order (timing estimates):

| Time | Notebook cell | What to show |
|---|---|---|
| 15:30 – 17:00 | Imports + data load | Confirm dtypes, dayfirst parsing, three-table cardinality. |
| 17:00 – 18:30 | Wrangling | Markdown imputation, CPI ffill, three-way join validation. |
| 18:30 – 22:00 | Charts 1-16 | Click each saved PNG; restate the insight per chart. |
| 22:00 – 23:30 | Hypothesis tests | Show the printed t-stat / F-stat / U-stat and p-values. |
| 23:30 – 24:30 | Feature engineering | Highlight the *within-group* shift/rolling — point at the `groupby` call. |
| 24:30 – 26:30 | Anomaly detection | Show flag counts, holiday-share table, overlap matrix. |
| 26:30 – 28:30 | Segmentation | Silhouette curve, PCA scatter, dendrogram, cluster profile heatmap. |
| 28:30 – 32:00 | Forecasting | Each model's metrics; the leaderboard CSV; the WMAE bar chart; XGBoost feature importance. |
| 32:00 – 33:30 | Market basket | Top-15 rules table, scatter plot, lift > 1.5 callouts. |
| 33:30 – 35:00 | Save + load + sanity | Run the load cell live (it's fast); show the actual-vs-predicted comparison. |

---

## Section 10 — Closing & Q&A Prep (35:00 – 42:00, **7 min — Q&A buffer**)

> Wrap-up:
>
> - Recap: 4 sub-problems, 12 models, single integrated pipeline, deployment-ready notebook,
>   pickled XGBoost, README + tech doc + this video.
> - Repo URL on screen.
> - Email if a reviewer wants the live walkthrough.

### Anticipated follow-up questions (rehearsed answers)

**Q1 — How would you improve demand forecasting accuracy?**
> Three ways. First, hierarchical reconciliation — fit at Store-Dept-Total levels and use
> MinT to make them consistent; this typically reduces low-volume forecast error by 10-20%.
> Second, richer features — promotional calendars from third-party data, weather forecasts
> rather than realised temperature, regional event calendars (sports, concerts). Third,
> per-cluster modelling — fit a separate XGBoost per store cluster instead of one global model;
> small-format Type-C and large urban have different feature importances and benefit from
> not being averaged together.

**Q2 — Why is anomaly detection important before forecasting?**
> Two reasons. One, data quality — a stock-out week looks like 'low demand' to a forecaster
> but is actually 'we couldn't sell anything because we had nothing in stock'. If we don't
> flag and either weight down or impute that week, the forecaster learns the wrong baseline.
> Two, planner action — the daily review queue lets a regional manager intervene on real
> demand surprises before they become a stock-out or a write-off. The same model that improves
> forecast accuracy also produces an operational signal.

**Q3 — How do external factors affect retail performance?**
> They affect the *level* and the *slope* of weekly sales. CPI and Unemployment have small
> direct effects but larger interactions — high-Unemployment clusters have higher markdown
> sensitivity, meaning the same dollar of markdown produces more lift. Fuel price affects
> Type-C stores disproportionately because they serve more car-dependent customers. Temperature
> mostly drives foot traffic — a small but visible effect, and indirect. The integration is
> via the feature set in XGBoost and via the cluster definition for segmentation.

**Q4 — How would you validate your segmentation quality?**
> Quantitatively with silhouette (close to or above 0.5 is healthy on this kind of behavioural
> data) and Davies-Bouldin (lower is better). Qualitatively by checking that each cluster is
> *interpretable* — if I can give it a one-sentence business name like 'large urban
> high-markdown' that a planner recognises, the cluster is doing its job. The dendrogram from
> hierarchical clustering serves as a sanity check on the K choice. And the gold-standard test
> is **holdout stability** — re-run on a different time window and check that most stores
> stay in the same cluster.

**Q5 — How would this system help a retailer make better decisions?**
> Three concrete decisions per week, per cluster:
> 1. *Inventory* — XGBoost forecast → demand mean; cluster volatility → safety stock.
> 2. *Marketing* — markdown depth and frequency tailored by cluster sensitivity.
> 3. *Cross-merchandising* — top association rules drive the joint-promotion calendar.
>
> Plus a *daily exception queue* from the anomaly detector. Together that means fewer
> stock-outs, lower write-offs, and a higher promotion ROI. The strategy section in the
> notebook makes the connection explicit and includes implementation challenges — data
> freshness, cold-start for new stores, drift, and attribution.

---

## Pre-recording Final Checklist

- [ ] Notebook fully executed; all charts rendered.
- [ ] `outputs/figures/` contains all PNGs.
- [ ] Microphone tested at conversational volume.
- [ ] Screen resolution at 1080p so cell text is readable.
- [ ] Camera framing (if showing yourself) is clean and well-lit.
- [ ] This script printed or on a second monitor.
- [ ] Total runtime ≥ 40 minutes (target 42).
