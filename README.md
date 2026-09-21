# Sales Forecasting — Superstore Retail Dataset

Monthly sales forecasting for a Superstore-style retail dataset
(order-level data, **2015-01 → 2018-12**), delivered as an interactive
**Streamlit app** backed by a leak-safe machine-learning pipeline.

The final model is a **tuned Random Forest** (walk-forward validation
MAPE ≈ 22.8%). The app forecasts the next 1–24 months on demand, plots
history + forecast together, and explains the numbers.

---

## 1. Overview

The project goes end-to-end from raw order lines to a deployable app:

1. **Clean & aggregate** order-level records into a monthly sales series.
2. **Engineer leakage-safe features** (only *past* information per row).
3. **Compare six models** with an expanding-window walk-forward backtest.
4. **Tune** the best model (Random Forest) with time-series-safe CV.
5. **Retrain on all data** and save the model + metadata.
6. **Serve** the model through a Streamlit app with charts, a result table,
   a CSV download button, and an automatic interpretation of the forecast.
7. **Generate** all EDA / time-series diagnostic figures into `figures/`.

---

## 2. Dataset

| File | Rows | Description |
|---|---|---|
| `data/raw.csv` | 9,799 | Raw order-line records, 18 columns, dates in **DD/MM/YYYY** |
| `data/Processed.csv` | 9,799 | `raw.csv` + `Year`/`Month`/`Day` (legacy EDA helper) |
| `data/monthly_sales.csv` | 48 | Monthly total sales, 2015-01 → 2018-12 |
| `data/features.csv` | 36 | 15 leakage-safe ML features (rows after the lag-12 drop) |
| `data/feature_cols.json` | 15 | Ordered feature list used by the ML models |

Key columns in `raw.csv`: `Order Date`, `Ship Date`, `Segment`, `Region`,
`Category`, `Sub-Category`, `Sales` (plus IDs and customer/product fields).

---

## 3. Project structure

```
sales-forecasting/
├── app.py                      # Streamlit app (forecast UI)
├── README.md
├── requirements.txt
├── note.txt                    # initial Git scaffolding notes
│
├── data/                       # data artifacts (inputs + outputs)
│   ├── raw.csv
│   ├── Processed.csv
│   ├── monthly_sales.csv
│   ├── features.csv
│   ├── feature_cols.json
│   ├── model_results_common.csv
│   ├── model_results_classical_long.csv
│   └── model_results_tuned.csv
│
├── docs/
│   ├── blockers and resolution.txt
│   ├── project architecture.md
│   └── references.md
│
├── figures/                    # EDA + time-series diagnostics (PNG)
│
├── models/                     # saved inference artifacts
│   ├── sales_forecast_rf.pkl   # final Random Forest (joblib)
│   └── model_meta.json         # model name, params, feature list
│
├── notebooks/
│   ├── sf_data_cleaning.ipynb  # data-cleaning walkthrough
│   └── sf_eda.ipynb            # exploratory data analysis
│
└── src/
    ├── common.py               # shared helpers (feature builder, metrics)
    ├── preprocess.py           # raw → monthly_sales / features / feature_cols
    ├── modelling.py            # walk-forward evaluation of 6 models
    ├── tuning.py               # RF GridSearchCV + auto_arima
    ├── finalize.py             # retrain best model + save artifacts
    ├── forecast.py             # production forecast(months_ahead) function
    └── make_figures.py         # regenerate figures/ from data/raw.csv
```

---

## 4. Pipeline

### Step 1 — Preprocess & feature engineering (`src/preprocess.py`)

- Loads `data/raw.csv` with `dayfirst=True` (dates are **DD/MM/YYYY**).
- Drops the meaningless `Row ID`, median-fills `Postal Code`, drops duplicates.
- Aggregates order lines to a monthly total → `data/monthly_sales.csv`
  (48 months).
- Builds 15 features per month → `data/features.csv` and
  `data/feature_cols.json`.

### Step 2 — Model evaluation (`src/modelling.py`)

- Expanding-window **walk-forward** backtest (never shuffled), 6-step-ahead.
- Models: Naive, Seasonal Naive, Linear Regression, ARIMA, SARIMA (m=12),
  Random Forest.
- Common comparison origins `t = 36..42` for all six models; a longer
  `t = 24..42` window for the classical models only.
- Writes `data/model_results_common.csv` and
  `data/model_results_classical_long.csv`.

### Step 3 — Hyperparameter tuning (`src/tuning.py`)

- Random Forest: `GridSearchCV` with `TimeSeriesSplit` (expanding CV).
- ARIMA / SARIMA: `pmdarima.auto_arima` (seasonal m=12), refit inside each
  origin to avoid test leakage.
- Re-runs the common backtest with the tuned models →
  `data/model_results_tuned.csv`.

### Step 4 — Finalize (`src/finalize.py`)

- Retrains the **tuned Random Forest** on all feature rows.
- Saves `models/sales_forecast_rf.pkl` and `models/model_meta.json`.

### Step 5 — Forecast (`src/forecast.py`)

- `forecast(months_ahead=6)` loads the saved model + metadata, then predicts
  recursively — each prediction is fed back into the lag features for the
  next month (multi-step, leakage-safe).

### Step 6 — Figures (`src/make_figures.py`)

- Regenerates every PNG in `figures/` from `data/raw.csv`.

---

## 5. Feature engineering (leakage-safe)

Every feature is computed from **past sales only** — never the target month's
own sales. This was the central fix after the previous pipeline produced a
fake near-zero error from leakage.

| Feature | Meaning |
|---|---|
| `Year`, `Month`, `Quarter` | Calendar position |
| `Month_sin`, `Month_cos` | Cyclical month encoding |
| `IsHoliday` | Whether the month contains a US holiday |
| `Sales_lag_1`, `Sales_lag_2`, `Sales_lag_3`, `Sales_lag_12` | Past sales levels |
| `Rolling_mean_3`, `Rolling_std_3`, `Rolling_mean_6` | Rolling statistics (shifted) |
| `Sales_diff_prev`, `Sales_pct_prev` | Previous-month change (replaces the leaky `Diff_1`/`Pct_change_1`) |

---

## 6. Models evaluated

- **Baselines:** Naive, Seasonal Naive
- **Classical:** ARIMA, SARIMA (m=12)
- **Machine learning:** Linear Regression, Random Forest

Removed from the comparison (after testing): ETS, Theta,
XGBoost/LightGBM, ensembles, Prophet, STL.

---

## 7. Evaluation methodology & results

**Method:** expanding-window (rolling-origin) walk-forward backtest, never
shuffled, with a **6-month-ahead** horizon. Metrics are pooled across all
forecast points: MAE, RMSE, MAPE.

### A) Common backtest — all 6 models, origins t = 36..42

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| Naive | 27,645 | 34,333 | 54.8% |
| ARIMA | 19,259 | 25,771 | 36.7% |
| SARIMA | 14,785 | 17,462 | 29.6% |
| Seasonal Naive | 14,144 | 17,506 | 23.5% |
| Random Forest | 13,715 | 16,737 | 23.8% |
| **Linear Regression** | **12,368** | **15,781** | 27.7% |

### B) Tuned models — origins t = 36..42

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| Naive | 27,645 | 34,333 | 54.8% |
| ARIMA (tuned) | 18,278 | 26,332 | 29.4% |
| SARIMA (tuned) | 15,985 | 20,954 | 27.1% |
| Seasonal Naive | 14,144 | 17,506 | 23.5% |
| **Random Forest (tuned)** | **13,549** | **17,014** | **22.8%** |
| Linear Regression | 12,368 | 15,781 | 27.7% |

> Linear Regression has the lowest MAE but a worse MAPE and is more fragile
> on this small dataset (36 training rows vs. 15 features). The **tuned
> Random Forest** was selected as the production model for its best MAPE
> (22.8%) and robustness.

### C) Longer classical backtest — origins t = 24..42 (114 points)

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| Naive | 25,391 | 30,871 | 55.8% |
| ARIMA | 19,047 | 24,848 | 38.8% |
| **SARIMA** | **13,264** | **15,586** | 28.5% |
| Seasonal Naive | 13,634 | 16,648 | 25.7% |

---

## 8. Final model

**Random Forest (tuned)** — retrained on all 36 feature rows and saved to
`models/sales_forecast_rf.pkl`.

| Hyperparameter | Value |
|---|---|
| `n_estimators` | 400 |
| `max_depth` | 3 |
| `min_samples_leaf` | 1 |
| `max_features` | 0.8 |
| `random_state` | 42 |

---

## 9. The Streamlit app (`app.py`)

Run it with:

```bash
streamlit run app.py
```

Features:

- **Months slider** — choose 1 to 24 months ahead.
- **Forecast chart** — Altair line chart of history (actual) + forecast,
  with the X-axis showing each year exactly once.
- **Result table** — compact, content-width table (Date + Forecast).
- **Download button** — downloads the generated table as a CSV file
  (`sales_forecast_<N>_months.csv`).
- **Interpretation section** — auto-generated plain-English reading of the
  table: total, monthly average vs. recent history, peak/trough months, and
  the direction of the horizon.

---

## 10. Figures (`figures/`)

Regenerated from the current data with `python src/make_figures.py`:

- **Target:** `sales distribution.png`, `sales boxplot.png`
- **Over time:** `Daily total sales.png`, `monthly total sales.png`
- **Seasonality:** `Total Sales by Month.png`, `total sales by week.png`,
  `total sales by year.png`
- **Categoricals:** `total sales by category.png`, `total sales by sub-category.png`,
  `total sales by segment.png`, `total sales by region.png`
- **Correlation:** `correlation heatmap.png`
- **Stationarity:** `rolling mean and std.png`, `first differenced monthly sales.png`
- **Decomposition:** `additive sales-seasonal-residual.png`,
  `multiplicative sales-seasonal-residual.png`
- **Autocorrelation:** `autocorrelation-partialautocorrelation.png`

---

## 11. Getting started

### Install

```bash
pip install -r requirements.txt
```

### Run the full pipeline (rebuild everything from raw data)

```bash
python src/preprocess.py     # raw.csv → monthly_sales.csv, features.csv, feature_cols.json
python src/modelling.py      # walk-forward model comparison
python src/tuning.py         # RF tuning + auto_arima + tuned backtest
python src/finalize.py       # retrain tuned RF + save model artifacts
python src/make_figures.py   # regenerate figures/
```

### Forecast from the command line

```bash
python src/forecast.py 6     # forecast 6 months ahead (prints the table)
```

### Launch the app

```bash
streamlit run app.py
```

---

## 12. Requirements

```text
pandas, numpy, scikit-learn, joblib, holidays, streamlit
statsmodels, pmdarima            # pipeline
matplotlib, seaborn              # figures
```

See `requirements.txt` for the version pins.

---

## 13. Key fixes vs. the previous pipeline

- **Dates** parsed with `dayfirst=True` (raw data is DD/MM/YYYY).
- **Leakage removed** — the leaky `Diff_1` / `Pct_change_1` features (which
  used the current month's target) were replaced with past-only
  `Sales_diff_prev` / `Sales_pct_prev`. This was why Linear Regression had
  shown a fake MAE ≈ 0.
- **Kept all 48 months** — the old lag-12 step silently dropped a full year,
  leaving only 36 rows for the classical models.
- **SARIMA seasonal period** corrected from `m=7` to `m=12` (annual).
- **Walk-forward validation** instead of a single 8-month holdout.

---

## 14. Blockers & resolutions

The full log of blockers encountered and how each was resolved lives in
[`docs/blockers and resolution.txt`](docs/blockers and resolution.txt).

## 15. References

Learning resources and tools used are listed in
[`docs/references.md`](docs/references.md).

## 16. Further documentation

- [`docs/project architecture.md`](docs/project architecture.md) — detailed
  architecture and data-flow.
