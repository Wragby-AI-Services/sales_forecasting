# Sales Forecasting — Project Architecture

Monthly sales forecasting for the Superstore-style retail dataset
(order-level data, 2015-01 → 2018-12), served as a Streamlit app.

## Overview

1. Raw order-line data is cleaned and aggregated to a monthly sales series.
2. Leakage-safe features are engineered (only PAST information per row).
3. Six models are compared with an expanding-window walk-forward backtest.
4. The best model (tuned Random Forest) is retrained on all data and saved.
5. A Streamlit app loads the saved model and forecasts N months ahead on demand.
6. EDA / time-series diagnostic figures are generated into `figures/`.

## Directory layout

```
sales-forecasting/
├── app.py                      # Streamlit app (forecast UI + charts)
├── README.md                   # project overview and usage
├── requirements.txt            # Python dependencies
├── note.txt                    # initial Git scaffolding notes
│
├── data/                       # all data artifacts (inputs and outputs)
│   ├── raw.csv                 # 9,799 order-line records (18 cols, DD/MM/YYYY)
│   ├── Processed.csv           # raw + Year/Month/Day (legacy EDA artifact)
│   ├── monthly_sales.csv       # 48 monthly total-sales rows (2015-01 → 2018-12)
│   ├── features.csv            # 36 rows × 15 leakage-safe features (after lag-12)
│   ├── feature_cols.json       # ordered feature list for the ML models
│   ├── model_results_common.csv          # all 6 models, origins t=36..42
│   ├── model_results_classical_long.csv  # classical models, origins t=24..42
│   └── model_results_tuned.csv           # tuned models, origins t=36..42
│
├── docs/                       # project documentation
│   ├── blockers and resolution.txt
│   ├── project architecture.md
│   └── references.md
│
├── figures/                    # EDA + time-series diagnostics (PNG)
│   └── regenerate with `python src/make_figures.py`
│
├── models/                     # saved inference artifacts
│   ├── sales_forecast_rf.pkl   # final Random Forest model (joblib)
│   └── model_meta.json         # model name, params, feature list
│
├── notebooks/
│   ├── sf_data_cleaning.ipynb  # data-cleaning walkthrough
│   └── sf_eda.ipynb            # exploratory data analysis
│
└── src/                        # pipeline source code
    ├── common.py               # shared helpers (feature builder, metrics)
    ├── preprocess.py           # raw → monthly_sales.csv / features.csv / feature_cols.json
    ├── modelling.py            # walk-forward evaluation of 6 models
    ├── tuning.py               # RF GridSearchCV + auto_arima, tuned backtest
    ├── finalize.py             # retrain best model + save artifacts
    ├── forecast.py             # production forecast(months_ahead) function
    └── make_figures.py         # regenerate figures/ from data/raw.csv
```

## Pipeline / data flow

```
data/raw.csv
      │  src/preprocess.py  (parse DD/MM/YYYY, clean, aggregate)
      ▼
data/monthly_sales.csv ────────────────────────────► data/features.csv
      │                                                  │  data/feature_cols.json
      │                                                  ▼
      │                            src/modelling.py   ──► data/model_results_common.csv
      │                            src/tuning.py      ──► data/model_results_tuned.csv
      │                                                  data/model_results_classical_long.csv
      ▼
src/finalize.py   (retrain tuned RF on all data)  ──► models/sales_forecast_rf.pkl
                                                       models/model_meta.json
      │
      ▼
src/forecast.py   (recursive multi-step forecast)
      │
      ▼
app.py            (Streamlit UI: chart + result table)
```

## Components

### src/preprocess.py
- Loads `data/raw.csv` with `dayfirst=True` (dates are DD/MM/YYYY).
- Drops `Row ID`, median-fills `Postal Code`, drops duplicates.
- Aggregates to monthly totals → `data/monthly_sales.csv` (48 months).
- Engineers 15 leakage-safe features → `data/features.csv` (36 rows, after the
  lag-12 drop) and `data/feature_cols.json`.
- Key design rule: every feature uses only PAST sales. The leakage features
  `Diff_1` / `Pct_change_1` from the old pipeline were replaced by
  `Sales_diff_prev` / `Sales_pct_prev` (previous-month change only).

### src/modelling.py
- Expanding-window walk-forward backtest (never shuffled), 6-step-ahead.
- Models: Naive, Seasonal Naive, Linear Regression, ARIMA, SARIMA (m=12),
  Random Forest.
- Common origins t=36..42 for all models; a longer t=24..42 window for
  classical models only.
- Writes `data/model_results_common.csv` and
  `data/model_results_classical_long.csv`.

### src/tuning.py
- Random Forest: GridSearchCV with TimeSeriesSplit (expanding CV).
- ARIMA/SARIMA: pmdarima auto_arima (seasonal m=12), refit per origin.
- Re-runs the common backtest with tuned models → `data/model_results_tuned.csv`.

### src/finalize.py
- Retrains the tuned Random Forest on all feature rows and saves
  `models/sales_forecast_rf.pkl` + `models/model_meta.json`.

### src/common.py
- Shared helpers: `build_features`, `holiday_months`, `recursive_forecast`,
  `mape`, `load_feature_cols`.

### src/forecast.py
- Production entry point `forecast(months_ahead=6)`: loads the saved model and
  metadata, then recursively forecasts the next N months, feeding predictions
  back into the lag features.

### app.py
- Streamlit UI: months slider, forecast line chart (Altair) and result table.

### src/make_figures.py
- Regenerates all PNGs in `figures/` from `data/raw.csv`.
- Run: `python src/make_figures.py`.

## Data artifacts

| File | Rows | Description |
|---|---|---|
| `data/raw.csv` | 9,799 | Raw order-line records (DD/MM/YYYY dates) |
| `data/Processed.csv` | 9,799 | raw + Year/Month/Day (legacy EDA helper) |
| `data/monthly_sales.csv` | 48 | Monthly total sales, 2015-01 → 2018-12 |
| `data/features.csv` | 36 | Leakage-safe ML features (after lag-12) |
| `data/feature_cols.json` | 15 | Ordered feature list |

## Model artifacts

| File | Description |
|---|---|
| `models/sales_forecast_rf.pkl` | Final Random Forest (tuned params) |
| `models/model_meta.json` | model, params, feature_cols, train_rows |

## Figures

`figures/` holds EDA and time-series diagnostic PNGs, regenerated from the
current data by `src/make_figures.py`:

- `sales distribution.png`, `sales boxplot.png` — target variable
- `Daily total sales.png`, `monthly total sales.png` — sales over time
- `Total Sales by Month.png`, `total sales by week.png`,
  `total sales by year.png` — calendar seasonality
- `total sales by category.png`, `total sales by sub-category.png`,
  `total sales by segment.png`, `total sales by region.png` — categoricals
- `correlation heatmap.png` — numeric feature correlations
- `rolling mean and std.png` — stationarity check
- `additive sales-seasonal-residual.png`,
  `multiplicative sales-seasonal-residual.png` — decomposition
- `autocorrelation-partialautocorrelation.png` — ACF/PACF
- `first differenced monthly sales.png` — first difference

## Key design decisions

- Leakage-safe feature engineering: every feature uses only PAST sales.
- All 48 months are kept for classical models; ML models use the 36 rows that
  remain after the lag-12 feature is dropped.
- Walk-forward (rolling-origin) validation instead of a single holdout.
- SARIMA seasonal period m=12 (annual) for monthly data.
