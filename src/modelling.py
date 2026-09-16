"""
Sales Forecasting — Model Training & Walk-Forward Evaluation (restarted pipeline)
=================================================================================

Models:
  1. Naive               (baseline)
  2. Seasonal Naive      (baseline, same month last year)
  3. Linear Regression
  4. ARIMA
  5. SARIMA  (m=12)
  6. Random Forest

Evaluation: expanding-window walk-forward (rolling-origin) backtest, NEVER shuffled.
  * Horizon = 6 months ahead.
  * Common comparison origins: t = 36..42 (7 origins covering 2018).
      - classical models train on the first `t` months of monthly_sales.csv
      - ML models train on feature rows with Date <= the last training month
        (>= 24 training rows), and forecast recursively.
  * A secondary, longer backtest (t = 24..42, 19 origins) is also reported for
    the classical models only (ML models have too few rows before 2017-12).

Metrics: MAE, RMSE, MAPE (pooled across all forecast points).
"""

import json
import warnings

import numpy as np
import pandas as pd
import holidays
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# 0. Load artifacts
# ----------------------------------------------------------------------------
monthly = pd.read_csv("data/monthly_sales.csv", parse_dates=["Date"])
features = pd.read_csv("data/features.csv", parse_dates=["Date"])
feature_cols = json.load(open("data/feature_cols.json"))

years = sorted(monthly["Date"].dt.year.unique())
us_holidays = holidays.US(years=years)
holiday_months = {(d.year, d.month) for d in us_holidays}

# ----------------------------------------------------------------------------
# 1. Metrics
# ----------------------------------------------------------------------------
def mape(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


# ----------------------------------------------------------------------------
# 2. Leakage-safe feature builder (matches src/preprocess.py)
# ----------------------------------------------------------------------------
def build_features(hist_sales, target_date):
    """hist_sales: array of Sales up to (but NOT including) target month."""
    target_date = pd.Timestamp(target_date)
    return pd.DataFrame([{
        "Year": target_date.year,
        "Month": target_date.month,
        "Quarter": (target_date.month - 1) // 3 + 1,
        "Month_sin": np.sin(2 * np.pi * target_date.month / 12),
        "Month_cos": np.cos(2 * np.pi * target_date.month / 12),
        "IsHoliday": int((target_date.year, target_date.month) in holiday_months),
        "Sales_lag_1": hist_sales[-1],
        "Sales_lag_2": hist_sales[-2],
        "Sales_lag_3": hist_sales[-3],
        "Sales_lag_12": hist_sales[-12],
        "Rolling_mean_3": hist_sales[-3:].mean(),
        "Rolling_std_3": hist_sales[-3:].std(ddof=1),
        "Rolling_mean_6": hist_sales[-6:].mean(),
        "Sales_diff_prev": hist_sales[-1] - hist_sales[-2],
        "Sales_pct_prev": hist_sales[-1] / hist_sales[-2] - 1,
    }])[feature_cols]


def ml_recursive_forecast(model, hist_sales, forecast_dates):
    """Predict future months one at a time, feeding predictions back into lags."""
    hist = list(hist_sales)
    preds = []
    for d in forecast_dates:
        pred = model.predict(build_features(np.array(hist), d))[0]
        preds.append(pred)
        hist.append(pred)  # predicted value becomes history for the next step
    return np.array(preds)


# ----------------------------------------------------------------------------
# 3. Backtest
# ----------------------------------------------------------------------------
def run_backtest(origins):
    """Return a dict: model -> list of (actual, predicted) for each forecast step."""
    results = {m: [] for m in ["Naive", "Seasonal Naive", "Linear Regression",
                               "ARIMA", "SARIMA", "Random Forest"]}

    for t in origins:
        train = monthly.iloc[:t]
        test = monthly.iloc[t:t + 6]
        y_true = test["Sales"].values
        y_dates = test["Date"].values
        last_train_date = train["Date"].iloc[-1]

        # --- Baselines + classical models (monthly series) ---
        series = train.set_index("Date")["Sales"]

        naive = np.full(6, series.iloc[-1])
        snaive = series.iloc[-12:-6].values          # same month last year

        preds = {"Naive": naive, "Seasonal Naive": snaive}

        try:
            arima_fit = ARIMA(series, order=(1, 1, 1)).fit()
            preds["ARIMA"] = arima_fit.forecast(6).values
        except Exception:
            preds["ARIMA"] = np.full(6, np.nan)

        try:
            sarima_fit = SARIMAX(
                series, order=(1, 1, 1), seasonal_order=(1, 0, 1, 12),
                enforce_stationarity=False, enforce_invertibility=False,
            ).fit(disp=False)
            preds["SARIMA"] = sarima_fit.forecast(6).values
        except Exception:
            preds["SARIMA"] = np.full(6, np.nan)

        # --- ML models (feature matrix, trained only on rows up to origin) ---
        tr = features[features["Date"] <= last_train_date]
        X_tr, y_tr = tr[feature_cols], tr["Sales"]

        lr = LinearRegression().fit(X_tr, y_tr)
        preds["Linear Regression"] = ml_recursive_forecast(lr, series.values, y_dates)

        rf = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42).fit(X_tr, y_tr)
        preds["Random Forest"] = ml_recursive_forecast(rf, series.values, y_dates)

        for m in results:
            results[m].append((y_true, preds[m]))

    return results


def summarize(results):
    rows = []
    for m, chunks in results.items():
        y_true = np.concatenate([a for a, _ in chunks])
        y_pred = np.concatenate([p for _, p in chunks])
        ok = ~np.isnan(y_pred)
        y_true, y_pred = y_true[ok], y_pred[ok]
        rows.append({
            "Model": m,
            "MAE": float(np.mean(np.abs(y_true - y_pred))),
            "RMSE": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
            "MAPE": float(mape(y_true, y_pred)),
            "n_points": int(len(y_true)),
        })
    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)


# ----------------------------------------------------------------------------
# 4. Run
# ----------------------------------------------------------------------------
print("=" * 70)
print("A) COMMON backtest  (all 6 models, origins t = 36..42, 6-step-ahead)")
print("=" * 70)
common = summarize(run_backtest(range(36, 43)))
print(common.to_string(index=False))

print()
print("=" * 70)
print("B) LONGER backtest  (classical models only, origins t = 24..42)")
print("=" * 70)
long_origins = range(24, 43)
long = run_backtest(long_origins)
# keep only classical models for the longer window
for m in ["Linear Regression", "Random Forest"]:
    long.pop(m)
long_summary = summarize(long)
print(long_summary.to_string(index=False))

common.to_csv("data/model_results_common.csv", index=False)
long_summary.to_csv("data/model_results_classical_long.csv", index=False)
print()
print("Saved: data/model_results_common.csv, data/model_results_classical_long.csv")
