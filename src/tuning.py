"""Hyperparameter tuning (time-series-safe) + re-evaluation.

Tunes:
  * Random Forest -> GridSearchCV with TimeSeriesSplit (expanding-window CV)
  * ARIMA / SARIMA -> pmdarima.auto_arima (AIC order selection, m=12),
                      refit inside each walk-forward origin (no test leakage)

Then re-runs the same common walk-forward backtest (origins t = 36..42,
6-step-ahead) so tuned models are directly comparable to the untuned ones.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import pmdarima as pm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_feature_cols, holiday_months, recursive_forecast, mape

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)

monthly = pd.read_csv("data/monthly_sales.csv", parse_dates=["Date"])
features = pd.read_csv("data/features.csv", parse_dates=["Date"])
feature_cols = load_feature_cols()
hmonths = holiday_months(monthly["Date"].dt.year.unique())

# ============================================================================
# A) Tune Random Forest (expanding-window CV over the feature matrix)
# ============================================================================
X, y = features[feature_cols], features["Sales"]
tscv = TimeSeriesSplit(n_splits=5)

param_grid = {
    "n_estimators": [100, 200, 400],
    "max_depth": [3, 5, 7],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 0.8],
}
gs = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    scoring="neg_mean_absolute_error",
    cv=tscv,
    n_jobs=-1,
)
gs.fit(X, y)
best_rf_params = gs.best_params_
print("A) Random Forest tuning (TimeSeriesSplit, 5 folds)")
print("   best params:", best_rf_params)
print(f"   best CV MAE: {-gs.best_score_:.1f}\n")


# ============================================================================
# B) auto_arima order selection (seasonal m=12 and non-seasonal)
# ============================================================================
def auto_sarima_forecast(series, h):
    model = pm.auto_arima(
        series, seasonal=True, m=12,
        start_p=0, start_q=0, max_p=3, max_q=3,
        start_P=0, start_Q=0, max_P=2, max_Q=2,
        d=None, D=None,
        stepwise=True, n_jobs=1,
        error_action="ignore", suppress_warnings=True, trace=False,
    )
    return model.predict(n_periods=h), model.order, model.seasonal_order


def auto_arima_forecast(series, h):
    model = pm.auto_arima(
        series, seasonal=False,
        start_p=0, start_q=0, max_p=3, max_q=3,
        d=None, stepwise=True, n_jobs=1,
        error_action="ignore", suppress_warnings=True, trace=False,
    )
    return model.predict(n_periods=h), model.order, (0, 0, 0, 0)


# ============================================================================
# C) Re-run common backtest with TUNED models
# ============================================================================
results = {m: [] for m in ["Naive", "Seasonal Naive", "Linear Regression",
                           "Random Forest (tuned)", "ARIMA (tuned)", "SARIMA (tuned)"]}

for t in range(36, 43):
    train = monthly.iloc[:t]
    test = monthly.iloc[t:t + 6]
    y_true = test["Sales"].values
    y_dates = test["Date"].values
    last_train_date = train["Date"].iloc[-1]
    series = train.set_index("Date")["Sales"]

    preds = {
        "Naive": np.full(6, series.iloc[-1]),
        "Seasonal Naive": series.iloc[-12:-6].values,
    }

    # tuned RF
    tr = features[features["Date"] <= last_train_date]
    rf = RandomForestRegressor(**best_rf_params, random_state=42).fit(tr[feature_cols], tr["Sales"])
    preds["Random Forest (tuned)"] = recursive_forecast(rf, series.values, y_dates,
                                                        feature_cols, hmonths)

    # linear regression (no hyperparameters)
    lr = __import__("sklearn.linear_model", fromlist=["LinearRegression"]).LinearRegression().fit(
        tr[feature_cols], tr["Sales"])
    preds["Linear Regression"] = recursive_forecast(lr, series.values, y_dates,
                                                    feature_cols, hmonths)

    # auto_arima (non-seasonal)
    try:
        arima_pred, order, _ = auto_arima_forecast(series, 6)
        preds["ARIMA (tuned)"] = np.asarray(arima_pred)
    except Exception:
        preds["ARIMA (tuned)"] = np.full(6, np.nan)

    # auto_arima (seasonal m=12)
    try:
        sarima_pred, order, sorder = auto_sarima_forecast(series, 6)
        preds["SARIMA (tuned)"] = np.asarray(sarima_pred)
    except Exception:
        preds["SARIMA (tuned)"] = np.full(6, np.nan)

    for m in results:
        results[m].append((y_true, preds[m]))


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
    })

tuned_df = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
print("=" * 70)
print("C) Common backtest with TUNED models (origins t = 36..42, 6-step-ahead)")
print("=" * 70)
print(tuned_df.to_string(index=False))

tuned_df.to_csv("data/model_results_tuned.csv", index=False)
print("\nSaved: data/model_results_tuned.csv")
