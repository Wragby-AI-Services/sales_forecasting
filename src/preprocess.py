"""
Sales Forecasting — Feature Engineering / Preprocessing (restarted pipeline)
=============================================================================

Produces three model-ready artifacts from raw order-level data:

  1. data/monthly_sales.csv  -> Date, Sales            (48 monthly observations)
                                 used by: Naive, Seasonal Naive, ARIMA, SARIMA
  2. data/features.csv       -> Date, Sales + features (36 rows after lag-12)
                                 used by: Linear Regression, Random Forest
  3. data/feature_cols.json  -> ordered feature list for the ML models

Key fixes vs. the previous pipeline
-----------------------------------
  * Dates are DD/MM/YYYY -> parsed with dayfirst=True.
  * Removed the leakage features `Diff_1` and `Pct_change_1`. Those used the
    CURRENT month's sales (i.e. the target) and were the reason Linear
    Regression showed a fake MAE ~ 0 and XGBoost assigned `Diff_1` ~30%
    importance. Their safe replacements (`Sales_diff_prev`, `Sales_pct_prev`)
    describe the PREVIOUS month's change only.
  * The monthly series keeps all 48 months. The old pipeline silently threw
    away the first 12 months when building Sales_lag_12, leaving only 36.
"""

import json
import warnings

import numpy as np
import pandas as pd
import holidays

warnings.filterwarnings("ignore")

RAW = "data/raw.csv"

# ----------------------------------------------------------------------------
# 1. Load raw order-level data
# ----------------------------------------------------------------------------
df = pd.read_csv(RAW, parse_dates=["Order Date", "Ship Date"], dayfirst=True)

# ----------------------------------------------------------------------------
# 2. Minimal cleaning (data was already mostly clean)
# ----------------------------------------------------------------------------
df = df.drop(columns=["Row ID"], errors="ignore")                    # meaningless index
df["Postal Code"] = df["Postal Code"].fillna(df["Postal Code"].median())
df = df.drop_duplicates()
print(f"Cleaned order lines: {len(df):,} rows")

# ----------------------------------------------------------------------------
# 3. Aggregate to monthly total sales
# ----------------------------------------------------------------------------
monthly = (
    df.set_index("Order Date")
    .resample("ME")["Sales"]
    .sum()
    .reset_index()
)
monthly.columns = ["Date", "Sales"]
print(
    f"Monthly series: {monthly.shape[0]} months, "
    f"{monthly['Date'].min():%Y-%m} to {monthly['Date'].max():%Y-%m}"
)

# ----------------------------------------------------------------------------
# 4. Feature engineering (leakage-safe: only PAST information per row)
# ----------------------------------------------------------------------------
ts = monthly.copy()

# 4.1 Calendar features
ts["Year"] = ts["Date"].dt.year
ts["Month"] = ts["Date"].dt.month
ts["Quarter"] = ts["Date"].dt.quarter
ts["Month_sin"] = np.sin(2 * np.pi * ts["Month"] / 12)   # cyclical encoding
ts["Month_cos"] = np.cos(2 * np.pi * ts["Month"] / 12)

# 4.2 Holiday flag: does this month contain a US holiday?
us_holidays = holidays.US(years=sorted(ts["Year"].unique()))
holiday_months = {(d.year, d.month) for d in us_holidays}
ts["IsHoliday"] = ts["Date"].apply(
    lambda d: int((d.year, d.month) in holiday_months)
)

# 4.3 Lags (past sales only)
for lag in [1, 2, 3, 12]:
    ts[f"Sales_lag_{lag}"] = ts["Sales"].shift(lag)

# 4.4 Rolling windows (shift(1) first, so the current month is never used)
ts["Rolling_mean_3"] = ts["Sales"].shift(1).rolling(3).mean()
ts["Rolling_std_3"] = ts["Sales"].shift(1).rolling(3).std()
ts["Rolling_mean_6"] = ts["Sales"].shift(1).rolling(6).mean()

# 4.5 Momentum (change in the PREVIOUS month — past-only, no leakage)
ts["Sales_diff_prev"] = ts["Sales"].diff().shift(1)        # Sales_{t-1} - Sales_{t-2}
ts["Sales_pct_prev"] = ts["Sales"].pct_change().shift(1)   # Sales_{t-1}/Sales_{t-2} - 1

# ----------------------------------------------------------------------------
# 5. Feature matrix for the ML models
# ----------------------------------------------------------------------------
feature_cols = [
    "Year", "Month", "Quarter", "Month_sin", "Month_cos", "IsHoliday",
    "Sales_lag_1", "Sales_lag_2", "Sales_lag_3", "Sales_lag_12",
    "Rolling_mean_3", "Rolling_std_3", "Rolling_mean_6",
    "Sales_diff_prev", "Sales_pct_prev",
]

ml = ts.dropna(subset=feature_cols).reset_index(drop=True)
print(f"ML feature matrix: {ml.shape[0]} rows x {len(feature_cols)} features")

# ----------------------------------------------------------------------------
# 6. Save artifacts
# ----------------------------------------------------------------------------
monthly[["Date", "Sales"]].to_csv("data/monthly_sales.csv", index=False)
ml[["Date", "Sales"] + feature_cols].to_csv("data/features.csv", index=False)
with open("data/feature_cols.json", "w") as f:
    json.dump(feature_cols, f, indent=2)

print("Saved: data/monthly_sales.csv, data/features.csv, data/feature_cols.json")
print("Feature columns:", feature_cols)
