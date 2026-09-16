"""Shared helpers for the sales-forecasting pipeline."""

import json

import numpy as np
import pandas as pd
import holidays


def load_feature_cols(path="data/feature_cols.json"):
    return json.load(open(path))


def holiday_months(years):
    """Set of (year, month) tuples that contain at least one US holiday."""
    us = holidays.US(years=sorted(set(years)))
    return {(d.year, d.month) for d in us}


def build_features(hist_sales, target_date, feature_cols, hmonths):
    """Leakage-safe features for `target_date` using only PAST sales.

    `hist_sales` must contain all sales up to (but NOT including) target month.
    """
    target_date = pd.Timestamp(target_date)
    s = np.asarray(hist_sales, float)
    return pd.DataFrame([{
        "Year": target_date.year,
        "Month": target_date.month,
        "Quarter": (target_date.month - 1) // 3 + 1,
        "Month_sin": np.sin(2 * np.pi * target_date.month / 12),
        "Month_cos": np.cos(2 * np.pi * target_date.month / 12),
        "IsHoliday": int((target_date.year, target_date.month) in hmonths),
        "Sales_lag_1": s[-1],
        "Sales_lag_2": s[-2],
        "Sales_lag_3": s[-3],
        "Sales_lag_12": s[-12],
        "Rolling_mean_3": s[-3:].mean(),
        "Rolling_std_3": s[-3:].std(ddof=1),
        "Rolling_mean_6": s[-6:].mean(),
        "Sales_diff_prev": s[-1] - s[-2],
        "Sales_pct_prev": s[-1] / s[-2] - 1,
    }])[feature_cols]


def recursive_forecast(model, hist_sales, dates, feature_cols, hmonths):
    """Predict future months one at a time, feeding predictions back as history."""
    hist = list(np.asarray(hist_sales, float))
    preds = []
    for d in dates:
        p = float(model.predict(build_features(np.array(hist), d, feature_cols, hmonths))[0])
        preds.append(p)
        hist.append(p)
    return np.array(preds)


def mape(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
