"""Production forecasting: load saved artifacts and forecast N months ahead.

Usage:
    python src/forecast.py          # forecast 6 months ahead
    python src/forecast.py 12       # forecast 12 months ahead
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import build_features, holiday_months  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def forecast(months_ahead=6):
    """Return a DataFrame with Date and Forecast for the next `months_ahead` months."""
    model = joblib.load(os.path.join(BASE, "models", "sales_forecast_rf.pkl"))
    meta = json.load(open(os.path.join(BASE, "models", "model_meta.json")))
    history = pd.read_csv(os.path.join(BASE, "data", "monthly_sales.csv"),
                          parse_dates=["Date"])

    feature_cols = meta["feature_cols"]
    last_date = history["Date"].max()

    dates = [last_date + pd.offsets.MonthEnd(i) for i in range(1, months_ahead + 1)]
    years = set(history["Date"].dt.year) | {d.year for d in dates}
    hmonths = holiday_months(years)

    hist = history["Sales"].values.tolist()
    rows = []
    for d in dates:
        p = float(model.predict(build_features(np.array(hist), d, feature_cols, hmonths))[0])
        rows.append({"Date": d, "Forecast": round(p, 2)})
        hist.append(p)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    h = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    out = forecast(h)
    out = out.copy()
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    print(out.to_string(index=False))
