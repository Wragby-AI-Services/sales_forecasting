"""Finalize: retrain the chosen model (Random Forest) on all data and save artifacts.

Artifacts produced:
  models/sales_forecast_rf.pkl   -> trained model
  models/model_meta.json         -> model name, params, feature list
"""

import json
import os
import sys
import warnings

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_feature_cols  # noqa: E402

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)  # run relative to the project root

features = pd.read_csv("data/features.csv", parse_dates=["Date"])
feature_cols = load_feature_cols()

# Tuned via GridSearchCV + TimeSeriesSplit (see src/tuning.py)
PARAMS = {
    "n_estimators": 400,
    "max_depth": 3,
    "min_samples_leaf": 1,
    "max_features": 0.8,
    "random_state": 42,
}

final_model = RandomForestRegressor(**PARAMS).fit(features[feature_cols], features["Sales"])
print(f"Final model trained on {len(features)} feature rows "
      f"({features['Date'].min():%Y-%m} .. {features['Date'].max():%Y-%m})")

os.makedirs("models", exist_ok=True)
joblib.dump(final_model, "models/sales_forecast_rf.pkl")

meta = {
    "model": "RandomForestRegressor",
    "params": PARAMS,
    "feature_cols": feature_cols,
    "train_rows": int(len(features)),
}
with open("models/model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("Saved: models/sales_forecast_rf.pkl, models/model_meta.json")
