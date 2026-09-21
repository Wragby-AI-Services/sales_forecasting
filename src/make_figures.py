"""Regenerate the figures/ folder from the current data.

Run from anywhere:
    python src/make_figures.py

Outputs (figures/):
    sales distribution.png
    sales boxplot.png
    Daily total sales.png
    monthly total sales.png
    Total Sales by Month.png
    total sales by week.png
    total sales by year.png
    total sales by category.png
    total sales by sub-category.png
    total sales by segment.png
    total sales by region.png
    correlation heatmap.png
    rolling mean and std.png
    additive sales-seasonal-residual.png
    multiplicative sales-seasonal-residual.png
    autocorrelation-partialautocorrelation.png
    first differenced monthly sales.png
"""

import os
import warnings

import matplotlib

matplotlib.use("Agg")  # headless / CI-safe

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)

FIG = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110


def savefig(name):
    """Tight-layout and save a PNG into figures/, then close the figure."""
    plt.tight_layout()
    path = os.path.join(FIG, name)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved figures/{name}")


# ----------------------------------------------------------------------------
# 0. Load current data (same parsing as src/preprocess.py)
# ----------------------------------------------------------------------------
df = pd.read_csv(
    "data/raw.csv", parse_dates=["Order Date", "Ship Date"], dayfirst=True
)
daily = df.groupby("Order Date")["Sales"].sum().reset_index()
monthly = (
    df.set_index("Order Date").resample("ME")["Sales"].sum().reset_index()
)
monthly.columns = ["Date", "Sales"]
monthly_series = monthly.set_index("Date")["Sales"]

# ----------------------------------------------------------------------------
# 1. Target variable
# ----------------------------------------------------------------------------
plt.figure(figsize=(10, 4))
sns.histplot(df["Sales"], bins=50, kde=True)
plt.title("Sales Distribution")
plt.xlabel("Order Sales ($)")
savefig("sales distribution.png")

plt.figure(figsize=(10, 4))
sns.boxplot(x=df["Sales"])
plt.title("Sales Boxplot")
plt.xlabel("Order Sales ($)")
savefig("sales boxplot.png")

# ----------------------------------------------------------------------------
# 2. Sales over time
# ----------------------------------------------------------------------------
plt.figure(figsize=(14, 5))
plt.plot(daily["Order Date"], daily["Sales"], linewidth=0.9)
plt.title("Daily Total Sales")
plt.xlabel("Date")
plt.ylabel("Sales ($)")
savefig("Daily total sales.png")

plt.figure(figsize=(14, 5))
plt.plot(monthly["Date"], monthly["Sales"], linewidth=1.5, color="#1f77b4")
plt.title("Monthly Total Sales")
plt.xlabel("Date")
plt.ylabel("Sales ($)")
savefig("monthly total sales.png")

# ----------------------------------------------------------------------------
# 3. Seasonality / calendar aggregation
# ----------------------------------------------------------------------------
plt.figure(figsize=(10, 4))
sns.barplot(
    x=monthly["Date"].dt.month,
    y=monthly["Sales"],
    estimator=sum,
    errorbar=None,
    color="#1f77b4",
)
plt.title("Total Sales by Month (all years combined)")
plt.xlabel("Month")
plt.ylabel("Sales ($)")
savefig("Total Sales by Month.png")

plt.figure(figsize=(12, 4))
week = df["Order Date"].dt.isocalendar().week.astype(int)
weekly = df.groupby(week)["Sales"].sum()
sns.barplot(x=weekly.index, y=weekly.values, color="#1f77b4")
plt.title("Total Sales by Week")
plt.xlabel("ISO Week")
plt.ylabel("Sales ($)")
savefig("total sales by week.png")

plt.figure(figsize=(10, 4))
sns.barplot(
    x=monthly["Date"].dt.year,
    y=monthly["Sales"],
    estimator=sum,
    errorbar=None,
    color="#1f77b4",
)
plt.title("Total Sales by Year")
plt.xlabel("Year")
plt.ylabel("Sales ($)")
savefig("total sales by year.png")

# ----------------------------------------------------------------------------
# 4. Categorical breakdowns
# ----------------------------------------------------------------------------
for col in ["Category", "Sub-Category", "Segment", "Region"]:
    plt.figure(figsize=(10, 4))
    agg = df.groupby(col)["Sales"].sum().sort_values(ascending=False)
    sns.barplot(x=agg.values, y=agg.index, color="#1f77b4")
    plt.title(f"Total Sales by {col}")
    plt.xlabel("Sales ($)")
    plt.ylabel(col)
    savefig(f"total sales by {col.lower()}.png")

# ----------------------------------------------------------------------------
# 5. Correlation heatmap (numeric columns only)
# ----------------------------------------------------------------------------
numeric_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(9, 7))
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5)
plt.title("Correlation Heatmap")
savefig("correlation heatmap.png")

# ----------------------------------------------------------------------------
# 6. Rolling statistics (stationarity assessment)
# ----------------------------------------------------------------------------
rolling_mean = monthly_series.rolling(window=3).mean()
rolling_std = monthly_series.rolling(window=3).std()

plt.figure(figsize=(14, 5))
plt.plot(monthly_series, label="Monthly Sales", linewidth=1.5)
plt.plot(rolling_mean, label="Rolling Mean (3mo)", linewidth=2)
plt.plot(rolling_std, label="Rolling Std (3mo)", linewidth=2)
plt.title("Rolling Mean & Std")
plt.xlabel("Date")
plt.ylabel("Sales ($)")
plt.legend()
savefig("rolling mean and std.png")

# ----------------------------------------------------------------------------
# 7. Time-series decomposition
# ----------------------------------------------------------------------------
for model in ["additive", "multiplicative"]:
    decomp = seasonal_decompose(monthly_series, model=model, period=12)
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    decomp.observed.plot(ax=axes[0], linewidth=1.2)
    axes[0].set_ylabel("Observed")
    decomp.trend.plot(ax=axes[1], linewidth=1.2)
    axes[1].set_ylabel("Trend")
    decomp.seasonal.plot(ax=axes[2], linewidth=1.2)
    axes[2].set_ylabel("Seasonal")
    decomp.resid.plot(ax=axes[3], linewidth=1.2, color="#7f7f7f")
    axes[3].set_ylabel("Residual")
    axes[3].set_xlabel("Date")
    fig.suptitle(f"{model.capitalize()} Decomposition — Sales, Seasonal, Residual",
                 y=1.0)
    savefig(f"{model} sales-seasonal-residual.png")

# ----------------------------------------------------------------------------
# 8. ACF / PACF
# ----------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
plot_acf(monthly_series, lags=24, ax=ax1, zero=False)
ax1.set_title("Autocorrelation (ACF)")
plot_pacf(monthly_series, lags=24, ax=ax2, zero=False)
ax2.set_title("Partial Autocorrelation (PACF)")
savefig("autocorrelation-partialautocorrelation.png")

# ----------------------------------------------------------------------------
# 9. First differenced series
# ----------------------------------------------------------------------------
plt.figure(figsize=(14, 5))
plt.plot(monthly_series.diff().dropna(), linewidth=1.5, color="#2ca02c")
plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.title("First Differenced Monthly Sales")
plt.xlabel("Date")
plt.ylabel("Δ Sales ($)")
savefig("first differenced monthly sales.png")

print("\nAll figures regenerated in figures/")
