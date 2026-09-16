"""Streamlit app — monthly sales forecasting.

Run:
    streamlit run app.py
"""

import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from forecast import forecast  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
HISTORY = os.path.join(BASE, "data", "monthly_sales.csv")

st.set_page_config(page_title="Sales Forecast", layout="wide")
st.title("📈 Monthly Sales Forecast")
st.caption("Random Forest · walk-forward tuned · 48 months of history (2015–2018)")

history = pd.read_csv(HISTORY, parse_dates=["Date"])

months = st.slider("Months to forecast ahead", min_value=1, max_value=24, value=6)

if st.button("Generate forecast", type="primary"):
    out = forecast(months)

    st.subheader(f"Forecast — next {months} month(s)")
    chart_df = pd.concat(
        [
            history.set_index("Date").rename(columns={"Sales": "History"}),
            out.set_index("Date").rename(columns={"Forecast": "Forecast"}),
        ],
        axis=1,
    )
    st.line_chart(chart_df)
    st.dataframe(out.set_index("Date"), use_container_width=True)
else:
    st.subheader("Historical monthly sales")
    st.line_chart(history.set_index("Date").rename(columns={"Sales": "Sales"}))
