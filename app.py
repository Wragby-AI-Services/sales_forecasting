"""Streamlit app — monthly sales forecasting.

Run:
    streamlit run app.py
"""

import os
import sys

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from forecast import forecast  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
HISTORY = os.path.join(BASE, "data", "monthly_sales.csv")

st.set_page_config(page_title="Sales Forecast", layout="wide")
st.title(" Monthly Sales Forecast")

history = pd.read_csv(HISTORY, parse_dates=["Date"])


def line_chart(df_wide, height=420):
    """Altair line chart whose X-axis is labelled by year only."""
    long = df_wide.reset_index().melt(
        id_vars="Date", var_name="Series", value_name="Value"
    )
    # Explicit ticks: exactly one per calendar year, placed at that year's
    # first data point. This stops the same year from repeating across the
    # monthly ticks on the temporal axis.
    year_ticks = [
        df_wide.index[df_wide.index.year == year].min()
        for year in sorted(df_wide.index.year.unique())
    ]
    return (
        alt.Chart(long)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "Date:T",
                axis=alt.Axis(format="%Y", title="Year", values=year_ticks),
            ),
            y=alt.Y("Value:Q", title="Sales ($)"),
            color=alt.Color("Series:N", title=None),
            tooltip=[
                alt.Tooltip("Date:T", title="Month", format="%b %Y"),
                alt.Tooltip("Value:Q", title="Sales", format=",.0f"),
            ],
        )
        .properties(height=height)
    )


def interpret_forecast(history, out):
    """Return markdown bullet strings interpreting the forecast table."""
    n = len(out)
    total = out["Forecast"].sum()
    avg = out["Forecast"].mean()

    peak = out.loc[out["Forecast"].idxmax()]
    trough = out.loc[out["Forecast"].idxmin()]

    recent_avg = history["Sales"].tail(6).mean()
    vs_recent = (avg - recent_avg) / recent_avg * 100 if recent_avg else 0.0

    bullets = [
        f"**Total forecast:** ${total:,.0f} across the next {n} month(s)."
    ]

    if n > 1:
        first = out["Forecast"].iloc[0]
        last = out["Forecast"].iloc[-1]
        change_pct = (last - first) / abs(first) * 100 if first else 0.0
        diffs = out["Forecast"].diff().dropna()
        up = int((diffs > 0).sum())
        down = int((diffs < 0).sum())
        
    return bullets


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
    st.altair_chart(line_chart(chart_df), use_container_width=True)
    st.dataframe(
        out.set_index("Date"),
        width="content",
        column_config={
            "Date": st.column_config.DateColumn(
                "Date", format="MMM YYYY", width="small"
            ),
            "Forecast": st.column_config.NumberColumn(
                "Forecast ($)", format="%,.2f", width="small"
            ),
        },
    )

    csv_data = out.assign(
        Date=out["Date"].dt.strftime("%Y-%m-%d")
    ).to_csv(index=False)
    st.download_button(
        "Download forecast (CSV)",
        data=csv_data,
        file_name=f"sales_forecast_{months}_months.csv",
        mime="text/csv",
    )

    st.subheader("Interpretation")
    for bullet in interpret_forecast(history, out):
        st.markdown(f"- {bullet}")
else:
    st.subheader("Historical monthly sales")
    hist_df = history.set_index("Date").rename(columns={"Sales": "Sales"})
    st.altair_chart(line_chart(hist_df), use_container_width=True)
