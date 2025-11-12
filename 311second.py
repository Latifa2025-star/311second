# 311.py — NYC 311 Service Requests Explorer (Streamlit)
# ------------------------------------------------------
# Features
# - Loads compressed CSV (.csv.gz) shipped with the repo
# - Sidebar filters (day, hour window, boroughs, top N)
# - KPIs + narratives to guide the viewer
# - Top complaint types (bar) + Status donut
# - Resolution-time box plot with explanation
# - Animated “day progression” chart (play button)
# - Heatmap (Day × Hour) with hover label = "Requests" (not "color")
# - Consistent large fonts for presentation

from __future__ import annotations
import os
from typing import Tuple, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# --------------------------- Config & Theme ---------------------------
st.set_page_config(
    page_title="NYC 311 Service Requests Explorer",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#5b5bd6"
ACCENT = "#f59e0b"
GOOD = "#10b981"
BAD = "#ef4444"

FONT_SIZE = 16  # global font size

def style_fig(fig: go.Figure, height: int = 480, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height,
        font=dict(size=FONT_SIZE),
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ) if legend else dict(visible=False),
    )
    return fig


# --------------------------- Data Loading ----------------------------
@st.cache_data(show_spinner=False)
def load_local_csv() -> Tuple[pd.DataFrame, str]:
    """
    Load a gzipped CSV that ships with the repo.
    Returns (DataFrame, file_used)
    """
    candidates = ["nyc311_12months.csv.gz", "nyc311_sample.csv.gz", "nyc311_12months.csv", "nyc311_sample.csv"]
    for f in candidates:
        if os.path.exists(f):
            if f.endswith(".gz"):
                df = pd.read_csv(f, compression="gzip", low_memory=False)
            else:
                df = pd.read_csv(f, low_memory=False)
            return df, f
    raise FileNotFoundError(
        "Could not find any of: nyc311_12months.csv.gz, nyc311_sample.csv.gz, "
        "nyc311_12months.csv, nyc311_sample.csv. Place one next to 311.py."
    )


@st.cache_data(show_spinner=False)
def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # Keep common columns if present
    keep_cols = [
        "created_date","closed_date","status","agency_name",
        "complaint_type","descriptor","borough","incident_zip",
        "latitude","longitude","resolution_description",
        "resolution_action_updated_date"
    ]
    cols = [c for c in keep_cols if c in df.columns]
    df = df[cols].copy()

    # Types
    for c in ["created_date", "closed_date", "resolution_action_updated_date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # hours_to_close
    if {"created_date","closed_date"}.issubset(df.columns):
        df["hours_to_close"] = (df["closed_date"] - df["created_date"]).dt.total_seconds() / 3600.0

    # Helpers
    if "status" in df.columns:
        df["status"] = df["status"].astype(str)
        df["is_closed"] = df["status"].str.lower().eq("closed")

    if "borough" in df.columns:
        df["borough"] = df["borough"].fillna("Unspecified")

    # Day/hour for slicing + heatmap
    if "created_date" in df.columns:
        df["day_of_week"] = df["created_date"].dt.day_name()
        df["hour"] = df["created_date"].dt.hour
        df["date"] = df["created_date"].dt.date

    return df


# ------------------------------ UI ----------------------------------
st.title("📞 NYC 311 Service Requests Explorer")
st.caption(
    "Explore complaint types, resolution times, and closure rates by day and hour — "
    "powered by a compressed local dataset (.csv.gz)."
)

with st.spinner("Loading data…"):
    raw, file_used = load_local_csv()
    df = prepare(raw)

st.success(f"Loaded {len(df):,} rows from **{file_used}**")

# -------------------------- Sidebar Filters --------------------------
st.sidebar.header("Filters")
days = ["All"] + sorted(df["day_of_week"].dropna().unique().tolist()) if "day_of_week" in df else ["All"]
sel_day = st.sidebar.selectbox("Day of Week", days, index=0)

hour_min, hour_max = st.sidebar.slider("Hour range (24h)", 0, 23, (0, 23))
boroughs_all = sorted(df["borough"].dropna().unique().tolist()) if "borough" in df else []
sel_boroughs: List[str] = st.sidebar.multiselect("Borough(s)", options=boroughs_all, default=boroughs_all[:1] or [])
top_n = st.sidebar.slider("Top complaint types to show", 5, 25, 20)

# Apply filters
df_f = df.copy()
if sel_day != "All":
    df_f = df_f[df_f["day_of_week"] == sel_day]
df_f = df_f[(df_f["hour"] >= hour_min) & (df_f["hour"] <= hour_max)]
if sel_boroughs:
    df_f = df_f[df_f["borough"].isin(sel_boroughs)]

# ------------------------------ KPIs ---------------------------------
kcol1, kcol2, kcol3, kcol4 = st.columns(4)
kcol1.metric("Rows (after filters)", f"{len(df_f):,}")
pct_closed = (df_f["is_closed"].mean() * 100) if "is_closed" in df_f else np.nan
kcol2.metric("% Closed", f"{pct_closed:,.1f}%" if pd.notnull(pct_closed) else "—")
median_hours = df_f["hours_to_close"].median() if "hours_to_close" in df_f else np.nan
kcol3.metric("Median Hours to Close", f"{median_hours:,.2f}" if pd.notnull(median_hours) else "—")
top_ct = df_f["complaint_type"].mode().iloc[0] if "complaint_type" in df_f and not df_f["complaint_type"].empty else "—"
kcol4.metric("Top Complaint Type", top_ct)

st.markdown(
    f"**Narrative:** With the current filters (day = `{sel_day}`; hours = `{hour_min}–{hour_max}`; "
    f"boroughs = `{', '.join(sel_boroughs) if sel_boroughs else 'All'}`), we’re seeing "
    f"{len(df_f):,} requests. About **{pct_closed:,.1f}%** are closed and the median time to close is "
    f"**{median_hours:,.2f} hours**."
)

st.markdown("---")

# ----------------------- Top complaint types -------------------------
st.subheader("Top Complaint Types")
st.caption("What issues are most frequently reported under the current filters?")

if "complaint_type" in df_f:
    counts = (
        df_f["complaint_type"]
        .value_counts()
        .nlargest(top_n)
        .reset_index()
        .rename(columns={"index": "Complaint Type", "complaint_type": "Requests"})
    )
    fig_bar = px.bar(
        counts,
        x="Requests",
        y="Complaint Type",
        orientation="h",
        color="Requests",
        color_continuous_scale=px.colors.sequential.Sunset,
        title=f"Top {top_n} Complaint Types",
    )
    fig_bar.update_yaxes(autorange="reversed")
    fig_bar = style_fig(fig_bar, height=520, legend=False)
else:
    fig_bar = go.Figure()
    fig_bar.add_annotation(text="complaint_type column not found", showarrow=False)

# ------------------------- Status donut ------------------------------
st.subheader("Status Breakdown")
st.caption("How many requests are Closed vs In Progress/Open/etc.?")
if "status" in df_f:
    status_counts = df_f["status"].value_counts().reset_index()
    status_counts.columns = ["Status", "Requests"]
    fig_donut = px.pie(
        status_counts, values="Requests", names="Status",
        hole=0.55, color_discrete_sequence=px.colors.qualitative.Set3,
        title="Status Breakdown"
    )
    fig_donut = style_fig(fig_donut, height=520)
else:
    fig_donut = go.Figure()
    fig_donut.add_annotation(text="status column not found", showarrow=False)

b1, b2 = st.columns([1.2, 1])
with b1:
    st.plotly_chart(fig_bar, use_container_width=True)
with b2:
    st.plotly_chart(fig_donut, use_container_width=True)

st.markdown(
    "**Narrative:** The left chart shows the **most common complaint categories**; "
    "the donut shows the **current status mix**. Large blue sections usually indicate "
    "a high percentage of completed/closed tickets."
)

st.markdown("---")

# --------------------- Resolution time — box plot --------------------
st.subheader("Resolution Time by Complaint Type (hours)")
st.caption(
    "A **box plot** shows the distribution of hours to close for each complaint type under your filters:\n"
    "- The **box** spans the middle 50% (from 25th to 75th percentile).\n"
    "- The **line inside** is the **median** time to close.\n"
    "- The **whiskers** extend to typical minimum/maximum values (outliers clipped here for readability)."
)

if "hours_to_close" in df_f and "complaint_type" in df_f:
    # clip to 99th percentile to make chart readable
    sub = df_f.dropna(subset=["hours_to_close"]).copy()
    if len(sub) > 0:
        q99 = sub["hours_to_close"].quantile(0.99)
        sub = sub[sub["hours_to_close"] <= q99]
        # only show for top N complaint types by count (reuse counts index)
        top_types = counts["Complaint Type"].tolist() if "counts" in locals() and not counts.empty else sub["complaint_type"].value_counts().head(top_n).index.tolist()
        sub = sub[sub["complaint_type"].isin(top_types)]
        fig_box = px.box(
            sub,
            x="complaint_type",
            y="hours_to_close",
            color="complaint_type",
            color_discrete_sequence=px.colors.qualitative.Vivid,
            title="Resolution time distribution (outliers clipped at 99th percentile)"
        )
        fig_box.update_xaxes(title=None, tickangle=45)
        fig_box.update_yaxes(title="Hours to Close")
        fig_box = style_fig(fig_box, height=520, legend=False)
        st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.info("No resolution-time data available after filters.")
else:
    st.info("`hours_to_close` or `complaint_type` columns not found.")

st.markdown("---")

# ----------------- Animated "through the day" chart ------------------
st.subheader("How requests accumulate through the day (play button)")
st.caption(
    "Press **Play** to watch requests by **hour**. We show counts per hour and color by status to hint "
    "what portion ends up closed vs still in progress at the moment of creation."
)

if {"hour","status"}.issubset(df_f.columns) and len(df_f) > 0:
    hourly = df_f.groupby(["hour","status"], as_index=False).size()
    hourly.rename(columns={"size":"Requests"}, inplace=True)
    # Use area to stack by status; animate by hour for smooth stepping
    fig_anim = px.bar(
        hourly.sort_values("hour"),
        x="status", y="Requests", color="status",
        animation_frame="hour", range_y=[0, max(1, hourly["Requests"].max()*1.1)],
        color_discrete_sequence=px.colors.qualitative.Set2,
        title="Requests by status across hours of the day"
    )
    fig_anim.update_xaxes(title="Status")
    fig_anim.update_yaxes(title="Requests")
    fig_anim = style_fig(fig_anim, height=520)
    st.plotly_chart(fig_anim, use_container_width=True)
else:
    st.info("Need `hour` and `status` columns for the animation.")

st.markdown("---")

# --------------------- Heatmap Day × Hour (Requests) -----------------
st.subheader("When are requests made? (Day × Hour)")
st.caption("Hover shows **Requests** (not color). Labels are large and clear for presentations.")

if {"day_of_week","hour"}.issubset(df_f.columns):
    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    heat = (
        df_f.groupby(["day_of_week","hour"], as_index=False)
        .size()
        .rename(columns={"size":"Requests"})
    )
    # pivot for imshow
    pivot = heat.pivot(index="day_of_week", columns="hour", values="Requests").reindex(order)
    # build custom hover text with 'Requests'
    hover_text = np.where(~pivot.isna(), "Requests: " + pivot.fillna(0).astype(int).astype(str), "")
    fig_hm = px.imshow(
        pivot,
        color_continuous_scale=px.colors.sequential.YlOrRd,
        aspect="auto",
        labels=dict(color="Requests"),
        title="Requests by Day and Hour (hover shows requests)"
    )
    # Replace default hovertemplate
    fig_hm.update_traces(
        hovertemplate="%{y} @ %{x}: <b>%{z:.0f}</b> Requests<extra></extra>"
    )
    fig_hm.update_xaxes(title="Hour of Day (24h)")
    fig_hm.update_yaxes(title="Day of Week")
    fig_hm = style_fig(fig_hm, height=560)
    st.plotly_chart(fig_hm, use_container_width=True)
else:
    st.info("Need day_of_week and hour columns for the heatmap.")

st.markdown(
    "> **Tip:** Use the filters on the left to focus by **day**, **hour window**, and **borough(s)**. "
    "All charts update instantly on your compressed local dataset."
)
