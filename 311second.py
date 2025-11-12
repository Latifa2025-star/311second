# 311.py — NYC 311 Service Requests Explorer (single file)
# -------------------------------------------------------
# Works with a compressed CSV placed next to this file:
#   nyc311_12months.csv.gz
#
# Streamlit Cloud friendly. Fancy, readable plots.
# -------------------------------------------------------

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# App config / theming
# -----------------------------
st.set_page_config(
    page_title="NYC 311 Service Requests Explorer",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#5A55FF"
ACCENT = "#F95D6A"

# Global plot style
BASE_LAYOUT = dict(
    template="simple_white",
    font=dict(size=14),
    margin=dict(t=60, r=20, l=20, b=60),
)


# -----------------------------
# Data loading & prep
# -----------------------------
DATA_PATHS = [
    "nyc311_12months.csv.gz",           # preferred (your file)
    "nyc311_sample.csv.gz",             # optional fallback
    "nyc311_12months.csv",              # uncompressed fallbacks
    "nyc311_sample.csv",
]

@st.cache_data(show_spinner=True)
def load_data() -> pd.DataFrame:
    path = None
    for p in DATA_PATHS:
        if os.path.exists(p):
            path = p
            break
    if path is None:
        raise FileNotFoundError(
            "No dataset found. Please add nyc311_12months.csv.gz (or a sample CSV) "
            "to the same folder as this app."
        )

    kwargs = {}
    if path.endswith(".gz"):
        kwargs["compression"] = "gzip"

    df = pd.read_csv(path, **kwargs)

    # Parse times (tolerant)
    for c in ("created_date", "closed_date", "resolution_action_updated_date"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # Derive hours_to_close if missing
    if "hours_to_close" not in df.columns and {"created_date", "closed_date"}.issubset(df.columns):
        delta = (df["closed_date"] - df["created_date"]).dt.total_seconds() / 3600.0
        df["hours_to_close"] = delta

    # Canonical helpers
    if "created_date" in df.columns:
        df["day_of_week"] = df["created_date"].dt.day_name()
        df["hour"] = df["created_date"].dt.hour

    # Normalize casing for status for better group-bys
    if "status" in df.columns:
        df["status"] = df["status"].astype(str)

    # Make borough clean
    if "borough" in df.columns:
        df["borough"] = df["borough"].fillna("Unspecified")

    return df, path


# -----------------------------
# Small utilities
# -----------------------------
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

def safe_number(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def filter_df(df: pd.DataFrame,
              day: str,
              hour_range: tuple[int, int],
              boroughs: list[str]) -> pd.DataFrame:
    out = df.copy()
    if day != "All" and "day_of_week" in out:
        out = out[out["day_of_week"] == day]

    if "hour" in out:
        lo, hi = hour_range
        out = out[(out["hour"] >= lo) & (out["hour"] <= hi)]

    if boroughs and "borough" in out and "All" not in boroughs:
        out = out[out["borough"].isin(boroughs)]

    return out


# -----------------------------
# Figure builders (robust)
# -----------------------------
def fig_top_types(df: pd.DataFrame, n: int) -> go.Figure:
    if df.empty or "complaint_type" not in df:
        return go.Figure().update_layout(BASE_LAYOUT | dict(title="No data for Top Complaint Types"))

    counts = (
        df["complaint_type"]
        .value_counts(dropna=False)
        .nlargest(n)
        .rename_axis("complaint_type")
        .reset_index(name="count")
    )

    fig = px.bar(
        counts,
        x="count", y="complaint_type",
        orientation="h",
        color="count",
        color_continuous_scale="Sunset",
        labels={"count": "Requests (count)", "complaint_type": "Complaint Type"},
        title=f"Top {n} Complaint Types",
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(BASE_LAYOUT | dict(coloraxis_showscale=False))
    return fig


def fig_status_donut(df: pd.DataFrame) -> go.Figure:
    if df.empty or "status" not in df:
        return go.Figure().update_layout(BASE_LAYOUT | dict(title="No status data"))

    s = df["status"].fillna("Unspecified").value_counts().reset_index()
    s.columns = ["status", "count"]
    fig = go.Figure(
        go.Pie(
            labels=s["status"], values=s["count"],
            hole=0.55, sort=False,
            marker=dict(colors=px.colors.qualitative.Set3),
            hovertemplate="Status: %{label}<br>Requests: %{value} (%{percent})<extra></extra>",
        )
    )
    fig.update_layout(BASE_LAYOUT | dict(title="Status Breakdown"))
    return fig


def fig_resolution_box(df: pd.DataFrame, top_n: int = 18) -> go.Figure:
    if df.empty or "hours_to_close" not in df or "complaint_type" not in df:
        return go.Figure().update_layout(BASE_LAYOUT | dict(title="No resolution-time data"))

    # Focus on top complaint types to keep x-axis readable
    tops = (
        df["complaint_type"].value_counts().nlargest(top_n).index
    )
    sub = df[df["complaint_type"].isin(tops)].copy()

    # Clip extreme outliers for readability (99th percentile per whole set)
    q99 = np.nanquantile(sub["hours_to_close"].astype(float), 0.99)
    sub = sub[sub["hours_to_close"] <= q99]

    fig = px.box(
        sub,
        x="complaint_type",
        y="hours_to_close",
        points=False,
        color_discrete_sequence=[PRIMARY],
        labels={"hours_to_close": "Hours to Close", "complaint_type": "Complaint Type"},
        title="Resolution Time by Complaint Type (clipped at 99th percentile)",
    )
    fig.update_layout(BASE_LAYOUT)
    fig.update_xaxes(tickangle=-35)
    return fig


def fig_day_hour_heatmap(df: pd.DataFrame) -> go.Figure:
    if df.empty or {"day_of_week", "hour"}.difference(df.columns):
        return go.Figure().update_layout(BASE_LAYOUT | dict(title="No day/hour data"))

    mat = (
        df.groupby(["day_of_week", "hour"])
        .size()
        .reset_index(name="requests")
        .pivot(index="day_of_week", columns="hour", values="requests")
        .reindex(DAY_ORDER)
        .fillna(0)
    )

    fig = px.imshow(
        mat,
        color_continuous_scale="YlOrRd",
        labels=dict(color="Number of Requests"),
        title="When are requests made? (Day × Hour)",
        aspect="auto",
    )
    # Better hover text
    fig.update_traces(
        hovertemplate="Day: %{y}<br>Hour: %{x}:00<br>Requests: %{z}<extra></extra>"
    )
    fig.update_layout(BASE_LAYOUT)
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(0, 24, 2)),
        ticktext=[f"{h:02d}" for h in range(0, 24, 2)],
        title="Hour of Day (24h)",
    )
    return fig


def fig_hour_animation(df: pd.DataFrame, top_k_types: int = 6) -> go.Figure:
    """Animated bar chart: counts by complaint_type across hours of day."""
    if df.empty or {"hour", "complaint_type"}.difference(df.columns):
        return go.Figure().update_layout(BASE_LAYOUT | dict(title="No hourly animation data"))

    # Keep only top overall types to keep frames readable
    keep = df["complaint_type"].value_counts().nlargest(top_k_types).index
    sub = df[df["complaint_type"].isin(keep)].copy()

    g = (
        sub.groupby(["hour", "complaint_type"])
        .size()
        .reset_index(name="requests")
    )

    max_x = int(g["requests"].max() * 1.15)

    fig = px.bar(
        g, x="requests", y="complaint_type",
        color="complaint_type",
        animation_frame="hour",
        orientation="h",
        range_x=[0, max_x],
        color_discrete_sequence=px.colors.qualitative.Set2,
        labels={"requests": "Requests (count)", "complaint_type": "Complaint Type", "hour": "Hour"},
        title="How requests evolve through the day (press ▶ to play)",
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(BASE_LAYOUT | dict(transition={"duration": 300}))
    return fig


# -----------------------------
# Sidebar (filters)
# -----------------------------
st.sidebar.header("Filters")

df, used_path = load_data()
st.sidebar.success(f"Loaded data from **{used_path}**")

day = st.sidebar.selectbox("Day of Week", options=["All"] + DAY_ORDER)
hour_range = st.sidebar.slider("Hour range (24h)", 0, 23, (0, 23))
if "borough" in df.columns:
    all_boroughs = ["All"] + sorted(df["borough"].dropna().unique().tolist())
    boroughs = st.sidebar.multiselect("Borough(s)", options=all_boroughs, default=["All"])
else:
    boroughs = ["All"]

top_n = st.sidebar.slider("Top complaint types to show", 5, 30, 20)

# Apply filters
df_f = filter_df(df, day=day, hour_range=hour_range, boroughs=boroughs)

# -----------------------------
# Header + KPIs
# -----------------------------
st.title("📞 NYC 311 Service Requests Explorer")
st.caption("Explore complaint types, resolution times, and closure rates by day and hour — powered by a compressed local dataset (.csv.gz).")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Rows (after filters)", f"{len(df_f):,}")
if "status" in df_f.columns:
    pct_closed = (df_f["status"].str.lower().eq("closed")).mean() * 100 if len(df_f) else 0
    k2.metric("% Closed", f"{pct_closed:.1f}%")
else:
    k2.metric("% Closed", "—")

if "hours_to_close" in df_f.columns:
    med_hours = df_f["hours_to_close"].median()
    k3.metric("Median Hours to Close", f"{med_hours:,.2f}" if pd.notnull(med_hours) else "—")
else:
    k3.metric("Median Hours to Close", "—")

if "complaint_type" in df_f.columns and not df_f.empty:
    top_type = df_f["complaint_type"].mode().iloc[0]
else:
    top_type = "—"
k4.metric("Top Complaint Type", top_type)

st.markdown("---")

# -----------------------------
# Storytelling blocks + charts
# -----------------------------

# 1) Top complaint types + status donut
st.subheader("Top Complaint Types")
st.markdown("**What issues are most frequently reported under the current filters?**")
c1, c2 = st.columns([2, 1])
with c1:
    st.plotly_chart(fig_top_types(df_f, top_n), use_container_width=True)
with c2:
    st.plotly_chart(fig_status_donut(df_f), use_container_width=True)

# 2) Resolution-time box plot (with narrative)
st.subheader("How long do different complaints take to resolve?")
st.markdown(
    "This box plot shows the distribution of **hours to close** by complaint type. "
    "The **box** captures the middle 50% of cases (IQR), the line inside is the **median**, "
    "and whiskers extend to typical values. We clip extreme outliers at the 99th percentile "
    "to keep the chart readable."
)
st.plotly_chart(fig_resolution_box(df_f, top_n=min(18, top_n)), use_container_width=True)

# 3) Day × hour heatmap (with better hover)
st.subheader("When are requests made? (Day × Hour)")
st.markdown(
    "Heat intensity indicates how many requests were made. "
    "Hover a square to see the **Number of Requests** for that day & hour."
)
st.plotly_chart(fig_day_hour_heatmap(df_f), use_container_width=True)

# 4) Animated “through the day” story
st.subheader("How do complaints evolve through the day?")
st.markdown(
    "Press **▶ Play** to watch requests change by hour. "
    "We focus on the top complaint categories so the animation remains clear."
)
st.plotly_chart(fig_hour_animation(df_f, top_k_types=6), use_container_width=True)

st.markdown("---")
st.caption(
    "Tip: use the filters on the left (Day, Hour range, Borough) to change the story. "
    "All visualizations update instantly on your compressed local dataset."
)
