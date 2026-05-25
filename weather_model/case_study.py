import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import os
from xgboost import XGBRegressor

_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(_DIR), "timelapse_tool", "Data")

st.set_page_config(page_title="Post-Intervention Case Studies", layout="wide")
st.title("🔬 Post-Intervention Case Studies")
st.markdown(
    "<p style='color:grey;font-size:16px;'>"
    "Comparing weather-normalized predicted cyclist counts against observed counts "
    "to isolate the effect of urban circulation plan changes in Aalst and Kortrijk."
    "</p>",
    unsafe_allow_html=True
)

# ── Constants ────────────────────────────────────────────────────────────────

AALST_INTERVENTION   = pd.Timestamp("2021-08-16")
KORTRIJK_TRIAL       = pd.Timestamp("2022-07-01")   # approximate — trial barricades
KORTRIJK_PERMANENT   = pd.Timestamp("2023-10-01")   # permanent plan locked in

CITY_CONFIG = {
    "Aalst": {
        "sites": {10: "Aalst 1", 11: "Aalst 2", 19: "Aalst 3"},
        "coords": {10: (50.93433, 4.01471), 11: (50.93549, 4.01571), 19: (50.93385, 4.01647)},
        "weather_coords": (50.9344, 4.0159),   # central Aalst
        "interventions": [
            {"date": AALST_INTERVENTION, "label": "Circulation plan — Aug 16, 2021", "dash": "solid", "color": "crimson"},
        ],
        "pre_cutoff": AALST_INTERVENTION,
        "note": None,
    },
    "Kortrijk": {
        "sites": {16: "Kortrijk 1", 17: "Kortrijk 2"},
        "coords": {16: (50.83332, 3.27777), 17: (50.83339, 3.27773)},
        "weather_coords": (50.8276, 3.2647),   # central Kortrijk
        "interventions": [
            {"date": KORTRIJK_TRIAL,     "label": "Trial phase — Summer 2022",     "dash": "dash",  "color": "orange"},
            {"date": KORTRIJK_PERMANENT, "label": "Permanent plan — Oct 2023",      "dash": "solid", "color": "crimson"},
        ],
        "pre_cutoff": KORTRIJK_TRIAL,
        "note": (
            "⚠️ Kortrijk teller 1 & 2 (installed July 2022) were placed specifically "
            "to monitor the trial and have no pre-intervention baseline — excluded from this analysis."
        ),
    },
}

MODEL_FEATURES = [
    "lat", "lon", "hour", "day_of_week",
    "month", "temperature", "humidity", "precipitation",
    "wind_speed", "cloud_cover"
]

# ── Data loading ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_model() -> XGBRegressor:
    m = XGBRegressor()
    m.load_model(os.path.join(_DIR, "weather_bike_model.ubj"))
    return m


@st.cache_data(show_spinner="Loading cycling data…")
def load_cycling_data(site_ids: tuple) -> pd.DataFrame:
    """Load all parquet files, filter to selected sites and cyclists, aggregate to hourly."""
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".parquet"))
    dfs = []
    for f in files:
        df = pd.read_parquet(
            os.path.join(DATA_DIR, f),
            columns=["site_ID", "direction", "type", "time_from", "count"]
        )
        df = df[df["site_ID"].isin(site_ids) & (df["type"] == "FIETSERS")]
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df["time_from"] = pd.to_datetime(df["time_from"]).dt.floor("h")
    df = (
        df.groupby(["site_ID", "time_from"])["count"]
        .sum()
        .reset_index()
        .rename(columns={"site_ID": "site_id", "time_from": "hour_timestamp", "count": "bike_count"})
    )
    return df


@st.cache_data(show_spinner="Fetching weather data from Open-Meteo…")
def fetch_weather(start_date: str, end_date: str, lat: float, lon: float) -> pd.DataFrame:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start_date, "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,cloud_cover",
        "timezone": "Europe/Brussels",
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()["hourly"]
    return pd.DataFrame({
        "hour_timestamp": pd.to_datetime(data["time"]),
        "temperature":    data["temperature_2m"],
        "humidity":       data["relative_humidity_2m"],
        "precipitation":  data["precipitation"],
        "wind_speed":     data["wind_speed_10m"],
        "cloud_cover":    data["cloud_cover"],
    })


# ── UI ────────────────────────────────────────────────────────────────────────

city = st.sidebar.radio("City", list(CITY_CONFIG.keys()))
cfg  = CITY_CONFIG[city]

if cfg["note"]:
    st.info(cfg["note"])

site_options = cfg["sites"]
selected_ids = st.sidebar.multiselect(
    "Monitoring sites",
    options=list(site_options.keys()),
    default=list(site_options.keys()),
    format_func=lambda x: site_options[x],
)

smoothing = st.sidebar.selectbox(
    "Smoothing", ["Daily", "Weekly (7d)", "Monthly (30d)"], index=1
)
smoothing_map = {"Daily": 1, "Weekly (7d)": 7, "Monthly (30d)": 30}
window = smoothing_map[smoothing]

if not selected_ids:
    st.warning("Select at least one monitoring site.")
    st.stop()

# ── Load & predict ────────────────────────────────────────────────────────────

model   = load_model()
cycling = load_cycling_data(tuple(sorted(selected_ids)))

# date range from actual data
start = cycling["hour_timestamp"].min().strftime("%Y-%m-%d")
end   = cycling["hour_timestamp"].max().strftime("%Y-%m-%d")

weather = fetch_weather(start, end, *cfg["weather_coords"])

# merge cycling + weather
merged = cycling.merge(weather, on="hour_timestamp", how="left")

# add temporal features and coordinates
merged["hour"]       = merged["hour_timestamp"].dt.hour
merged["day_of_week"]= merged["hour_timestamp"].dt.dayofweek
merged["month"]      = merged["hour_timestamp"].dt.month

coord_map = cfg["coords"]
merged["lat"] = merged["site_id"].map(lambda x: coord_map[x][0])
merged["lon"] = merged["site_id"].map(lambda x: coord_map[x][1])

merged = merged.dropna(subset=MODEL_FEATURES)
merged["predicted"] = model.predict(merged[MODEL_FEATURES]).clip(min=0)

# ── Charts ────────────────────────────────────────────────────────────────────

st.markdown("---")

for site_id in selected_ids:
    site_name = site_options[site_id]
    df = merged[merged["site_id"] == site_id].copy().sort_values("hour_timestamp")

    # aggregate to daily totals then apply rolling
    daily = (
        df.set_index("hour_timestamp")[["bike_count", "predicted"]]
        .resample("D").sum()
        .rolling(window, min_periods=1, center=True).mean()
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=daily.index, y=daily["bike_count"],
        name="Actual", line=dict(color="#1f77b4", width=1.5)
    ))
    fig.add_trace(go.Scatter(
        x=daily.index, y=daily["predicted"],
        name="Predicted (weather baseline)",
        line=dict(color="#ff7f0e", width=1.5, dash="dot")
    ))

    # divergence fill
    fig.add_trace(go.Scatter(
        x=pd.concat([daily.index.to_series(), daily.index.to_series()[::-1]]),
        y=pd.concat([daily["bike_count"], daily["predicted"][::-1]]),
        fill="toself",
        fillcolor="rgba(200,200,200,0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        showlegend=False, hoverinfo="skip",
    ))

    # intervention lines
    for iv in cfg["interventions"]:
        fig.add_vline(
            x=iv["date"].timestamp() * 1000,
            line_dash=iv["dash"],
            line_color=iv["color"],
            line_width=2,
            annotation_text=iv["label"],
            annotation_position="top left",
            annotation_font_size=11,
        )

    fig.update_layout(
        title=f"{site_name} — Actual vs Weather-Normalised Baseline",
        xaxis_title="Date",
        yaxis_title=f"Cyclist count ({smoothing.lower()} avg)",
        legend=dict(orientation="h", y=1.08),
        hovermode="x unified",
        height=400,
        margin=dict(t=80, b=40),
    )

    st.plotly_chart(fig, width="stretch")

    # divergence metric post-intervention
    cutoff = cfg["pre_cutoff"]
    post = df[df["hour_timestamp"] >= cutoff]
    if not post.empty:
        actual_total    = post["bike_count"].sum()
        predicted_total = post["predicted"].sum()
        delta_pct = (actual_total - predicted_total) / predicted_total * 100 if predicted_total else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("Actual counts (post-intervention)",    f"{int(actual_total):,}")
        col2.metric("Predicted baseline (post-intervention)", f"{int(predicted_total):,}")
        col3.metric("Divergence from baseline", f"{delta_pct:+.1f}%",
                    delta_color="normal" if delta_pct >= 0 else "inverse")

    st.markdown("---")
