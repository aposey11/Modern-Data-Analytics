import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import os
from xgboost import XGBRegressor
from scipy import stats

_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(_DIR), "timelapse_tool", "Data")
SITES_CSV = os.path.join(os.path.dirname(_DIR), "timelapse_tool", "sites.csv")

st.set_page_config(page_title="Post-Intervention Case Studies", layout="wide")
st.title("🔬 Post-Intervention Case Studies")
st.markdown(
    "<p style='color:grey;font-size:16px;'>"
    "Comparing weather-normalised predicted cyclist counts against observed counts "
    "to isolate the effect of urban circulation plan changes in Aalst and Kortrijk."
    "</p>",
    unsafe_allow_html=True,
)

# ── Constants ─────────────────────────────────────────────────────────────────

AALST_INTERVENTION   = pd.Timestamp("2021-08-16")
KORTRIJK_TRIAL       = pd.Timestamp("2022-07-01")
KORTRIJK_PERMANENT   = pd.Timestamp("2023-10-01")

CITY_CONFIG = {
    "Aalst": {
        "sites":         {10: "Aalst 1", 11: "Aalst 2", 19: "Aalst 3"},
        "coords":        {10: (50.93433, 4.01471), 11: (50.93549, 4.01571), 19: (50.93385, 4.01647)},
        "weather_coords": (50.9344, 4.0159),
        "city_site_ids": [10, 11, 19],
        "pre_cutoff":    AALST_INTERVENTION,
        "interventions": [
            {"date": AALST_INTERVENTION, "label": "Circulation plan — Aug 16, 2021",
             "dash": "solid", "color": "crimson"},
        ],
        "note": None,
    },
    "Kortrijk": {
        "sites":         {16: "Kortrijk 1", 17: "Kortrijk 2"},
        "coords":        {16: (50.83332, 3.27777), 17: (50.83339, 3.27773)},
        "weather_coords": (50.8276, 3.2647),
        "city_site_ids": [16, 17],
        "pre_cutoff":    KORTRIJK_TRIAL,
        "interventions": [
            {"date": KORTRIJK_TRIAL,     "label": "Trial phase — Summer 2022",
             "dash": "dash",  "color": "orange"},
            {"date": KORTRIJK_PERMANENT, "label": "Permanent plan — Oct 2023",
             "dash": "solid", "color": "crimson"},
        ],
        "note": (
            "ℹ️ **About the Kortrijk phases**  \n"
            "**Trial phase (Jul 2022 – Sep 2023):** temporary bollards and signage redirected "
            "through-traffic away from the city centre, creating a low-traffic neighbourhood.  \n"
            "**Permanent plan (Oct 2023 onwards):** physical infrastructure changes made the "
            "plan permanent.  \n\n"
            "⚠️ The two monitoring sensors (Kortrijk 1 & 2) were installed in July 2022 "
            "specifically to monitor the trial — they have no pre-intervention baseline. "
            "The weather-normalised baseline is therefore derived from the model's spatial "
            "generalisation using all other Flemish sensors, not from historical Kortrijk data."
        ),
    },
}

MODEL_FEATURES = [
    "lat", "lon", "hour", "day_of_week", "month",
    "temperature", "humidity", "precipitation", "wind_speed", "cloud_cover",
]

# ── Data helpers ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_all_sites() -> pd.DataFrame:
    cols = ["site_id", "site_nr", "lon", "lat", "naam", "domein",
            "wegnr", "district", "gemeente", "interval", "datum_van"]
    sites = pd.read_csv(SITES_CSV, names=cols, header=None)
    return sites[["site_id", "lat", "lon"]].copy()


@st.cache_data(show_spinner="Loading cycling data…")
def load_all_cycling_data() -> pd.DataFrame:
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".parquet"))
    dfs = []
    for f in files:
        df = pd.read_parquet(
            os.path.join(DATA_DIR, f),
            columns=["site_ID", "direction", "type", "time_from", "count"],
        )
        dfs.append(df[df["type"] == "FIETSERS"])
    df = pd.concat(dfs, ignore_index=True)
    df["time_from"] = pd.to_datetime(df["time_from"]).dt.floor("h")
    return (
        df.groupby(["site_ID", "time_from"])["count"]
        .sum()
        .reset_index()
        .rename(columns={"site_ID": "site_id", "time_from": "hour_timestamp", "count": "bike_count"})
    )


@st.cache_data(show_spinner="Fetching weather data…")
def fetch_weather(start_date: str, end_date: str, lat: float, lon: float) -> pd.DataFrame:
    r = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": lat, "longitude": lon,
            "start_date": start_date, "end_date": end_date,
            "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,cloud_cover",
            "timezone": "Europe/Brussels",
        },
    )
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


# ── City-specific pre-intervention model ──────────────────────────────────────

@st.cache_resource(show_spinner="Training city-specific model (first load only)…")
def train_city_model(city: str) -> XGBRegressor:
    """
    Train on all Flemish sensors but exclude post-intervention data for the
    target city. This ensures the model learns that city's baseline behaviour
    from the pre-intervention period while still benefiting from the full
    Flanders dataset for generalisation.

    For Kortrijk: sensors were installed at trial start so city_pre is empty —
    the model learns Kortrijk purely from spatial/weather generalisation.
    """
    cfg           = CITY_CONFIG[city]
    city_site_ids = cfg["city_site_ids"]
    cutoff        = cfg["pre_cutoff"]

    all_cycling = load_all_cycling_data()
    all_sites   = load_all_sites()

    non_city = all_cycling[~all_cycling["site_id"].isin(city_site_ids)]
    city_pre = all_cycling[
        all_cycling["site_id"].isin(city_site_ids) &
        (all_cycling["hour_timestamp"] < cutoff)
    ]
    train_df = pd.concat([non_city, city_pre], ignore_index=True)

    start = train_df["hour_timestamp"].min().strftime("%Y-%m-%d")
    end   = train_df["hour_timestamp"].max().strftime("%Y-%m-%d")
    # Central Flanders weather for training (same approximation as general model)
    weather = fetch_weather(start, end, lat=51.05, lon=3.72)

    merged = (
        train_df
        .merge(weather, on="hour_timestamp", how="left")
        .merge(all_sites, on="site_id", how="left")
    )
    merged["hour"]        = merged["hour_timestamp"].dt.hour
    merged["day_of_week"] = merged["hour_timestamp"].dt.dayofweek
    merged["month"]       = merged["hour_timestamp"].dt.month
    merged = merged.dropna(subset=MODEL_FEATURES + ["bike_count"])

    model = XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, random_state=42, n_jobs=-1,
    )
    model.fit(merged[MODEL_FEATURES], merged["bike_count"])
    return model


# ── UI controls ───────────────────────────────────────────────────────────────

city = st.sidebar.radio("City", list(CITY_CONFIG.keys()))
cfg  = CITY_CONFIG[city]

if cfg["note"]:
    st.info(cfg["note"])

selected_ids = st.sidebar.multiselect(
    "Monitoring sites",
    options=list(cfg["sites"].keys()),
    default=list(cfg["sites"].keys()),
    format_func=lambda x: cfg["sites"][x],
)
smoothing = st.sidebar.selectbox(
    "Smoothing", ["Daily", "Weekly (7d)", "Monthly (30d)"], index=1
)
window = {"Daily": 1, "Weekly (7d)": 7, "Monthly (30d)": 30}[smoothing]

if not selected_ids:
    st.warning("Select at least one monitoring site.")
    st.stop()

# ── Load, predict ─────────────────────────────────────────────────────────────

model       = train_city_model(city)
all_cycling = load_all_cycling_data()
coord_map   = cfg["coords"]

city_df = all_cycling[all_cycling["site_id"].isin(selected_ids)].copy()
start   = city_df["hour_timestamp"].min().strftime("%Y-%m-%d")
end     = city_df["hour_timestamp"].max().strftime("%Y-%m-%d")

# City-specific weather for accurate predictions
weather = fetch_weather(start, end, *cfg["weather_coords"])

merged = city_df.merge(weather, on="hour_timestamp", how="left")
merged["hour"]        = merged["hour_timestamp"].dt.hour
merged["day_of_week"] = merged["hour_timestamp"].dt.dayofweek
merged["month"]       = merged["hour_timestamp"].dt.month
merged["lat"]         = merged["site_id"].map(lambda x: coord_map[x][0])
merged["lon"]         = merged["site_id"].map(lambda x: coord_map[x][1])
merged                = merged.dropna(subset=MODEL_FEATURES)
merged["predicted"]   = model.predict(merged[MODEL_FEATURES]).clip(min=0)

# ── Per-site charts ───────────────────────────────────────────────────────────

st.markdown("---")

site_daily = {}   # store for stats section below

for site_id in selected_ids:
    site_name = cfg["sites"][site_id]
    df = merged[merged["site_id"] == site_id].copy().sort_values("hour_timestamp")

    daily = (
        df.set_index("hour_timestamp")[["bike_count", "predicted"]]
        .resample("D").sum()
    )
    daily["residual"] = daily["bike_count"] - daily["predicted"]
    site_daily[site_id] = daily

    # Prediction interval from pre-intervention residuals
    pre_mask  = daily.index < cfg["pre_cutoff"]
    pre_resid = daily.loc[pre_mask, "residual"]
    if len(pre_resid) >= 10:
        lo_band = daily["predicted"] + pre_resid.quantile(0.05)
        hi_band = daily["predicted"] + pre_resid.quantile(0.95)
    else:
        lo_band = hi_band = None   # Kortrijk: no pre-intervention data

    # Apply smoothing
    sm = daily[["bike_count", "predicted"]].rolling(window, min_periods=1, center=True).mean()
    if lo_band is not None:
        lo_s = lo_band.rolling(window, min_periods=1, center=True).mean()
        hi_s = hi_band.rolling(window, min_periods=1, center=True).mean()

    fig = go.Figure()

    # Prediction interval band (only when pre-intervention data exists)
    if lo_band is not None:
        fig.add_trace(go.Scatter(
            x=pd.concat([daily.index.to_series(), daily.index.to_series()[::-1]]),
            y=pd.concat([hi_s, lo_s[::-1]]),
            fill="toself",
            fillcolor="rgba(255,127,14,0.12)",
            line=dict(color="rgba(255,255,255,0)"),
            name="90% prediction interval",
            hoverinfo="skip",
        ))

    fig.add_trace(go.Scatter(
        x=sm.index, y=sm["bike_count"],
        name="Actual", line=dict(color="#1f77b4", width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=sm.index, y=sm["predicted"],
        name="Predicted (weather baseline)",
        line=dict(color="#ff7f0e", width=1.5, dash="dot"),
    ))

    for iv in cfg["interventions"]:
        fig.add_vline(
            x=iv["date"].timestamp() * 1000,
            line_dash=iv["dash"], line_color=iv["color"], line_width=2,
            annotation_text=iv["label"],
            annotation_position="top left", annotation_font_size=11,
        )

    fig.update_layout(
        title=f"{site_name} — Actual vs Weather-Normalised Baseline",
        xaxis_title="Date",
        yaxis_title=f"Cyclist count ({smoothing.lower()} avg)",
        legend=dict(orientation="h", y=1.08),
        hovermode="x unified",
        height=420,
        margin=dict(t=80, b=40),
    )
    st.plotly_chart(fig, width="stretch")

# ── Statistical summary ───────────────────────────────────────────────────────

st.markdown("---")
st.subheader("📊 Statistical Summary")

if city == "Aalst":
    st.markdown(
        "**Test: Wilcoxon rank-sum (Mann-Whitney U)**  \n"
        "Tests whether post-intervention daily residuals (actual − predicted) differ "
        "significantly from the pre-intervention residuals.  \n"
        "H₀: the two distributions are identical (no intervention effect)."
    )
    st.markdown("")

    for site_id in selected_ids:
        site_name = cfg["sites"][site_id]
        daily     = site_daily[site_id]

        pre  = daily.loc[daily.index < AALST_INTERVENTION,  "residual"].dropna()
        post = daily.loc[daily.index >= AALST_INTERVENTION, "residual"].dropna()

        if len(pre) < 5 or len(post) < 5:
            st.write(f"**{site_name}**: insufficient data for test.")
            continue

        _, p       = stats.mannwhitneyu(pre, post, alternative="two-sided")
        delta_pct  = (post.mean() - pre.mean()) / abs(pre.mean()) * 100 if pre.mean() != 0 else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"{site_name}",               "",         label_visibility="visible")
        c2.metric("Pre-intervention avg residual",  f"{pre.mean():+.1f} cyclists/day")
        c3.metric("Post-intervention avg residual", f"{post.mean():+.1f} cyclists/day",
                  delta=f"{delta_pct:+.1f}%")
        c4.metric("p-value", f"{p:.4f}")

        if p < 0.05:
            st.success(f"✅ **{site_name}**: statistically significant change (p = {p:.4f})")
        else:
            st.info(f"ℹ️ **{site_name}**: no statistically significant change detected (p = {p:.4f})")
        st.markdown("")

else:  # Kortrijk
    st.markdown(
        "**Test: Mann-Whitney U — Trial phase vs Permanent plan**  \n"
        "Because the monitoring sensors were installed at the start of the trial, there is no "
        "pre-intervention baseline. The test compares daily residuals between the two "
        "observed phases to detect a change in cycling behaviour between them.  \n"
        "H₀: residuals in the trial phase and permanent phase come from the same distribution."
    )
    st.markdown("")

    for site_id in selected_ids:
        site_name = cfg["sites"][site_id]
        daily     = site_daily[site_id]

        trial = daily.loc[
            (daily.index >= KORTRIJK_TRIAL) & (daily.index < KORTRIJK_PERMANENT),
            "residual"
        ].dropna()
        perm = daily.loc[daily.index >= KORTRIJK_PERMANENT, "residual"].dropna()

        # Also show divergence from model baseline for each phase
        trial_div = trial.mean()
        perm_div  = perm.mean()

        if len(trial) < 5 or len(perm) < 5:
            st.write(f"**{site_name}**: insufficient data for test.")
            continue

        _, p      = stats.mannwhitneyu(trial, perm, alternative="two-sided")
        delta_pct = (perm_div - trial_div) / abs(trial_div) * 100 if trial_div != 0 else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"{site_name}", "", label_visibility="visible")
        c2.metric("Trial phase avg residual",     f"{trial_div:+.1f} cyclists/day")
        c3.metric("Permanent phase avg residual", f"{perm_div:+.1f} cyclists/day",
                  delta=f"{delta_pct:+.1f}%")
        c4.metric("p-value (trial vs permanent)", f"{p:.4f}")

        if p < 0.05:
            st.success(f"✅ **{site_name}**: significant change between trial and permanent phase (p = {p:.4f})")
        else:
            st.info(f"ℹ️ **{site_name}**: no significant difference between trial and permanent phase (p = {p:.4f})")
        st.markdown("")
