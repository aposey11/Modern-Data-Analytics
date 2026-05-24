import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium
import branca.colormap as cm

st.set_page_config(page_title="GTRI Flanders Risk Map", layout="wide")

@st.cache_resource
def load_resources():
    artifacts     = joblib.load("GTRI_model_artifacts.pkl")
    model         = artifacts['model']
    baseline_risk = artifacts['baseline_risk']
    feature_names = artifacts['feature_names']
    metrics       = artifacts.get('metrics', {})

    df_sites = pd.read_parquet("gtri_site_spatial_features.parquet")
    if 'long' in df_sites.columns and 'lon' not in df_sites.columns:
        df_sites = df_sites.rename(columns={'long': 'lon'})
    return model, df_sites, baseline_risk, feature_names, metrics

try:
    model, sensors, BASELINE_RISK, FEATURE_NAMES, TRAIN_METRICS = load_resources()
except Exception as e:
    st.error(f"Error loading resources: {e}")
    st.stop()

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.header("GTRI Control Panel")

st.sidebar.subheader("Temporal")
month = st.sidebar.select_slider("Month", options=list(range(1, 13)), value=10)
hour  = st.sidebar.select_slider("Hour of Day", options=list(range(24)), value=17)

st.sidebar.divider()
st.sidebar.subheader("Weather — Current Hour")
temp      = st.sidebar.slider("Temperature (°C)", -10.0, 35.0, 10.0)
prev_temp = st.sidebar.slider("Temperature — Previous Hour (°C)", -10.0, 35.0, 10.0,
                               help="Used to compute the temperature drop into this hour.")
temp_drop = max(0.0, prev_temp - temp)

precip      = st.sidebar.slider("Precipitation — Current Hour (mm/h)", 0.0, 20.0, 5.0)
prev_precip = st.sidebar.slider("Precipitation — Previous Hour (mm/h)", 0.0, 20.0, 1.0,
                                  help="Higher than current = rain is easing. "
                                       "Lower than current = sudden onset.")
wind       = st.sidebar.slider("Wind Speed (km/h)", 0.0, 100.0, 25.0)
wind_spike = st.sidebar.slider("Wind Gusts Delta (km/h)", 0.0, 30.0, 5.0)

# Rain surge using log-difference: bounded to roughly -3 to +3 for typical precipitation
# values. Positive = rain intensifying this hour; negative = easing; near zero = steady.
rain_surge_ratio = float(np.clip(precip / (prev_precip + 1e-6), 1.0, 20.0))
rain_surge    = float(np.log1p(precip) - np.log1p(prev_precip))

st.sidebar.divider()
st.sidebar.subheader("Traffic")
bike_vol      = st.sidebar.number_input("Avg Hourly Bike Volume", min_value=0, max_value=3000, value=400, step=50)
prev_bike_vol = st.sidebar.number_input("Previous Hour Bike Volume", min_value=0, max_value=3000, value=350, step=50,
                                         help="Used to compute traffic growth into this hour.")
traffic_delta = float(bike_vol - prev_bike_vol)

piek_index    = st.sidebar.slider("Traffic Clustering Factor", 1.0, 4.0, 2.0,
                                   help="Controls how concentrated cyclist volume is within "
                                        "the hour. A higher value means more cyclists arrive "
                                        "in short bursts rather than spread evenly. "
                                        "Used to derive the max traffic spike fed to the model: "
                                        "Max Spike = Avg Volume × (Clustering Factor / 4).")
weekend_ratio = st.sidebar.slider("Weekend Traffic Ratio", 0.0, 1.0, 0.3, step=0.05,
                                   help="0 = weekday commuter usage, 1 = weekend leisure usage.")

st.sidebar.divider()
st.sidebar.subheader("Alert Settings")
custom_threshold = st.sidebar.slider("Red Alert Cutoff (x Times Average)", 1.0, 10.0, 4.0, step=0.5)

st.sidebar.divider()
with st.sidebar.expander("How to read these scores"):
    st.markdown(f"""
    **Baseline risk**: {BASELINE_RISK:.5f} expected accidents per site-hour, computed as
    the mean model prediction across the full dataset. All site scores are expressed
    relative to this value — a score of 3.0x means the predicted accident rate is three
    times the average.

    **Rain Surge**: log-difference of current vs previous hour precipitation.
    Positive = rain intensifying (sudden onset); negative = rain easing off;
    near zero = steady conditions. The log scale avoids saturation when the
    previous hour is dry.

    **Traffic delta**: positive = cyclists building up into this hour (e.g. rush hour
    start); negative = traffic declining (e.g. rush hour end).
    """)

# ==========================================
# PREDICTION
# ==========================================
def get_predictions():
    pred_df = sensors.copy()

    pred_df['Hour_Sin']  = np.sin(2 * np.pi * hour  / 24)
    pred_df['Hour_Cos']  = np.cos(2 * np.pi * hour  / 24)
    pred_df['Month_Sin'] = np.sin(2 * np.pi * month / 12)
    pred_df['Month_Cos'] = np.cos(2 * np.pi * month / 12)

    pred_df['Avg_Traffic_Volume']        = bike_vol
    pred_df['Avg_Traffic_Max_Spike']     = bike_vol * (piek_index / 4.0)
    pred_df['Avg_Traffic_Volatility']    = bike_vol * 0.15
    pred_df['Traffic_Weekend_Ratio']     = weekend_ratio
    pred_df['Traffic_Volume_Delta_Mean'] = traffic_delta

    pred_df['Avg_Temperature']    = temp
    pred_df['Avg_Precipitation']  = precip
    pred_df['Avg_Wind_Speed']     = wind
    pred_df['Avg_Temp_Drop']      = temp_drop
    pred_df['Avg_Wind_Spike']     = wind_spike
    pred_df['Rain_Mean_Intensity'] = prev_precip       # mean level approximated by previous-hour value
    pred_df['Rain_Surge_Ratio']    = rain_surge_ratio
    pred_df['Rain_Lag1h_Mean']     = prev_precip
    pred_df['Rain_Lag1h_Max']      = prev_precip
    pred_df['Rain_Surge']          = rain_surge

    X = pred_df[FEATURE_NAMES].copy()
    pred_df['raw_risk']      = model.predict(X)
    pred_df['relative_risk'] = pred_df['raw_risk'] / BASELINE_RISK
    return pred_df

results = get_predictions()

# ==========================================
# DASHBOARD LAYOUT
# ==========================================
st.title("Geo-Temporal Risk Index (GTRI) — Flanders Cycling Risk Map")

with st.expander("Model Performance (2024 hold-out test set)", expanded=False):
    if TRAIN_METRICS:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tweedie Deviance",     f"{TRAIN_METRICS.get('tweedie_deviance', 'N/A'):.4f}",
                  help="Primary scoring rule for the Tweedie objective. Lower is better.")
        c2.metric("Spearman rho",         f"{TRAIN_METRICS.get('spearman_rho', 'N/A'):.4f}",
                  help="Rank correlation between predicted risk and actual accident counts "
                       "across the 2024 test set. Measures risk ordering quality.")
        c3.metric("Top-Decile Capture",   f"{TRAIN_METRICS.get('top_decile_pct', 'N/A'):.1f}%",
                  help="Percentage of all 2024 accidents that fell in the top 10% of "
                       "site-hours ranked by predicted risk. Random baseline = 10%.")
        c4.metric("RMSE",                 f"{TRAIN_METRICS.get('rmse', 'N/A'):.5f}",
                  help="Root mean squared error on accident counts (2024 test set).")

high_risk_count   = len(results[results['relative_risk'] >= custom_threshold])
max_relative_risk = results['relative_risk'].max()
avg_relative_risk = results['relative_risk'].mean()

# Rain onset interpretation based on log-difference scale
if rain_surge > 1.5:
    rain_label = "Sudden downpour"
elif rain_surge > 0.5:
    rain_label = "Intensifying rain"
elif rain_surge < -0.5:
    rain_label = "Rain easing off"
else:
    rain_label = "Steady conditions"

traffic_label = (
    f"Traffic {'building' if traffic_delta >= 0 else 'declining'} "
    f"({traffic_delta:+.0f} cyclists/h)"
)

m1, m2, m3, m4 = st.columns(4)
m1.metric(f"Sites at High Risk (>{custom_threshold}x)", high_risk_count)
m2.metric("Avg. Regional Risk",  f"{avg_relative_risk:.1f}x")
m3.metric("Highest Risk Spike",  f"{max_relative_risk:.1f}x")
m4.metric("Temp Drop",           f"-{temp_drop:.1f} C" if temp_drop > 0 else "0.0 C")

st.caption(
    f"{rain_label} (Surge = {rain_surge:.2f})  |  {traffic_label}  |  "
    f"Baseline: {BASELINE_RISK:.5f}"
)

# ==========================================
# FOLIUM MAP
# ==========================================
m = folium.Map(location=[51.00, 4.35], zoom_start=9, tiles="cartodbpositron")

colormap = cm.StepColormap(
    colors=['#228B22', '#FFD700', '#FF8C00', '#FF0000'],
    index=[0.0, 1.0, 1.0 + ((custom_threshold - 1.0) * 0.4), custom_threshold],
    vmin=0.0, vmax=max(custom_threshold + 2.0, 10.0),
    caption=f'Relative Risk (Red >= {custom_threshold}x)'
)

for _, row in results.iterrows():
    if pd.notnull(row['lat']) and pd.notnull(row['lon']):
        rel_risk = row['relative_risk']
        color    = colormap(rel_risk)
        is_alert = rel_risk >= custom_threshold

        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=12 if is_alert else 6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.9 if is_alert else 0.6,
            popup=folium.Popup(
                f"<b>{row.get('naam_site', 'Unknown Site')}</b><br>"
                f"Relative Risk: <b>{rel_risk:.1f}x</b><br>"
                f"Status: {'HIGH ALERT' if is_alert else 'Normal'}<br>"
                f"<hr>"
                f"<i>Spatial Score: {row['spatial_risk_score']:.3f}</i><br>"
                f"<i>Accidents 250m (train): {int(row['historical_accidents_250m'])}</i>",
                max_width=260
            )
        ).add_to(m)

colormap.add_to(m)
st_folium(m, width=1400, height=700)
