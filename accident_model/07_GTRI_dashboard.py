import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
import os
from streamlit_folium import st_folium
import branca.colormap as cm

_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="GTRI Flanders Risk Map", layout="wide")

@st.cache_resource
def load_resources():
    artifacts     = joblib.load(os.path.join(_DIR, "GTRI_model_artifacts.pkl"))
    model         = artifacts['model']
    baseline_risk = artifacts['baseline_risk']
    feature_names = artifacts['feature_names']
    metrics       = artifacts.get('metrics', {})

    df_sites = pd.read_parquet(os.path.join(_DIR, "gtri_site_spatial_features.parquet"))
    if 'long' in df_sites.columns and 'lon' not in df_sites.columns:
        df_sites = df_sites.rename(columns={'long': 'lon'})
    return model, df_sites, baseline_risk, feature_names, metrics

try:
    model, sensors, BASELINE_RISK, FEATURE_NAMES, TRAIN_METRICS = load_resources()
except Exception as e:
    st.error(f"Error loading resources: {e}")
    st.stop()

st.sidebar.header("GTRI Control Panel")

st.sidebar.subheader("Temporal")
month = st.sidebar.select_slider("Month", options=list(range(1, 13)), value=10)
hour  = st.sidebar.select_slider("Hour of Day", options=list(range(24)), value=17)

st.sidebar.divider()
st.sidebar.subheader("Weather — Current Hour")
temp      = st.sidebar.slider("Temperature (°C)", -10.0, 35.0, 10.0)
prev_temp = st.sidebar.slider("Temperature — Previous Hour (°C)", -10.0, 35.0, 10.0)
temp_drop = max(0.0, prev_temp - temp)

precip      = st.sidebar.slider("Precipitation — Current Hour (mm/h)", 0.0, 20.0, 5.0)
prev_precip = st.sidebar.slider("Precipitation — Previous Hour (mm/h)", 0.0, 20.0, 1.0)
wind       = st.sidebar.slider("Wind Speed (km/h)", 0.0, 100.0, 25.0)
wind_spike = st.sidebar.slider("Wind Gusts Delta (km/h)", 0.0, 30.0, 5.0)

rain_surge_ratio = float(np.clip(precip / (prev_precip + 1e-6), 1.0, 20.0))
rain_surge       = float(np.log1p(precip) - np.log1p(prev_precip))

st.sidebar.divider()
st.sidebar.subheader("Traffic")
bike_vol      = st.sidebar.number_input("Avg Hourly Bike Volume", min_value=0, max_value=3000, value=400, step=50)
prev_bike_vol = st.sidebar.number_input("Previous Hour Bike Volume", min_value=0, max_value=3000, value=350, step=50)
traffic_delta = float(bike_vol - prev_bike_vol)

piek_index    = st.sidebar.slider("Traffic Clustering Factor", 1.0, 4.0, 2.0,
                                   help="How bunched up cyclist arrivals are within the hour. Higher = more spikes.")
weekend_ratio = st.sidebar.slider("Weekend Traffic Ratio", 0.0, 1.0, 0.3, step=0.05,
                                   help="0 = weekday commuter usage, 1 = weekend leisure usage.")

st.sidebar.divider()
st.sidebar.subheader("Alert Settings")
custom_threshold = st.sidebar.slider("Red Alert Cutoff (x Times Average)", 1.0, 10.0, 4.0, step=0.5)


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
    pred_df['Rain_Mean_Intensity'] = prev_precip
    pred_df['Rain_Surge_Ratio']    = rain_surge_ratio
    pred_df['Rain_Lag1h_Mean']     = prev_precip
    pred_df['Rain_Lag1h_Max']      = prev_precip
    pred_df['Rain_Surge']          = rain_surge

    X = pred_df[FEATURE_NAMES].copy()
    pred_df['raw_risk']      = model.predict(X)
    pred_df['relative_risk'] = pred_df['raw_risk'] / BASELINE_RISK
    return pred_df

results = get_predictions()

st.title("Geo-Temporal Risk Index (GTRI) — Flanders Cycling Risk Map")

with st.expander("Model Performance (2024 hold-out test set)", expanded=False):
    if TRAIN_METRICS:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tweedie Deviance",   f"{TRAIN_METRICS.get('tweedie_deviance', 'N/A'):.4f}")
        c2.metric("Spearman rho",       f"{TRAIN_METRICS.get('spearman_rho', 'N/A'):.4f}")
        c3.metric("Top-Decile Capture", f"{TRAIN_METRICS.get('top_decile_pct', 'N/A'):.1f}%")
        c4.metric("RMSE",               f"{TRAIN_METRICS.get('rmse', 'N/A'):.5f}")

high_risk_count   = len(results[results['relative_risk'] >= custom_threshold])
max_relative_risk = results['relative_risk'].max()
avg_relative_risk = results['relative_risk'].mean()

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

view = st.segmented_control(
    "View",
    ["🗺️ Risk Map", "📊 Site Rankings"],
    default="🗺️ Risk Map",
    label_visibility="collapsed"
)

if view == "🗺️ Risk Map":
    fmap = folium.Map(location=[51.00, 4.35], zoom_start=9, tiles="cartodbpositron")

    GREEN_CUTOFF = 1.25
    orange_start = GREEN_CUTOFF * 2
    LEGEND_MAX   = max(custom_threshold + 2.0, 10.0)

    colormap = cm.StepColormap(
        colors=['#228B22', '#FFD700', '#FF8C00', '#FF0000'],
        index=[0.0, GREEN_CUTOFF, orange_start, custom_threshold],
        vmin=0.0, vmax=LEGEND_MAX,
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
            ).add_to(fmap)

    _w1 = GREEN_CUTOFF / LEGEND_MAX * 100
    _w2 = (orange_start - GREEN_CUTOFF) / LEGEND_MAX * 100
    _w3 = (custom_threshold - orange_start) / LEGEND_MAX * 100
    _w4 = (LEGEND_MAX - custom_threshold) / LEGEND_MAX * 100

    _legend_html = (
        f'<div style="position:absolute;top:12px;right:10px;z-index:9999;'
        f'background:white;padding:8px 12px 6px 12px;border-radius:5px;'
        f'border:1px solid rgba(0,0,0,0.25);font-size:11px;'
        f'font-family:Arial,sans-serif;box-shadow:0 1px 5px rgba(0,0,0,0.15);min-width:220px;">'
        f'<div style="font-weight:600;margin-bottom:5px;">Relative Risk'
        f'&nbsp;&nbsp;<span style="color:#888;font-weight:400;">(Red ≥ {custom_threshold}×)</span></div>'
        f'<div style="display:flex;height:14px;border-radius:2px;overflow:hidden;margin-bottom:3px;">'
        f'<div style="width:{_w1:.1f}%;background:#228B22;"></div>'
        f'<div style="width:{_w2:.1f}%;background:#FFD700;"></div>'
        f'<div style="width:{_w3:.1f}%;background:#FF8C00;"></div>'
        f'<div style="width:{_w4:.1f}%;background:#FF0000;"></div>'
        f'</div>'
        f'<div style="position:relative;height:14px;">'
        f'<span style="position:absolute;left:0;">0</span>'
        f'<span style="position:absolute;left:{_w1:.1f}%;transform:translateX(-50%);">{GREEN_CUTOFF}</span>'
        f'<span style="position:absolute;left:{(_w1+_w2):.1f}%;transform:translateX(-50%);">{orange_start}</span>'
        f'<span style="position:absolute;left:{(_w1+_w2+_w3):.1f}%;transform:translateX(-50%);">{custom_threshold}</span>'
        f'<span style="position:absolute;right:0;">{LEGEND_MAX:.0f}+</span>'
        f'</div></div>'
    )
    fmap.get_root().html.add_child(folium.Element(_legend_html))
    st_folium(fmap, width=1400, height=700)

else:
    st.subheader("Site Risk Rankings")
    st.caption("Rankings update live with the sidebar sliders.")

    display_cols = {
        "naam_site":                 "Site",
        "relative_risk":             "Risk (×baseline)",
        "historical_accidents_250m": "Accidents ≤250 m",
        "spatial_risk_score":        "Spatial score",
    }

    col_high, col_low = st.columns(2)

    with col_high:
        st.markdown("#### 🔴 Top 10 Highest Risk")
        top10 = (
            results.nlargest(10, "relative_risk")
            [list(display_cols.keys())]
            .rename(columns=display_cols)
            .reset_index(drop=True)
        )
        top10.index = top10.index + 1
        top10["Risk (×baseline)"] = top10["Risk (×baseline)"].map(lambda x: f"{x:.2f}×")
        top10["Spatial score"]    = top10["Spatial score"].map(lambda x: f"{x:.3f}")
        st.dataframe(
            top10,
            width='stretch',
            column_config={
                "Site":             st.column_config.TextColumn("Site", width="medium"),
                "Risk (×baseline)": st.column_config.TextColumn("Risk (×avg)", width="small"),
                "Accidents ≤250 m": st.column_config.NumberColumn("Acc. ≤250 m", width="small"),
                "Spatial score":    st.column_config.TextColumn("Spatial", width="small"),
            },
        )

    with col_low:
        st.markdown("#### 🟢 Top 10 Lowest Risk")
        bot10 = (
            results.nsmallest(10, "relative_risk")
            [list(display_cols.keys())]
            .rename(columns=display_cols)
            .reset_index(drop=True)
        )
        bot10.index = bot10.index + 1
        bot10["Risk (×baseline)"] = bot10["Risk (×baseline)"].map(lambda x: f"{x:.2f}×")
        bot10["Spatial score"]    = bot10["Spatial score"].map(lambda x: f"{x:.3f}")
        st.dataframe(
            bot10,
            width='stretch',
            column_config={
                "Site":             st.column_config.TextColumn("Site", width="medium"),
                "Risk (×baseline)": st.column_config.TextColumn("Risk (×avg)", width="small"),
                "Accidents ≤250 m": st.column_config.NumberColumn("Acc. ≤250 m", width="small"),
                "Spatial score":    st.column_config.TextColumn("Spatial", width="small"),
            },
        )
