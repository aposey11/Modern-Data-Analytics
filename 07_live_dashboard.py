import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium
import branca.colormap as cm

# --- CONSTANTS ---
# The absolute average risk score from the Tweedie model training logs
BASELINE_RISK = 0.0067 

# 1. Page Configuration
st.set_page_config(page_title="GTRI Flanders Risk Map", layout="wide")

@st.cache_resource
def load_resources():
    model = joblib.load("lightgbm_accident_model.pkl")
    df_sites = pd.read_parquet("site_spatial_features.parquet")
    if 'long' in df_sites.columns and 'lon' not in df_sites.columns:
        df_sites = df_sites.rename(columns={'long': 'lon'})
    return model, df_sites

try:
    model, sensors = load_resources()
except Exception as e:
    st.error(f"Error loading data or model: {e}")
    st.stop()

# 2. Sidebar Controls
st.sidebar.header("GTRI Control Panel")

st.sidebar.subheader("📅 Temporal Constraints")
month = st.sidebar.select_slider("Month", options=list(range(1, 13)), value=10)
hour = st.sidebar.select_slider("Hour of Day", options=list(range(24)), value=17) 

st.sidebar.divider()
st.sidebar.subheader("🌤️ Weather Simulation")
temp = st.sidebar.slider("Temperature (°C)", -10.0, 35.0, 10.0)
precip = st.sidebar.slider("Precipitation (mm/h)", 0.0, 20.0, 5.0) 
wind = st.sidebar.slider("Wind Speed (km/h)", 0.0, 100.0, 25.0)    
wind_spike = st.sidebar.slider("Sudden Wind Gusts (+km/h)", 0.0, 30.0, 5.0)

st.sidebar.divider()
st.sidebar.subheader("🚲 Traffic Simulation")
bike_vol = st.sidebar.number_input("Avg Hourly Bike Volume", min_value=0, max_value=3000, value=400, step=50)
piek_index = st.sidebar.slider("Peak Index (Clustering Factor)", 1.0, 4.0, 2.0)

st.sidebar.divider()
st.sidebar.subheader("🚨 Dashboard Settings")
# Threshold is a tangible "Multiplier" (e.g., alert if risk is 4x normal)
custom_threshold = st.sidebar.slider("Red Alert Cutoff (x Times Average)", 1.0, 10.0, 4.0, step=0.5)

st.sidebar.divider()
with st.sidebar.expander("ℹ️ How to read these scores"):
    st.markdown("""
    **What is the Baseline?**
    The average risk across all intersections on a normal day is **1.0x**. 
    
    Mathematically, this 1.0x baseline equals an expected accident count of **0.0067** per hour. 
    Because bicycle accidents are rare, this translates to roughly a **0.67% chance** of an accident occurring in that specific 500m zone during that single hour.
    
    **Why Relative Risk?**
    If a severe storm pushes the risk at an intersection to **5.0x**, it means the chance of a crash has spiked from its normal 0.67% up to roughly **3.3%**. While 3.3% sounds small, applied across thousands of cyclists and hundreds of intersections, it indicates a critical hazard zone.
    """)

# 3. Prediction Logic
def get_predictions():
    pred_df = sensors.copy()
    
    pred_df['Hour_Sin'] = np.sin(2 * np.pi * hour / 24)
    pred_df['Hour_Cos'] = np.cos(2 * np.pi * hour / 24)
    pred_df['Month_Sin'] = np.sin(2 * np.pi * month / 12)
    pred_df['Month_Cos'] = np.cos(2 * np.pi * month / 12)
    
    pred_df['Avg_Traffic_Volume'] = bike_vol
    pred_df['Avg_Piek_Index'] = piek_index
    pred_df['Avg_Traffic_Max_Spike'] = bike_vol * (piek_index / 4.0)
    pred_df['Avg_Traffic_Volatility'] = bike_vol * 0.15 
    
    pred_df['Avg_Temperature'] = temp
    pred_df['Avg_Precipitation'] = precip
    pred_df['Avg_Wind_Speed'] = wind
    pred_df['Avg_Temp_Drop'] = 0.0  
    pred_df['Avg_Wind_Spike'] = wind_spike
    
    feature_cols = [
        'Hour_Sin', 'Hour_Cos', 'Month_Sin', 'Month_Cos', 
        'Avg_Traffic_Volume', 'Avg_Traffic_Max_Spike', 'Avg_Traffic_Volatility', 'Avg_Piek_Index',
        'Avg_Temperature', 'Avg_Precipitation', 'Avg_Wind_Speed', 'Avg_Temp_Drop', 'Avg_Wind_Spike',
        'spatial_risk_score', 'historical_accidents_250m', 'historical_accidents_250m_to_500m'
    ]
    
    X = pred_df[feature_cols].copy()
    
    # Calculate Raw Score using Regressor
    pred_df['raw_risk'] = model.predict(X)
    
    # Calculate Relative Risk Multiplier
    pred_df['relative_risk'] = pred_df['raw_risk'] / BASELINE_RISK
    return pred_df

results = get_predictions()

# 4. Dashboard Layout
st.title("🚲 Geo-Temporal Risk Index (GTRI) - Relative Risk Model")

high_risk_count = len(results[results['relative_risk'] >= custom_threshold])
max_relative_risk = results['relative_risk'].max()
avg_relative_risk = results['relative_risk'].mean()

m1, m2, m3 = st.columns(3)
m1.metric(f"Sensors at High Risk (>{custom_threshold}x)", high_risk_count)
m2.metric("Avg. Regional Risk Level", f"{avg_relative_risk:.1f}x Baseline")
m3.metric("Highest Local Risk Spike", f"{max_relative_risk:.1f}x Baseline")

# 5. Folium Map Visualization
# 5. Folium Map Visualization
m = folium.Map(location=[51.00, 4.35], zoom_start=9, tiles="cartodbpositron")

# THE FIX: Dynamically link the colormap to your custom slider!
# Red starts exactly at your custom_threshold.
# Yellow and Orange are mathematically spaced between 1.0x and your threshold.
colormap = cm.StepColormap(
    colors=['#228B22', '#FFD700', '#FF8C00', '#FF0000'], 
    index=[0.0, 1.0, 1.0 + ((custom_threshold - 1.0) * 0.4), custom_threshold],
    vmin=0.0, vmax=max(custom_threshold + 2.0, 10.0),
    caption=f'Relative Risk (Red = Alert >= {custom_threshold}x)'
)

for _, row in results.iterrows():
    if pd.notnull(row['lat']) and pd.notnull(row['lon']):
        rel_risk = row['relative_risk']
        color = colormap(rel_risk)
        
        is_alert = rel_risk >= custom_threshold
        status = "🚨 HIGH ALERT" if is_alert else "✅ Normal"
        
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=12 if is_alert else 6, 
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.9 if is_alert else 0.6,
            popup=folium.Popup(
                f"<b>{row.get('naam_site', 'Unknown Site')}</b><br>"
                f"Relative Risk: <b>{rel_risk:.1f}x</b> higher than average<br>"
                f"Status: {status}<br>"
                f"<hr>"
                f"<i>Spatial Base: {row['spatial_risk_score']:.2f}</i>", 
                max_width=250
            )
        ).add_to(m)

st_folium(m, width=1400, height=700)