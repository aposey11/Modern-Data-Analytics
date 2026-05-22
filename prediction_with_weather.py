import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
import requests
from datetime import datetime, timedelta

# Page config
st.set_page_config(page_title="7-Day Cycling Forecast", layout="wide")
st.title("🚴 7-Day Cycling Forecast")

# Load model
@st.cache_resource
def load_model():
    with open(r"C:\Users\Admin\vscode\python\cycling project\weather_bike_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()


# Load sites
@st.cache_data
def load_sites():
    cols = [
        "site_id", "site_nr", "long", "lat", "naam", "domein",
        "wegnr", "district", "gemeente", "interval", "datum_van"
    ]
    sites = pd.read_csv(
        r"C:\Users\Admin\Desktop\Master Stats\Modern Analytics & Python\Weather Regression\sites.csv",
        names=cols,
        header=None
    )
    sites = sites[["site_id", "lat", "long", "naam", "gemeente", "district", "wegnr"]]
    sites = sites.rename(columns={"long": "lon"})
    return sites

sites = load_sites()

# Fetch weather prediction (7days forecast) (uses forecast api instead of archive api)
# Kinda same structure as the historical data but this time with prediction
@st.cache_data(ttl=3600)  # streamlit will refresh automatically every hour 
def fetch_forecast(lat=51.05, lon=3.72):
    """
    Fetch 7-day hourly forecast from Open-Meteo.
    Coordinates default to Ghent (central Flanders).
    """
    
    url = "https://api.open-meteo.com/v1/forecast"
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,cloud_cover",
        "forecast_days": 7,
        "timezone": "Europe/Brussels"
    }
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    
    data = response.json()
    
    forecast = pd.DataFrame({
        "hour_timestamp": pd.to_datetime(data["hourly"]["time"]),
        "temperature": data["hourly"]["temperature_2m"],
        "humidity": data["hourly"]["relative_humidity_2m"],
        "precipitation": data["hourly"]["precipitation"],
        "wind_speed": data["hourly"]["wind_speed_10m"],
        "cloud_cover": data["hourly"]["cloud_cover"]
    })
    
    return forecast

forecast = fetch_forecast()

# Build sidebar
st.sidebar.header("📅 Select date and time")

# Define the valid date range (today through today + 6 days)
today = datetime.today().date()
max_date = today + timedelta(days=6)

# Date picker restricted to the forecast window
selected_date = st.sidebar.date_input(
    "Date",
    value=today,
    min_value=today,
    max_value=max_date
)

# Hour slider
selected_hour = st.sidebar.slider("Hour of day", 0, 23, 8)

# Filter forecast on selected date/hour
# Combine selected date and hour into a single timestamp
selected_datetime = pd.Timestamp(datetime.combine(selected_date, datetime.min.time())) + timedelta(hours=selected_hour)

# Filter forecast to that exact hour
forecast_row = forecast[forecast["hour_timestamp"] == selected_datetime]

# Handle case where the hour hasn't arrived yet in the forecast- app stays visible
#  display error instead of other confusing things (for example time-zone mismatches); no predictions with missing data , shows for which hr not visible
if forecast_row.empty:
    st.error(f"No forecast data available for {selected_datetime}. Try a different date/hour.")
    st.stop()

# Extract weather values for this hour
weather = forecast_row.iloc[0]
temperature = weather["temperature"]
humidity = weather["humidity"]
precipitation = weather["precipitation"]
wind_speed = weather["wind_speed"]
cloud_cover = weather["cloud_cover"]


# Build the reature matrix 
# Define feature order (must match what the model was trained on)
features = [
    "site_id", "lat", "lon", "hour", "day_of_week", "month",
    "temperature", "humidity", "precipitation", "wind_speed", "cloud_cover"
]

# Start with site info
X = sites[["site_id", "lat", "lon"]].copy()

# Add time features derived from the selected datetime
X["hour"] = selected_hour
X["day_of_week"] = selected_datetime.dayofweek
X["month"] = selected_datetime.month

# Add weather values from the forecast
X["temperature"] = temperature
X["humidity"] = humidity
X["precipitation"] = precipitation
X["wind_speed"] = wind_speed
X["cloud_cover"] = cloud_cover

# Reorder to match model's expected input
X = X[features]

# same structure as slider, just predictions this time 

# Predict and display map 
# Run predictions
preds = model.predict(X)

# Clip negative values to zero
preds = np.clip(preds, 0, None)

# Attach predictions to sites for plotting
sites_plot = sites.copy()
sites_plot["predicted_cyclists"] = preds.round(0).astype(int)

# Map title with selected datetime
st.subheader(f"Predicted cyclist counts — {selected_datetime.strftime('%A %d %b %Y')}, {selected_hour:02d}:00")

# Build the map
fig = px.scatter_map(
    sites_plot,
    lat="lat",
    lon="lon",
    size="predicted_cyclists",
    size_max=40,
    zoom=9,
    hover_name="naam",
    hover_data={
        "predicted_cyclists": True,
        "gemeente": True,
        "district": True,
        "wegnr": True,
        "site_id": True,
        "lat": False,
        "lon": False
    },
    labels={"predicted_cyclists": "Cyclists"},
    map_style="carto-darkmatter"
)

fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=600)
st.plotly_chart(fig, use_container_width=True)

#push down
st.markdown("<div style='margin-top: 80px;'></div>", unsafe_allow_html=True)
# To see why counts are high/low 
st.subheader("🌤️ Weather conditions for selected hour")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("🌡️ Temperature", f"{temperature:.1f} °C")
col2.metric("💧 Humidity", f"{humidity:.0f} %")
col3.metric("🌧️ Precipitation", f"{precipitation:.1f} mm")
col4.metric("💨 Wind Speed", f"{wind_speed:.1f} km/h")
col5.metric("☁️ Cloud Cover", f"{cloud_cover:.0f} %")