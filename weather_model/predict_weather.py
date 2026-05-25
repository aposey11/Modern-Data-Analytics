import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from xgboost import XGBRegressor
import requests
from datetime import datetime, timedelta

_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model():
    model = XGBRegressor()
    model.load_model(os.path.join(_DIR, "weather_bike_model.json"))
    return model

@st.cache_data
def load_sites():
    cols = [
        "site_id", "site_nr", "long", "lat", "naam", "domein",
        "wegnr", "district", "gemeente", "interval", "datum_van"
    ]
    sites = pd.read_csv(
        os.path.join(_DIR, "Data", "sites.csv"),
        names=cols,
        header=None
    )
    sites = sites[["site_id", "lat", "long", "naam", "gemeente", "district", "wegnr"]]
    sites = sites.rename(columns={"long": "lon"})
    return sites

# Fetch 7-day forecast from Open-Meteo; refreshes every hour
@st.cache_data(ttl=3600)
def fetch_forecast(lat=51.05, lon=3.72):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,cloud_cover",
        "forecast_days": 7,
        "timezone": "Europe/Brussels"
    }
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
    return pd.DataFrame({
        "hour_timestamp": pd.to_datetime(data["hourly"]["time"]),
        "temperature":    data["hourly"]["temperature_2m"],
        "humidity":       data["hourly"]["relative_humidity_2m"],
        "precipitation":  data["hourly"]["precipitation"],
        "wind_speed":     data["hourly"]["wind_speed_10m"],
        "cloud_cover":    data["hourly"]["cloud_cover"]
    })


def show():
    model  = load_model()
    sites  = load_sites()

    try:
        forecast = fetch_forecast()
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 429:
            st.warning("The weather forecast API is temporarily rate-limited. Please wait a minute and refresh the page.")
        else:
            st.error(f"Weather API returned an error: {e}")
        st.stop()
    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach the weather forecast API: {e}")
        st.stop()

    st.title("🚴 7-Day Cycling Forecast")

    # Sidebar
    st.sidebar.header("📅 Select date and time")

    today    = datetime.today().date()
    max_date = today + timedelta(days=6)

    selected_date = st.sidebar.date_input(
        "Date",
        value=today,
        min_value=today,
        max_value=max_date
    )
    selected_hour = st.sidebar.slider("Hour of day", 0, 23, 8)

    selected_datetime = (
        pd.Timestamp(datetime.combine(selected_date, datetime.min.time()))
        + timedelta(hours=selected_hour)
    )

    forecast_row = forecast[forecast["hour_timestamp"] == selected_datetime]

    if forecast_row.empty:
        st.error(f"No forecast data available for {selected_datetime}. Try a different date/hour.")
        st.stop()

    weather       = forecast_row.iloc[0]
    temperature   = weather["temperature"]
    humidity      = weather["humidity"]
    precipitation = weather["precipitation"]
    wind_speed    = weather["wind_speed"]
    cloud_cover   = weather["cloud_cover"]

    # Build feature matrix
    features = [
        "lat", "lon", "hour", "day_of_week", "month",
        "temperature", "humidity", "precipitation", "wind_speed", "cloud_cover"
    ]
    X = sites[["lat", "lon"]].copy()
    X["hour"]          = selected_hour
    X["day_of_week"]   = selected_datetime.dayofweek
    X["month"]         = selected_datetime.month
    X["temperature"]   = temperature
    X["humidity"]      = humidity
    X["precipitation"] = precipitation
    X["wind_speed"]    = wind_speed
    X["cloud_cover"]   = cloud_cover
    X = X[features]

    preds = model.predict(X)
    preds = np.clip(preds, 0, None)

    sites_plot = sites.copy()
    sites_plot["predicted_cyclists"] = preds.round(0).astype(int)

    st.subheader(f"Predicted cyclist counts — {selected_datetime.strftime('%A %d %b %Y')}, {selected_hour:02d}:00")

    fig = px.scatter_map(
        sites_plot,
        lat="lat",
        lon="lon",
        size="predicted_cyclists",
        size_max=40,
        zoom=10,
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
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=550)
    st.plotly_chart(fig, width='stretch')

    st.markdown("<div style='margin-top: 80px;'></div>", unsafe_allow_html=True)
    st.subheader("🌤️ Weather conditions for selected hour")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("🌡️ Temperature", f"{temperature:.1f} °C")
    col2.metric("💧 Humidity",     f"{humidity:.0f} %")
    col3.metric("🌧️ Precipitation", f"{precipitation:.1f} mm")
    col4.metric("💨 Wind Speed",   f"{wind_speed:.1f} km/h")
    col5.metric("☁️ Cloud Cover",  f"{cloud_cover:.0f} %")
